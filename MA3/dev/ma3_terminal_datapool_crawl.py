#!/usr/bin/env python3
"""ma3-terminal-datapool-crawl: Crawl MA3 DataPool through the native terminal app.
Exists because raw MA object discovery must come from the MA terminal/CLI surface.
Connects the local grandMA3 terminal app to artifact capture for full DataPool traversal.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path

APP_TERMINAL = Path("/Applications/grandMA3.app/Contents/MacOS/gma3_2.3.2/app_terminal")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CHILD_RE = re.compile(r'^\s+#(?P<index>\d+): Name = "(?P<name>.*)", Class = "(?P<class>.*)"$')
PROPERTY_RE = re.compile(
    r'^\s+(?P<name>[A-Z0-9_]+) = "(?P<value>.*)"(?P<readonly> \(Read Only\))?$'
)
CHILDREN_ORDER_RE = re.compile(r"^(?P<ordinal>\d+)\|(?P<class>[^|]*)\|(?P<name>.*)$")
PROMPT_RE = re.compile(r"[^\n>]*>\s*$")
PROMPT_PREFIX_RE = re.compile(r"^[^>\n]*>(?=(Name:|Class:|Path:|Properties:|Children:))")


@dataclass
class CrawlNode:
    """One dumped MA object plus parsed metadata."""

    path: list[int]
    expression: str
    dump_text: str
    name: str
    class_name: str
    object_path: str
    properties: list[dict[str, object]]
    children: list[dict[str, object]]


@dataclass
class DumpFields:
    """Structured fields parsed from one MA object dump."""

    name: str
    class_name: str
    object_path: str
    properties: list[dict[str, object]]
    children: list[dict[str, object]]


class MA3TerminalSession:
    """Interactive session wrapper around the native grandMA3 terminal app."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        timeout_seconds: float = 10.0,
        quiet_period_seconds: float = 0.6,
    ) -> None:
        self._host = str(host)
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._quiet_period_seconds = max(0.1, float(quiet_period_seconds))
        self._proc: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self.transcript: list[str] = []

    def __enter__(self) -> "MA3TerminalSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                [str(APP_TERMINAL)],
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)
        self._proc = proc
        self._master_fd = master_fd
        self._read_until_prompt()
        self.send_command(f"cmdline {self._host}")

    def close(self) -> None:
        proc = self._proc
        master_fd = self._master_fd
        self._proc = None
        self._master_fd = None
        if proc is not None:
            try:
                self.send_command("exit")
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass

    def send_command(self, command: str) -> str:
        """Send one terminal command and return the output without echoed prompt lines."""

        if self._master_fd is None:
            raise RuntimeError("Terminal session is not started")
        os.write(self._master_fd, command.encode("utf-8") + b"\n")
        raw_output = self._read_until_quiet()
        cleaned = _strip_ansi(raw_output)
        self.transcript.append(cleaned)
        return _trim_command_response(cleaned, command)

    def _read_until_prompt(self) -> str:
        if self._master_fd is None:
            raise RuntimeError("Terminal session is not started")
        chunks: list[bytes] = []
        deadline = time.monotonic() + self._timeout_seconds
        prompt_seen_at: float | None = None
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._master_fd], [], [], 0.2)
            if not ready:
                text = _strip_ansi(b"".join(chunks).decode("utf-8", errors="replace"))
                if PROMPT_RE.search(text):
                    if prompt_seen_at is None:
                        prompt_seen_at = time.monotonic()
                    elif time.monotonic() - prompt_seen_at >= self._quiet_period_seconds:
                        return text
                else:
                    prompt_seen_at = None
                continue
            chunk = os.read(self._master_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            text = _strip_ansi(b"".join(chunks).decode("utf-8", errors="replace"))
            if PROMPT_RE.search(text):
                prompt_seen_at = time.monotonic()
            else:
                prompt_seen_at = None
        text = _strip_ansi(b"".join(chunks).decode("utf-8", errors="replace"))
        if prompt_seen_at is not None:
            return text
        raise TimeoutError(f"Timed out waiting for terminal prompt. Last output:\n{text}")

    def _read_until_quiet(self) -> str:
        if self._master_fd is None:
            raise RuntimeError("Terminal session is not started")
        chunks: list[bytes] = []
        deadline = time.monotonic() + self._timeout_seconds
        last_data_at: float | None = None
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._master_fd], [], [], 0.2)
            if ready:
                chunk = os.read(self._master_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
                last_data_at = time.monotonic()
                continue
            if (
                last_data_at is not None
                and time.monotonic() - last_data_at >= self._quiet_period_seconds
            ):
                return _strip_ansi(b"".join(chunks).decode("utf-8", errors="replace"))
        text = _strip_ansi(b"".join(chunks).decode("utf-8", errors="replace"))
        if text.strip():
            return text
        raise TimeoutError(f"Timed out waiting for terminal output. Last output:\n{text}")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "")


def _trim_command_response(text: str, command: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip() == command.strip():
        lines.pop(0)
    while lines and PROMPT_RE.fullmatch(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def datapool_expression(path: list[int]) -> str:
    """Return a Lua expression for one DataPool object handle."""

    return "DataPool()" + "".join(f"[{index}]" for index in path)


def dump_expression(path: list[int]) -> str:
    """Return the terminal Lua dump command for one path."""

    return f'Lua "{datapool_expression(path)}:Dump()"'


def parse_dump_text(dump_text: str) -> DumpFields:
    """Parse raw MA dump text into structured fields."""

    name = ""
    class_name = ""
    object_path = ""
    properties: list[dict[str, object]] = []
    children: list[dict[str, object]] = []
    in_properties = False
    in_children = False

    for raw_line in dump_text.splitlines():
        line = PROMPT_PREFIX_RE.sub("", raw_line.rstrip())
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Class:"):
            class_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Path:"):
            object_path = line.split(":", 1)[1].strip()
            continue
        if line == "Properties:":
            in_properties = True
            in_children = False
            continue
        if line == "Children:":
            in_properties = False
            in_children = True
            continue
        if in_properties:
            match = PROPERTY_RE.match(line)
            if match is None:
                continue
            properties.append(
                {
                    "name": match.group("name"),
                    "value": match.group("value"),
                    "read_only": bool(match.group("readonly")),
                }
            )
            continue
        if in_children:
            match = CHILD_RE.match(line)
            if match is None:
                continue
            children.append(
                {
                    "index": int(match.group("index")),
                    "name": match.group("name"),
                    "class": match.group("class"),
                }
            )

    return DumpFields(
        name=name,
        class_name=class_name,
        object_path=object_path,
        properties=properties,
        children=children,
    )


def parse_dump(path: list[int], dump_text: str) -> CrawlNode:
    """Parse the MA dump text into a structured node."""

    fields = parse_dump_text(dump_text)
    return CrawlNode(
        path=list(path),
        expression=datapool_expression(path),
        dump_text=dump_text,
        name=fields.name,
        class_name=fields.class_name,
        object_path=fields.object_path,
        properties=fields.properties,
        children=fields.children,
    )


def build_children_probe_command(expression: str) -> str:
    """Return a Lua command that prints ordered :Children() results."""

    return (
        'Lua "'
        f"local c={expression}:Children(); "
        "local found=0; "
        "for i=1,512 do "
        "local h=c[i]; "
        "if h then "
        "found=found+1; "
        "Printf(i..'|'..h:GetClass()..'|'..h.name); "
        "end; "
        "end; "
        "Printf('__CHILDREN_COUNT__\\t'..found)\""
    )


def parse_children_probe_output(output_text: str) -> list[dict[str, object]]:
    """Parse ordered :Children() probe output."""

    children: list[dict[str, object]] = []
    for raw_line in output_text.splitlines():
        line = raw_line.rstrip().strip()
        if ">" in line and not CHILDREN_ORDER_RE.match(line):
            _, suffix = line.split(">", 1)
            candidate = suffix.strip()
            if CHILDREN_ORDER_RE.match(candidate):
                line = candidate
        match = CHILDREN_ORDER_RE.match(line)
        if match is None:
            continue
        children.append(
            {
                "ordinal": int(match.group("ordinal")),
                "class": match.group("class"),
                "name": match.group("name"),
            }
        )
    return children


def probe_children_order(
    session: MA3TerminalSession,
    *,
    expression: str,
) -> list[dict[str, object]]:
    """Return ordered live children from :Children() for one expression."""

    output = session.send_command(build_children_probe_command(expression))
    return parse_children_probe_output(output)


def crawl_datapool(
    session: MA3TerminalSession, *, max_nodes: int | None = None
) -> list[CrawlNode]:
    """Recursively dump DataPool and its descendants through the native terminal."""

    queue: deque[list[int]] = deque([[]])
    seen: set[tuple[int, ...]] = set()
    nodes: list[CrawlNode] = []

    while queue:
        path = queue.popleft()
        path_key = tuple(path)
        if path_key in seen:
            continue
        seen.add(path_key)

        response = session.send_command(dump_expression(path))
        node = parse_dump(path, response)
        nodes.append(node)
        if max_nodes is not None and len(nodes) >= max(0, int(max_nodes)):
            break

        for child in node.children:
            child_index = int(child["index"])
            queue.append(path + [child_index])

    return nodes


def crawl_datapool_with_progress(
    session: MA3TerminalSession,
    *,
    max_nodes: int | None = None,
    progress_every: int = 1,
    checkpoint_every: int = 0,
    checkpoint_callback=None,
) -> list[CrawlNode]:
    """Recursively dump DataPool with simple progress prints."""

    queue: deque[list[int]] = deque([[]])
    seen: set[tuple[int, ...]] = set()
    nodes: list[CrawlNode] = []

    while queue:
        path = queue.popleft()
        path_key = tuple(path)
        if path_key in seen:
            continue
        seen.add(path_key)

        expression = datapool_expression(path)
        print(f"CRAWL {len(nodes) + 1}: {expression}", flush=True)
        response = session.send_command(dump_expression(path))
        node = parse_dump(path, response)
        nodes.append(node)
        if progress_every > 0 and len(nodes) % progress_every == 0:
            print(
                f"DONE {len(nodes)}: {expression} class={node.class_name} children={len(node.children)}",
                flush=True,
            )
        if (
            checkpoint_every > 0
            and checkpoint_callback is not None
            and len(nodes) % checkpoint_every == 0
        ):
            checkpoint_callback(nodes)
        if max_nodes is not None and len(nodes) >= max(0, int(max_nodes)):
            break

        for child in node.children:
            child_index = int(child["index"])
            queue.append(path + [child_index])

    return nodes


def write_artifacts(
    output_dir: Path, *, nodes: list[CrawlNode], transcript: list[str]
) -> dict[str, Path]:
    """Write crawl artifacts to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "datapool_terminal_crawl.json"
    markdown_path = output_dir / "datapool_terminal_hierarchy.md"
    transcript_path = output_dir / "terminal_transcript.txt"

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "node_count": len(nodes),
        "nodes": [
            {
                "path": node.path,
                "expression": node.expression,
                "name": node.name,
                "class": node.class_name,
                "object_path": node.object_path,
                "properties": node.properties,
                "children": node.children,
                "dump_text": node.dump_text,
            }
            for node in nodes
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# MA3 DataPool Terminal Crawl",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Node count: `{len(nodes)}`",
        "",
    ]
    for node in nodes:
        indent = "  " * len(node.path)
        label = node.name or node.expression
        lines.append(
            f"{indent}- `{node.expression}` [{node.class_name or 'Unknown'}] "
            f"name=`{label}` children={len(node.children)} properties={len(node.properties)}"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    transcript_path.write_text("\n\n".join(transcript), encoding="utf-8")
    return {
        "json": json_path,
        "markdown": markdown_path,
        "transcript": transcript_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl grandMA3 DataPool using the native terminal app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ma3-terminal-crawl" / "latest",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the full native DataPool crawl and write artifacts."""

    args = _build_parser().parse_args(argv)
    nodes: list[CrawlNode] = []
    files: dict[str, Path] = {}
    session: MA3TerminalSession | None = None
    try:
        with MA3TerminalSession(
            host=args.host, timeout_seconds=args.timeout_seconds
        ) as active_session:
            session = active_session

            def checkpoint_callback(current_nodes: list[CrawlNode]) -> None:
                write_artifacts(
                    args.output_dir, nodes=current_nodes, transcript=active_session.transcript
                )

            nodes = crawl_datapool_with_progress(
                active_session,
                max_nodes=args.max_nodes,
                progress_every=args.progress_every,
                checkpoint_every=args.checkpoint_every,
                checkpoint_callback=checkpoint_callback,
            )
            files = write_artifacts(
                args.output_dir, nodes=nodes, transcript=active_session.transcript
            )
    except KeyboardInterrupt:
        if session is not None and nodes:
            files = write_artifacts(args.output_dir, nodes=nodes, transcript=session.transcript)
            print(
                json.dumps(
                    {
                        "host": args.host,
                        "interrupted": True,
                        "node_count": len(nodes),
                        "output_dir": str(args.output_dir),
                        "files": {key: str(path) for key, path in files.items()},
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 130
        raise
    print(
        json.dumps(
            {
                "host": args.host,
                "node_count": len(nodes),
                "output_dir": str(args.output_dir),
                "files": {key: str(path) for key, path in files.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
