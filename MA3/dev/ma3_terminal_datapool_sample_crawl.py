#!/usr/bin/env python3
"""ma3-terminal-datapool-sample-crawl: Sample MA3 DataPool structures through the native terminal.
Exists because we want class and layer structure without brute-forcing every project-specific object.
Connects the local grandMA3 terminal app to a sample-first artifact bundle for MA object-model discovery.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_terminal_datapool_crawl import (
    MA3TerminalSession,
    CrawlNode,
    dump_expression,
    parse_dump,
)


@dataclass(frozen=True)
class SampleTarget:
    """One top-level pool and how to sample it."""

    pool_index: int
    pool_name: str
    mode: str


DEFAULT_SAMPLE_TARGETS = (
    SampleTarget(1, "Worlds", "sample_one"),
    SampleTarget(2, "Filters", "sample_one"),
    SampleTarget(3, "GeneratorTypes", "sample_one"),
    SampleTarget(4, "PresetPools", "sample_one"),
    SampleTarget(5, "Groups", "sample_one"),
    SampleTarget(6, "Sequences", "sample_one"),
    SampleTarget(7, "Plugins", "sample_one"),
    SampleTarget(8, "Macros", "sample_one"),
    SampleTarget(9, "Quickeys", "sample_one"),
    SampleTarget(10, "MAtricks", "sample_one"),
    SampleTarget(11, "Configurations", "sample_one"),
    SampleTarget(12, "Pages", "sample_one"),
    SampleTarget(13, "Layouts", "sample_one"),
    SampleTarget(14, "Timecodes", "sample_one"),
    SampleTarget(15, "Timers", "sample_one"),
)


def _sample_child_path(node: CrawlNode) -> list[int] | None:
    if not node.children:
        return None
    first_child_index = int(node.children[0]["index"])
    return node.path + [first_child_index]


def _dump_node(session: MA3TerminalSession, path: list[int]) -> CrawlNode:
    response = session.send_command(dump_expression(path))
    return parse_dump(path, response)


def _crawl_subtree(session: MA3TerminalSession, root_path: list[int]) -> list[CrawlNode]:
    queue: deque[list[int]] = deque([list(root_path)])
    seen: set[tuple[int, ...]] = set()
    nodes: list[CrawlNode] = []
    while queue:
        path = queue.popleft()
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        node = _dump_node(session, path)
        nodes.append(node)
        for child in node.children:
            queue.append(path + [int(child["index"])])
    return nodes


def run_sample_crawl(
    session: MA3TerminalSession,
    *,
    progress: bool = False,
    checkpoint_callback=None,
) -> dict[str, object]:
    """Capture root, top-level pools, and one representative subtree per pool."""

    root_node = _dump_node(session, [])
    top_level_nodes: list[CrawlNode] = []
    sampled_subtrees: dict[str, list[CrawlNode]] = {}

    if progress:
        print(
            f"ROOT {root_node.expression} class={root_node.class_name} children={len(root_node.children)}",
            flush=True,
        )

    for target in DEFAULT_SAMPLE_TARGETS:
        pool_path = [target.pool_index]
        pool_node = _dump_node(session, pool_path)
        top_level_nodes.append(pool_node)
        if progress:
            print(
                f"POOL {target.pool_name}: {pool_node.expression} class={pool_node.class_name} children={len(pool_node.children)}",
                flush=True,
            )
        sample_path = _sample_child_path(pool_node)
        if sample_path is None:
            sampled_subtrees[target.pool_name] = []
            if checkpoint_callback is not None:
                checkpoint_callback(root_node, top_level_nodes, sampled_subtrees)
            continue
        if progress:
            print(f"SAMPLE {target.pool_name}: starting {sample_path}", flush=True)
        sampled_subtrees[target.pool_name] = _crawl_subtree(session, sample_path)
        if progress:
            print(
                f"SAMPLE {target.pool_name}: done nodes={len(sampled_subtrees[target.pool_name])}",
                flush=True,
            )
        if checkpoint_callback is not None:
            checkpoint_callback(root_node, top_level_nodes, sampled_subtrees)

    return {
        "root": root_node,
        "top_level_nodes": top_level_nodes,
        "sampled_subtrees": sampled_subtrees,
    }


def _node_payload(node: CrawlNode) -> dict[str, object]:
    return {
        "path": node.path,
        "expression": node.expression,
        "name": node.name,
        "class": node.class_name,
        "object_path": node.object_path,
        "properties": node.properties,
        "children": node.children,
        "dump_text": node.dump_text,
    }


def write_artifacts(
    output_dir: Path, *, crawl: dict[str, object], transcript: list[str]
) -> dict[str, Path]:
    """Write sample crawl artifacts to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "datapool_sample_crawl.json"
    markdown_path = output_dir / "datapool_sample_crawl.md"
    transcript_path = output_dir / "terminal_transcript.txt"

    root_node: CrawlNode = crawl["root"]
    top_level_nodes: list[CrawlNode] = crawl["top_level_nodes"]
    sampled_subtrees: dict[str, list[CrawlNode]] = crawl["sampled_subtrees"]
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    payload = {
        "generated_at": generated_at,
        "strategy": "root + top-level pools + one representative subtree per pool",
        "root": _node_payload(root_node),
        "top_level_nodes": [_node_payload(node) for node in top_level_nodes],
        "sampled_subtrees": {
            key: [_node_payload(node) for node in nodes] for key, nodes in sampled_subtrees.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# MA3 DataPool Sample Crawl",
        "",
        f"- Generated at: `{generated_at}`",
        "- Strategy: root + top-level pools + one representative subtree per pool",
        "",
        f"- Root: `{root_node.expression}` [{root_node.class_name}] children={len(root_node.children)}",
        "",
        "## Top Level Pools",
        "",
    ]
    for node in top_level_nodes:
        lines.append(
            f"- `{node.expression}` [{node.class_name}] name=`{node.name}` "
            f"children={len(node.children)} properties={len(node.properties)}"
        )
    lines.extend(["", "## Sampled Subtrees", ""])
    for pool_name, nodes in sampled_subtrees.items():
        lines.append(f"### {pool_name}")
        lines.append("")
        if not nodes:
            lines.append("- `(no sample child)`")
            lines.append("")
            continue
        for node in nodes:
            indent = "  " * max(0, len(node.path) - 1)
            lines.append(
                f"{indent}- `{node.expression}` [{node.class_name}] "
                f"name=`{node.name}` children={len(node.children)} properties={len(node.properties)}"
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
        description="Run a sample-first MA3 DataPool crawl using the native terminal app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ma3-terminal-crawl" / "sample",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the native sample crawl and write artifacts."""

    args = _build_parser().parse_args(argv)
    files: dict[str, Path] = {}
    with MA3TerminalSession(host=args.host, timeout_seconds=args.timeout_seconds) as session:

        def checkpoint_callback(root_node, top_level_nodes, sampled_subtrees) -> None:
            write_artifacts(
                args.output_dir,
                crawl={
                    "root": root_node,
                    "top_level_nodes": list(top_level_nodes),
                    "sampled_subtrees": dict(sampled_subtrees),
                },
                transcript=session.transcript,
            )

        crawl = run_sample_crawl(
            session,
            progress=True,
            checkpoint_callback=checkpoint_callback,
        )
        files = write_artifacts(args.output_dir, crawl=crawl, transcript=session.transcript)
    print(
        json.dumps(
            {
                "host": args.host,
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
