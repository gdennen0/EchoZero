#!/usr/bin/env python3
"""ma3-terminal-timecode-deep-sample: Capture one richer native timecode subtree through ordered children.
Exists because timecode objects have indexing quirks that need a dedicated native path proof.
Connects the live MA terminal to a focused timecode artifact bundle down to CmdSubTrack and CmdEvent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_terminal_class_catalog import (
    build_property_probe_command,
    parse_property_probe_output,
)
from MA3.dev.ma3_terminal_datapool_crawl import (
    MA3TerminalSession,
    parse_dump_text,
    probe_children_order,
)


def collect_node(session: MA3TerminalSession, *, expression: str) -> dict[str, object]:
    """Collect dump, ordered children, and property metadata for one expression."""

    dump_output = session.send_command(f'Lua "{expression}:Dump()"')
    dump_fields = parse_dump_text(dump_output)
    property_output = session.send_command(build_property_probe_command(expression))
    property_fields = parse_property_probe_output(property_output)
    ordered_children = probe_children_order(session, expression=expression)
    return {
        "expression": expression,
        "name": dump_fields.name,
        "class": dump_fields.class_name,
        "object_path": dump_fields.object_path,
        "dump_properties": dump_fields.properties,
        "dump_children": dump_fields.children,
        "ordered_children": ordered_children,
        "property_count": property_fields["property_count"],
        "properties": property_fields["properties"],
        "dump_text": dump_output,
        "property_probe_output": property_output,
    }


def build_timecode_walk(
    session: MA3TerminalSession, *, timecode_expression: str
) -> list[dict[str, object]]:
    """Walk one representative timecode path down to CmdEvent using ordered children."""

    collected: list[dict[str, object]] = []
    timecode = collect_node(session, expression=timecode_expression)
    collected.append(timecode)

    track_group_expression = f"{timecode_expression}[1]"
    track_group = collect_node(session, expression=track_group_expression)
    collected.append(track_group)

    track_child = next(
        (child for child in track_group["ordered_children"] if child["class"] == "Track"),
        None,
    )
    if track_child is None:
        return collected

    track_expression = f"{track_group_expression}[{track_child['ordinal']}]"
    track = collect_node(session, expression=track_expression)
    collected.append(track)

    time_range_child = next(
        (child for child in track["ordered_children"] if child["class"] == "TimeRange"),
        None,
    )
    if time_range_child is None:
        return collected

    time_range_expression = f"{track_expression}[{time_range_child['ordinal']}]"
    time_range = collect_node(session, expression=time_range_expression)
    collected.append(time_range)

    subtrack_child = next(
        (
            child
            for child in time_range["ordered_children"]
            if child["class"] in {"CmdSubTrack", "FaderSubTrack"}
        ),
        None,
    )
    if subtrack_child is None:
        return collected

    subtrack_expression = f"{time_range_expression}[{subtrack_child['ordinal']}]"
    subtrack = collect_node(session, expression=subtrack_expression)
    collected.append(subtrack)

    event_child = next(
        (
            child
            for child in subtrack["ordered_children"]
            if child["class"] in {"CmdEvent", "FaderEvent"}
        ),
        None,
    )
    if event_child is None:
        return collected

    event_expression = f"{subtrack_expression}[{event_child['ordinal']}]"
    event = collect_node(session, expression=event_expression)
    collected.append(event)
    return collected


def write_artifacts(
    output_dir: Path,
    *,
    nodes: list[dict[str, object]],
    transcript: list[str],
) -> dict[str, Path]:
    """Write the rich native timecode sample artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    json_path = output_dir / "timecode_deep_sample.json"
    markdown_path = output_dir / "timecode_deep_sample.md"
    transcript_path = output_dir / "terminal_transcript.txt"

    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "node_count": len(nodes),
                "nodes": nodes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Timecode Deep Sample",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Node count: `{len(nodes)}`",
        "",
    ]
    for node in nodes:
        lines.append(f"## {node['class']} `{node['expression']}`")
        lines.append("")
        lines.append(f"- Name: `{node['name']}`")
        lines.append(f"- Object path: `{node['object_path']}`")
        lines.append(f"- Property count: `{node['property_count']}`")
        if node["ordered_children"]:
            child_summary = "; ".join(
                f"{child['ordinal']}: {child['class']} `{child['name']}`"
                for child in node["ordered_children"]
            )
            lines.append(f"- Ordered children: `{child_summary}`")
        else:
            lines.append("- Ordered children: `(none)`")
        key_properties = {prop["name"]: prop for prop in node["properties"]}
        for key in (
            "INDEX",
            "NO",
            "TIME",
            "ABSTIME",
            "RAWTIME",
            "START",
            "DURATION",
            "TRACK",
            "TRACKGROUP",
            "TOKEN",
            "CUEDESTINATION",
            "EXECUTECOMMAND",
        ):
            if key not in key_properties:
                continue
            prop = key_properties[key]
            lines.append(
                f"- `{key}`: type={prop['type']} read_only={prop['read_only']} enum={prop['enum_collection'] or '-'}"
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
    parser = argparse.ArgumentParser(description="Capture a richer native MA3 timecode subtree.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--timecode-expression", default="DataPool()[14][1]")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ma3-terminal-crawl" / "timecode-deep-sample",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Capture one richer native timecode sample bundle."""

    args = _build_parser().parse_args(argv)
    with MA3TerminalSession(host=args.host, timeout_seconds=args.timeout_seconds) as session:
        nodes = build_timecode_walk(session, timecode_expression=args.timecode_expression)
        files = write_artifacts(args.output_dir, nodes=nodes, transcript=session.transcript)
    print(
        json.dumps(
            {
                "host": args.host,
                "timecode_expression": args.timecode_expression,
                "node_count": len(nodes),
                "output_dir": str(args.output_dir),
                "files": {key: str(value) for key, value in files.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
