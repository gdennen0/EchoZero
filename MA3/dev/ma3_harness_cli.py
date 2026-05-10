#!/usr/bin/env python3
"""Unified MA3 harness CLI for EchoZero bridge probes, browse calls, and smoke flows."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
from time import monotonic
from time import sleep
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_harness_common import build_bridge  # noqa: E402
from MA3.dev.ma3_terminal_datapool_crawl import MA3TerminalSession  # noqa: E402
from echozero.infrastructure.sync.ma3_adapter import (  # noqa: E402
    MA3EventSnapshot,
    MA3PresetSnapshot,
    MA3SequenceRangeSnapshot,
    MA3SequenceSnapshot,
    MA3TimecodeSnapshot,
    MA3TrackGroupSnapshot,
    MA3TrackSnapshot,
)
from echozero.infrastructure.sync.ma3_osc import MA3OSCBridge  # noqa: E402

DEFAULT_LIVE_PLUGIN_ROOT = Path("/Users/march/MALightingTechnology/gma3_library/datapools/plugins")

SUPPORTED_PRESET_TYPES = (
    (1, "Dimmer"),
    (2, "Position"),
    (4, "Color"),
    (5, "Beam"),
    (6, "Focus"),
    (22, "Optical"),
)
LOOK_21_MIXED_TYPE_STEPS = (
    ["1.1", "4.1"],
    ["2.1", "6.1"],
    ["5.1", "22.1"],
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified EchoZero MA3 harness toolkit.",
    )
    parser.add_argument("--ma3-host", default=None)
    parser.add_argument("--ma3-port", type=int, default=None)
    parser.add_argument("--command-path", default=None)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--settings-path", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--json", action="store_true", help="Print structured JSON output.")
    parser.add_argument(
        "--terminal-feedback",
        action="store_true",
        help="Also run a matching native MA3 terminal probe and attach its output.",
    )
    parser.add_argument(
        "--terminal-host",
        default=None,
        help="Host passed to the native MA3 terminal cmdline session. Defaults to --ma3-host.",
    )
    parser.add_argument(
        "--terminal-timeout",
        type=float,
        default=10.0,
        help="Timeout for native MA3 terminal probes when --terminal-feedback is enabled.",
    )
    parser.add_argument(
        "--transcript-out",
        type=Path,
        default=None,
        help="Optional JSON file to write the inbound bridge transcript and command summary.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ping", help="Send EZ.Ping() and wait for reply.")
    subparsers.add_parser("version", help="Send EZ.Version() and wait for plugin.version reply.")
    subparsers.add_parser("health", help="Send EZ.GetPluginHealth() and print reply.")
    health_check = subparsers.add_parser(
        "health-check",
        help="Compare live MA3 plugin health against expected local plugin markers.",
    )
    health_check.add_argument("--expected-root", type=Path, default=None)
    health_check.add_argument("--no-compare", action="store_true")
    validation_report = subparsers.add_parser(
        "validation-report",
        help="Run the canonical MA3 hardware-validation bundle and write evidence artifacts.",
    )
    validation_report.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ma3-harness" / "latest",
        help="Directory where summary.json, summary.md, and transcript.json will be written.",
    )
    validation_report.add_argument("--expected-root", type=Path, default=None)
    validation_report.add_argument("--no-compare", action="store_true")
    validation_report.add_argument(
        "--receive-duration-seconds",
        type=float,
        default=0.0,
        help="Optional receive-capture window to include in the validation bundle.",
    )
    validation_report.add_argument(
        "--receive-trigger-command",
        default="",
        help="Optional MA3 Lua command to trigger during the receive-capture window.",
    )
    receive_capture = subparsers.add_parser(
        "receive-capture",
        help="Listen for inbound MA3 -> EchoZero traffic and write a capture transcript.",
    )
    receive_capture.add_argument(
        "--duration-seconds",
        type=float,
        default=3.0,
        help="How long to keep the MA3 listener open after target configuration.",
    )
    receive_capture.add_argument(
        "--ping-first",
        action="store_true",
        help="Send EZ.Ping() before listening so the command/reply plane is verified first.",
    )
    receive_capture.add_argument(
        "--trigger-command",
        default="",
        help="Optional MA3 Lua command to send after the capture window opens.",
    )
    stream = subparsers.add_parser(
        "stream",
        help="Print inbound MA3 -> EchoZero messages to stdout as NDJSON.",
    )
    stream.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Optional bounded stream window. Uses an open-ended stream when omitted or <= 0.",
    )
    stream.add_argument(
        "--ping-first",
        action="store_true",
        help="Send EZ.Ping() before streaming so the command/reply plane is verified first.",
    )
    stream.add_argument(
        "--trigger-command",
        default="",
        help="Optional MA3 Lua command to send once the stream listener is ready.",
    )
    subparsers.add_parser("reload", help="Send RP over the configured /cmd path.")
    subparsers.add_parser("timecodes", help="List MA3 timecodes.")

    create_timecode = subparsers.add_parser(
        "create-timecode",
        help="Create one MA3 timecode with an optional preferred name.",
    )
    create_timecode.add_argument("--name", default=None)

    track_groups = subparsers.add_parser(
        "track-groups", help="List track groups for one timecode."
    )
    track_groups.add_argument("--timecode-no", type=int, required=True)

    create_track_group = subparsers.add_parser(
        "create-track-group",
        help="Create one MA3 track group inside one timecode.",
    )
    create_track_group.add_argument("--timecode-no", type=int, required=True)
    create_track_group.add_argument("--name", default=None)

    tracks = subparsers.add_parser(
        "tracks", help="List tracks for one timecode or one track group."
    )
    tracks.add_argument("--timecode-no", type=int, default=None)
    tracks.add_argument("--track-group-no", type=int, default=None)

    create_track = subparsers.add_parser(
        "create-track",
        help="Create one MA3 track inside one timecode and track group.",
    )
    create_track.add_argument("--timecode-no", type=int, required=True)
    create_track.add_argument("--track-group-no", type=int, required=True)
    create_track.add_argument("--name", default=None)

    events = subparsers.add_parser("events", help="List events for one MA3 track coord.")
    events.add_argument("--track-coord", required=True)

    create_static_preset = subparsers.add_parser(
        "create-static-preset",
        help="Create one deterministic static preset from explicit selection and programmer commands.",
    )
    create_static_preset.add_argument("--preset-type", type=int, required=True)
    create_static_preset.add_argument("--preset-no", type=int, required=True)
    create_static_preset.add_argument("--store-mode", required=True)
    create_static_preset.add_argument("--name", required=True)
    create_static_preset.add_argument("--selection-command", required=True)
    create_static_preset.add_argument("--value-command", required=True)

    create_phaser_preset = subparsers.add_parser(
        "create-phaser-preset",
        help="Create one deterministic phaser preset from explicit step preset references.",
    )
    create_phaser_preset.add_argument("--preset-type", type=int, required=True)
    create_phaser_preset.add_argument("--preset-no", type=int, required=True)
    create_phaser_preset.add_argument("--store-mode", required=True)
    create_phaser_preset.add_argument("--name", required=True)
    create_phaser_preset.add_argument("--selection-command", required=True)
    create_phaser_preset.add_argument(
        "--step",
        action="append",
        dest="steps",
        required=True,
        help="One phaser step. Use '+' inside a step to combine multiple preset refs.",
    )
    create_phaser_preset.add_argument("--speed-bpm", type=float, default=None)

    create_phaser_fixture_set = subparsers.add_parser(
        "create-phaser-fixture-set",
        help=(
            "Create one phaser for each supported preset type plus one mixed-type phaser in pool 21."
        ),
    )
    create_phaser_fixture_set.add_argument("--selection-command", required=True)
    create_phaser_fixture_set.add_argument("--speed-bpm", type=float, default=120.0)
    create_phaser_fixture_set.add_argument("--look-21-speed-bpm", type=float, default=96.0)

    create_recipe_preset = subparsers.add_parser(
        "create-recipe-preset",
        help="Create one deterministic recipe preset from one explicit source preset ref.",
    )
    create_recipe_preset.add_argument("--preset-type", type=int, required=True)
    create_recipe_preset.add_argument("--preset-no", type=int, required=True)
    create_recipe_preset.add_argument("--store-mode", required=True)
    create_recipe_preset.add_argument("--name", required=True)
    create_recipe_preset.add_argument("--selection-command", required=True)
    create_recipe_preset.add_argument("--source-preset-ref", required=True)
    create_recipe_preset.add_argument("--selection-mode", default="Strict")

    edit_static_preset = subparsers.add_parser(
        "edit-static-preset",
        help="Deterministically replace one static preset with explicit authoring inputs.",
    )
    edit_static_preset.add_argument("--preset-type", type=int, required=True)
    edit_static_preset.add_argument("--preset-no", type=int, required=True)
    edit_static_preset.add_argument("--store-mode", required=True)
    edit_static_preset.add_argument("--name", required=True)
    edit_static_preset.add_argument("--selection-command", required=True)
    edit_static_preset.add_argument("--value-command", required=True)

    edit_phaser_preset = subparsers.add_parser(
        "edit-phaser-preset",
        help="Deterministically replace one phaser preset from explicit step preset refs.",
    )
    edit_phaser_preset.add_argument("--preset-type", type=int, required=True)
    edit_phaser_preset.add_argument("--preset-no", type=int, required=True)
    edit_phaser_preset.add_argument("--store-mode", required=True)
    edit_phaser_preset.add_argument("--name", required=True)
    edit_phaser_preset.add_argument("--selection-command", required=True)
    edit_phaser_preset.add_argument(
        "--step",
        action="append",
        dest="steps",
        required=True,
        help="One phaser step. Use '+' inside a step to combine multiple preset refs.",
    )
    edit_phaser_preset.add_argument("--speed-bpm", type=float, default=None)

    edit_recipe_preset = subparsers.add_parser(
        "edit-recipe-preset",
        help="Deterministically replace one recipe preset from one explicit source preset ref.",
    )
    edit_recipe_preset.add_argument("--preset-type", type=int, required=True)
    edit_recipe_preset.add_argument("--preset-no", type=int, required=True)
    edit_recipe_preset.add_argument("--store-mode", required=True)
    edit_recipe_preset.add_argument("--name", required=True)
    edit_recipe_preset.add_argument("--selection-command", required=True)
    edit_recipe_preset.add_argument("--source-preset-ref", required=True)
    edit_recipe_preset.add_argument("--selection-mode", default="Strict")

    sequences = subparsers.add_parser("sequences", help="List MA3 sequences.")
    sequences.add_argument("--start-no", type=int, default=None)
    sequences.add_argument("--end-no", type=int, default=None)

    create_sequence_next_available = subparsers.add_parser(
        "create-sequence-next-available",
        help="Create one MA3 sequence at the next available sequence number.",
    )
    create_sequence_next_available.add_argument("--name", default=None)

    create_sequence_in_current_song_range = subparsers.add_parser(
        "create-sequence-in-current-song-range",
        help="Create one MA3 sequence inside the current-song range.",
    )
    create_sequence_in_current_song_range.add_argument("--name", default=None)

    sequence_cues = subparsers.add_parser("sequence-cues", help="List cue rows for one sequence.")
    sequence_cues.add_argument("--sequence-no", type=int, required=True)

    datapool_children = subparsers.add_parser(
        "datapool-children",
        help="List generic DataPool children for one path (or root when omitted).",
    )
    datapool_children.add_argument("--path", default=None)

    datapool_object = subparsers.add_parser(
        "datapool-object",
        help="Describe one generic DataPool object path (or root when omitted).",
    )
    datapool_object.add_argument("--path", default=None)

    datapool_report = subparsers.add_parser(
        "datapool-report",
        help="Render one DataPool object and its child objects as plain text.",
    )
    datapool_report_group = datapool_report.add_mutually_exclusive_group(required=True)
    datapool_report_group.add_argument("--path", default=None)
    datapool_report_group.add_argument(
        "--preset-ref",
        default=None,
        help="Convenience preset ref such as 21.221 or 'Preset 21.221'.",
    )
    datapool_report.add_argument(
        "--depth",
        type=int,
        default=1,
        help="How many child levels to recurse when building the report.",
    )

    list_presets = subparsers.add_parser(
        "list-presets",
        help="List presets in one MA3 preset pool through the explicit OSC preset API.",
    )
    list_presets.add_argument("--preset-type", type=int, required=True)

    describe_preset = subparsers.add_parser(
        "describe-preset",
        help="Describe one preset and its child recipe/phaser lines through the explicit OSC preset API.",
    )
    describe_preset_group = describe_preset.add_mutually_exclusive_group(required=True)
    describe_preset_group.add_argument("--preset-ref", default=None)
    describe_preset_group.add_argument("--path", default=None, help=argparse.SUPPRESS)

    preview_replace_preset_when_group = subparsers.add_parser(
        "preview-replace-preset-when-group",
        help="Preview recipe-line preset replacements filtered by group and sequence through OSC.",
    )
    preview_replace_preset_when_group.add_argument("--preset-type", type=int, required=True)
    preview_replace_preset_when_group.add_argument("--source-preset-ref", required=True)
    preview_replace_preset_when_group.add_argument("--dest-preset-ref", required=True)
    preview_replace_preset_when_group.add_argument("--group-filter", required=True)
    preview_replace_preset_when_group.add_argument("--sequence-numbers", required=True)

    replace_preset_when_group = subparsers.add_parser(
        "replace-preset-when-group",
        help="Apply recipe-line preset replacements filtered by group and sequence through OSC.",
    )
    replace_preset_when_group.add_argument("--preset-type", type=int, required=True)
    replace_preset_when_group.add_argument("--source-preset-ref", required=True)
    replace_preset_when_group.add_argument("--dest-preset-ref", required=True)
    replace_preset_when_group.add_argument("--group-filter", required=True)
    replace_preset_when_group.add_argument("--sequence-numbers", required=True)

    analyze_cue_recipe_state = subparsers.add_parser(
        "analyze-cue-recipe-state",
        help="Analyze effective recipe contributors for one cue through OSC.",
    )
    analyze_cue_recipe_state.add_argument("--sequence-no", type=int, required=True)
    analyze_cue_recipe_state.add_argument("--cue-no", required=True)

    preview_recipe_cue_only = subparsers.add_parser(
        "preview-recipe-cue-only",
        help="Preview cue-only restore lines when copying one cue's recipe lines into another cue.",
    )
    preview_recipe_cue_only.add_argument("--sequence-no", type=int, required=True)
    preview_recipe_cue_only.add_argument("--source-cue-no", required=True)
    preview_recipe_cue_only.add_argument("--target-cue-no", required=True)

    apply_recipe_cue_only = subparsers.add_parser(
        "apply-recipe-cue-only",
        help="Apply cue-only recipe copy semantics by restoring affected state in the following cue.",
    )
    apply_recipe_cue_only.add_argument("--sequence-no", type=int, required=True)
    apply_recipe_cue_only.add_argument("--source-cue-no", required=True)
    apply_recipe_cue_only.add_argument("--target-cue-no", required=True)

    preview_copy_cue_with_status = subparsers.add_parser(
        "preview-copy-cue-with-status",
        help="Preview effective recipe contributors that a status copy would bring forward.",
    )
    preview_copy_cue_with_status.add_argument("--sequence-no", type=int, required=True)
    preview_copy_cue_with_status.add_argument("--source-cue-no", required=True)
    preview_copy_cue_with_status.add_argument("--dest-cue-no", required=True)

    copy_cue_with_status = subparsers.add_parser(
        "copy-cue-with-status",
        help="Copy the effective current recipe contributor set for one cue into another cue.",
    )
    copy_cue_with_status.add_argument("--sequence-no", type=int, required=True)
    copy_cue_with_status.add_argument("--source-cue-no", required=True)
    copy_cue_with_status.add_argument("--dest-cue-no", required=True)

    subparsers.add_parser("current-song-range", help="Resolve current-song sequence range.")
    subparsers.add_parser("smoke", help="Run the canonical MA3 harness smoke flow.")
    return parser


def _bridge_from_args(args: argparse.Namespace) -> tuple[MA3OSCBridge, dict[str, Any]]:
    bridge, target = build_bridge(
        ma3_host=args.ma3_host,
        ma3_port=args.ma3_port,
        command_path=args.command_path,
        settings_path=args.settings_path,
        listen_host=str(args.listen_host or "0.0.0.0"),
        listen_port=int(args.listen_port),
        timeout=float(args.timeout),
    )
    return bridge, target


def _timecode_payload(snapshot: MA3TimecodeSnapshot) -> dict[str, object]:
    return {"number": snapshot.number, "name": snapshot.name}


def _track_group_payload(snapshot: MA3TrackGroupSnapshot) -> dict[str, object]:
    return {
        "number": snapshot.number,
        "name": snapshot.name,
        "track_count": snapshot.track_count,
    }


def _track_payload(snapshot: MA3TrackSnapshot) -> dict[str, object]:
    return {
        "coord": snapshot.coord,
        "name": snapshot.name,
        "number": snapshot.number,
        "timecode_name": snapshot.timecode_name,
        "note": snapshot.note,
        "event_count": snapshot.event_count,
        "sequence_no": snapshot.sequence_no,
    }


def _event_payload(snapshot: MA3EventSnapshot) -> dict[str, object]:
    return {
        "event_id": snapshot.event_id,
        "label": snapshot.label,
        "start": snapshot.start,
        "end": snapshot.end,
        "cmd": snapshot.cmd,
        "cue_number": snapshot.cue_number,
        "cue_ref": snapshot.cue_ref,
        "color": snapshot.color,
        "notes": snapshot.notes,
        "payload_ref": snapshot.payload_ref,
    }


def _sequence_payload(snapshot: MA3SequenceSnapshot) -> dict[str, object]:
    return {
        "number": snapshot.number,
        "name": snapshot.name,
        "cue_count": snapshot.cue_count,
    }


def _sequence_range_payload(snapshot: MA3SequenceRangeSnapshot | None) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "song_label": snapshot.song_label,
        "start": snapshot.start,
        "end": snapshot.end,
    }


def _preset_payload(snapshot: MA3PresetSnapshot) -> dict[str, object]:
    return {
        "preset_type": snapshot.preset_type,
        "number": snapshot.number,
        "name": snapshot.name,
        "store_mode": snapshot.store_mode,
        "kind": snapshot.kind,
        "step_count": snapshot.step_count,
    }


def _message_payload(message: Any) -> dict[str, object]:
    return {
        "key": message.key,
        "message_type": message.message_type,
        "change": message.change,
        "timestamp": message.timestamp,
        "fields": dict(message.fields),
        "raw_payload": message.raw_payload,
    }


def _resolve_preset_ref_path(raw_preset_ref: str) -> str:
    normalized = str(raw_preset_ref or "").strip()
    if normalized.lower().startswith("preset "):
        normalized = normalized[7:].strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", normalized)
    if match is None:
        raise ValueError("preset-ref must look like 21.221")
    preset_type_no, preset_no = match.groups()
    return f"PresetPools/{int(preset_type_no)}/{int(preset_no)}"


def _parse_preset_ref(raw_preset_ref: str) -> tuple[int, int]:
    normalized = str(raw_preset_ref or "").strip()
    if normalized.lower().startswith("preset "):
        normalized = normalized[7:].strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", normalized)
    if match is None:
        raise ValueError("preset-ref must look like 21.221")
    return int(match.group(1)), int(match.group(2))


def _fetch_datapool_tree(
    bridge: MA3OSCBridge,
    *,
    path: str,
    depth: int,
) -> dict[str, object]:
    node = dict(bridge.describe_datapool_object(path=path))
    if not node:
        raise RuntimeError(f"MA3 DataPool object not found for {path}")
    if depth <= 0:
        node["children"] = []
        return node
    children = bridge.list_datapool_objects(path=path)
    node["children"] = [
        _fetch_datapool_tree(
            bridge,
            path=str(child.get("path") or ""),
            depth=depth - 1,
        )
        for child in children
        if str(child.get("path") or "").strip()
    ]
    return node


def _format_property_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _render_datapool_tree_text(node: dict[str, object], *, indent: int = 0) -> str:
    prefix = "  " * indent
    lines = [
        f"{prefix}path: {node.get('path') or ''}",
        f"{prefix}class: {node.get('class') or ''}",
        f"{prefix}name: {node.get('name') or ''}",
    ]
    if node.get("no") is not None:
        lines.append(f"{prefix}no: {node.get('no')}")
    if node.get("address"):
        lines.append(f"{prefix}address: {node.get('address')}")
    if node.get("child_count") is not None:
        lines.append(f"{prefix}child_count: {node.get('child_count')}")
    property_items = node.get("property_items")
    if isinstance(property_items, list) and property_items:
        lines.append(f"{prefix}properties:")
        for item in property_items:
            if not isinstance(item, dict):
                continue
            property_name = str(item.get("name") or "").strip()
            if not property_name:
                continue
            property_value = _format_property_value(item.get("value"))
            property_type = str(item.get("property_type") or "").strip()
            rendered = f"{prefix}  {property_name} = {property_value}"
            if property_type:
                rendered += f" [{property_type}]"
            lines.append(rendered)
    children = node.get("children")
    if isinstance(children, list) and children:
        lines.append(f"{prefix}children:")
        for child in children:
            if not isinstance(child, dict):
                continue
            lines.append(_render_datapool_tree_text(child, indent=indent + 1))
    return "\n".join(lines)


def _run_datapool_report(
    bridge: MA3OSCBridge,
    *,
    path: str | None,
    preset_ref: str | None,
    depth: int,
) -> dict[str, object]:
    resolved_path = _resolve_preset_ref_path(preset_ref) if preset_ref else str(path or "").strip()
    tree = _fetch_datapool_tree(bridge, path=resolved_path, depth=max(0, int(depth)))
    return {
        "path": resolved_path,
        "depth": max(0, int(depth)),
        "text": _render_datapool_tree_text(tree),
        "tree": tree,
    }


def _run_list_presets(
    bridge: MA3OSCBridge,
    *,
    preset_type_no: int,
) -> dict[str, object]:
    presets = bridge.list_presets(preset_type_no=int(preset_type_no))
    return {
        "preset_type": int(preset_type_no),
        "count": len(presets),
        "presets": presets,
    }


def _run_describe_preset(
    bridge: MA3OSCBridge,
    *,
    preset_ref: str | None,
) -> dict[str, object]:
    if not preset_ref:
        raise ValueError("preset-ref is required")
    preset_type_no, preset_no = _parse_preset_ref(preset_ref)
    description = bridge.describe_preset(
        preset_type_no=preset_type_no,
        preset_no=preset_no,
    )
    return {
        "preset_type": preset_type_no,
        "preset_no": preset_no,
        "text": _render_datapool_tree_text(description),
        "object": description,
    }


def _run_preview_replace_preset_when_group(
    bridge: MA3OSCBridge,
    *,
    preset_type_no: int,
    source_preset_ref: str,
    dest_preset_ref: str,
    group_filter: str,
    sequence_numbers: str,
) -> dict[str, object]:
    return bridge.preview_replace_preset_when_group(
        preset_type_no=int(preset_type_no),
        source_preset_ref=str(source_preset_ref),
        dest_preset_ref=str(dest_preset_ref),
        group_filter_csv=str(group_filter),
        sequence_numbers_csv=str(sequence_numbers),
    )


def _run_replace_preset_when_group(
    bridge: MA3OSCBridge,
    *,
    preset_type_no: int,
    source_preset_ref: str,
    dest_preset_ref: str,
    group_filter: str,
    sequence_numbers: str,
) -> dict[str, object]:
    return bridge.replace_preset_when_group(
        preset_type_no=int(preset_type_no),
        source_preset_ref=str(source_preset_ref),
        dest_preset_ref=str(dest_preset_ref),
        group_filter_csv=str(group_filter),
        sequence_numbers_csv=str(sequence_numbers),
    )


def _run_analyze_cue_recipe_state(
    bridge: MA3OSCBridge,
    *,
    sequence_no: int,
    cue_no: str,
) -> dict[str, object]:
    return bridge.analyze_cue_recipe_state(
        sequence_no=int(sequence_no),
        cue_no=str(cue_no),
    )


def _run_preview_recipe_cue_only(
    bridge: MA3OSCBridge,
    *,
    sequence_no: int,
    source_cue_no: str,
    target_cue_no: str,
) -> dict[str, object]:
    return bridge.preview_recipe_cue_only(
        sequence_no=int(sequence_no),
        source_cue_no=str(source_cue_no),
        target_cue_no=str(target_cue_no),
    )


def _run_apply_recipe_cue_only(
    bridge: MA3OSCBridge,
    *,
    sequence_no: int,
    source_cue_no: str,
    target_cue_no: str,
) -> dict[str, object]:
    return bridge.apply_recipe_cue_only(
        sequence_no=int(sequence_no),
        source_cue_no=str(source_cue_no),
        target_cue_no=str(target_cue_no),
    )


def _run_copy_cue_with_status(
    bridge: MA3OSCBridge,
    *,
    sequence_no: int,
    source_cue_no: str,
    dest_cue_no: str,
) -> dict[str, object]:
    return bridge.copy_cue_with_status(
        sequence_no=int(sequence_no),
        source_cue_no=str(source_cue_no),
        dest_cue_no=str(dest_cue_no),
    )


def _run_create_phaser_fixture_set(
    bridge: MA3OSCBridge,
    *,
    selection_command: str,
    speed_bpm: float,
    look_21_speed_bpm: float,
) -> dict[str, object]:
    per_type_phasers = []
    for preset_type_no, preset_label in SUPPORTED_PRESET_TYPES:
        snapshot = bridge.create_phaser_preset(
            preset_type_no=int(preset_type_no),
            preset_no=200 + int(preset_type_no),
            store_mode="Global",
            preset_name=f"{preset_label} Chase",
            selection_command=selection_command,
            step_preset_refs=[
                f"{preset_type_no}.1",
                f"{preset_type_no}.2",
                f"{preset_type_no}.3",
            ],
            speed_bpm=float(speed_bpm),
        )
        per_type_phasers.append(_preset_payload(snapshot))

    look_21_snapshot = bridge.create_phaser_preset(
        preset_type_no=21,
        preset_no=221,
        store_mode="Selective",
        preset_name="Mixed Type Phaser",
        selection_command=selection_command,
        step_preset_refs=[list(step_refs) for step_refs in LOOK_21_MIXED_TYPE_STEPS],
        speed_bpm=float(look_21_speed_bpm),
    )
    return {
        "per_type_phasers": per_type_phasers,
        "look_21_phaser": _preset_payload(look_21_snapshot),
    }


def _write_transcript(
    path: Path, *, target: dict[str, Any], commands: list[str], bridge: MA3OSCBridge
) -> None:
    _write_transcript_with_terminal_feedback(
        path,
        target=target,
        commands=commands,
        bridge=bridge,
        terminal_feedback=None,
    )


def _write_transcript_with_terminal_feedback(
    path: Path,
    *,
    target: dict[str, Any],
    commands: list[str],
    bridge: MA3OSCBridge,
    terminal_feedback: dict[str, object] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": dict(target),
        "commands": list(commands),
        "messages": [_message_payload(message) for message in bridge.messages],
    }
    if terminal_feedback is not None:
        payload["terminal_feedback"] = terminal_feedback
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown_report(path: Path, *, report: dict[str, Any], transcript_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = report["target"]
    health_check = report["health_check"]
    health = health_check["health"]
    summary = report["summary"]
    lines = [
        "# MA3 Hardware Validation Report",
        "",
        f"- Generated: {report['generated_at_utc']}",
        f"- Target: {target['ma3_host']}:{target['ma3_port']} {target['command_path']}",
        f"- Settings Path: {target['settings_path']}",
        f"- Status: {report['status']}",
        "",
        "## Identity",
        "",
        f"- EZ version: {health.get('ez_version')}",
        f"- EZ build: {health.get('ez_build')}",
        f"- HitMaker version: {health.get('hitmaker_version')}",
        f"- HitMaker build: {health.get('hitmaker_build')}",
        f"- Health compare enabled: {health_check['compare_enabled']}",
        f"- Expected root: {health_check['expected_root']}",
        "",
        "## Browse Summary",
        "",
        f"- Timecodes: {summary['timecode_count']}",
        f"- Sequences: {summary['sequence_count']}",
        f"- Current-song range present: {summary['has_current_song_range']}",
        f"- First timecode number: {summary['first_timecode_no']}",
        f"- First track-group count: {summary['first_timecode_track_group_count']}",
        f"- First track count: {summary['first_track_group_track_count']}",
        "",
    ]
    receive_capture = report.get("receive_capture")
    if isinstance(receive_capture, dict):
        lines.extend(
            [
                "## Receive Capture",
                "",
                f"- Duration seconds: {receive_capture.get('duration_seconds')}",
                f"- Trigger command: {receive_capture.get('trigger_command')}",
                f"- Message count: {receive_capture.get('message_count')}",
                f"- Transport update count: {receive_capture.get('transport_update_count')}",
                f"- Latest transport update: {json.dumps(receive_capture.get('latest_transport_update'), sort_keys=True)}",
                "",
            ]
        )
    terminal_feedback = report.get("terminal_feedback")
    if isinstance(terminal_feedback, dict):
        lines.extend(
            [
                "## Terminal Feedback",
                "",
                f"- Status: {terminal_feedback.get('status')}",
                f"- Host: {terminal_feedback.get('host')}",
                f"- Command: {terminal_feedback.get('command')}",
                f"- Probe: {terminal_feedback.get('probe')}",
                "",
            ]
        )
        output_text = str(terminal_feedback.get("output") or "").strip()
        if output_text:
            lines.extend(
                [
                    "```text",
                    output_text,
                    "```",
                    "",
                ]
            )
    failures = list(health_check.get("failures") or [])
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Transcript: {transcript_path}",
            f"- Summary JSON: {report['summary_json_path']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_marker(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    if match is None:
        return None
    return str(match.group(1)).strip() or None


def _resolve_expected_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    if DEFAULT_LIVE_PLUGIN_ROOT.exists():
        return DEFAULT_LIVE_PLUGIN_ROOT
    return REPO_ROOT / "MA3/plugins"


def _expected_local_markers(expected_root: Path) -> dict[str, str | None]:
    expected_root = expected_root.resolve()
    ez_candidates = [
        expected_root / "EZ/ez_core.lua",
        expected_root / "echozero.lua",
    ]
    hitmaker_candidates = [
        expected_root / "HitMaker/main.lua",
    ]

    def first_match(paths: list[Path], pattern: str) -> str | None:
        for path in paths:
            value = _extract_marker(path, pattern)
            if value is not None:
                return value
        return None

    return {
        "ez_version": first_match(ez_candidates, r'EZ\._version\s*=\s*"([^"]+)"'),
        "ez_build": (
            first_match(ez_candidates, r'EZ\._build\s*=\s*EZ\._build\s*or\s*"([^"]+)"')
            or first_match(ez_candidates, r'EZ\._build\s*=\s*"([^"]+)"')
        ),
        "hitmaker_version": first_match(
            hitmaker_candidates,
            r'HitMaker\._version\s*=\s*HitMaker\._version\s*or\s*"([^"]+)"',
        ),
        "hitmaker_build": (
            first_match(
                hitmaker_candidates,
                r'HitMaker\._build\s*=\s*HitMaker\._build\s*or\s*"([^"]+)"',
            )
            or first_match(
                hitmaker_candidates,
                r'HitMaker\._build\s*=\s*"([^"]+)"',
            )
        ),
    }


def _run_health_check(
    bridge: MA3OSCBridge, *, expected_root: Path | None, no_compare: bool
) -> dict[str, object]:
    health = bridge.get_plugin_health()
    resolved_expected_root = _resolve_expected_root(expected_root)
    expected = _expected_local_markers(resolved_expected_root)
    failures: list[str] = []

    if not no_compare:
        for key in ("ez_version", "ez_build", "hitmaker_version", "hitmaker_build"):
            expected_value = expected.get(key)
            if expected_value is None:
                continue
            actual_value = str(health.get(key) or "")
            if actual_value != expected_value:
                failures.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")

        if not bool(health.get("hitmaker_loaded", False)):
            failures.append("hitmaker_loaded: expected True, got False")
        if not bool(health.get("hitmaker_supports_event_type_create", False)):
            failures.append("hitmaker_supports_event_type_create: expected True, got False")

    return {
        "expected_root": str(resolved_expected_root),
        "health": health,
        "compare_enabled": not no_compare,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def _lua_quote(text: str) -> str:
    return '"' + str(text or "").replace("\\", "\\\\").replace('"', '\\"') + '"'


def _terminal_feedback_probe(args: argparse.Namespace) -> tuple[str, str] | None:
    if args.command == "ping":
        return ("ping", 'Lua "EZ.Ping()"')
    if args.command == "version":
        return ("version", 'Lua "EZ.Version()"')
    if args.command == "health":
        return (
            "health",
            (
                'Lua "local h = EZ.GetPluginHealth(); '
                "if not h then Printf('[EZ HARNESS] health nil'); return end; "
                "Printf('[EZ HARNESS] health ez=%s build=%s hitmaker=%s loaded=%s', "
                "tostring(h.ez_version or ''), tostring(h.ez_build or ''), "
                "tostring(h.hitmaker_version or ''), tostring(h.hitmaker_loaded))\""
            ),
        )
    if args.command == "analyze-cue-recipe-state":
        return (
            "analyze-cue-recipe-state",
            (
                'Lua "local p = EZ.AnalyzeCueRecipeState('
                f"{int(args.sequence_no)}, {_lua_quote(str(args.cue_no))}"
                "); "
                "if not p then Printf('[EZ HARNESS] analyze nil'); return end; "
                "Printf('[EZ HARNESS] analyze seq=%s cue=%s status=%s supported=%s local=%s contributors=%s', "
                "tostring(p.sequence_no or ''), tostring(p.cue_no or ''), tostring(p.status or ''), "
                "tostring(p.supported), tostring(p.local_line_count or 0), tostring(p.contributor_count or 0)); "
                "for _, warning in ipairs(p.warnings or {}) do Printf('[EZ HARNESS] warning: %s', tostring(warning)) end; "
                "for _, reason in ipairs(p.unsupported_reasons or {}) do Printf('[EZ HARNESS] unsupported: %s', tostring(reason)) end\""
            ),
        )
    if args.command == "preview-recipe-cue-only":
        return (
            "preview-recipe-cue-only",
            (
                'Lua "local p = EZ.PreviewRecipeCueOnly('
                f"{int(args.sequence_no)}, {_lua_quote(str(args.source_cue_no))}, {_lua_quote(str(args.target_cue_no))}"
                "); "
                "if not p then Printf('[EZ HARNESS] cue-only preview nil'); return end; "
                "Printf('[EZ HARNESS] cue-only preview seq=%s source=%s target=%s status=%s supported=%s stored=%s restore=%s changed=%s', "
                "tostring(p.sequence_no or ''), tostring(p.source_cue_no or ''), tostring(p.target_cue_no or ''), "
                "tostring(p.status or ''), tostring(p.supported), tostring(#(p.stored_lines or {})), "
                "tostring(#(p.restore_lines or {})), tostring(#(p.changed_keys or {}))); "
                "for _, reason in ipairs(p.unsupported_reasons or {}) do Printf('[EZ HARNESS] unsupported: %s', tostring(reason)) end\""
            ),
        )
    if args.command == "preview-copy-cue-with-status":
        return (
            "preview-copy-cue-with-status",
            (
                'Lua "local p = EZ.PreviewCopyCueWithStatus('
                f"{int(args.sequence_no)}, {_lua_quote(str(args.source_cue_no))}, {_lua_quote(str(args.dest_cue_no))}"
                "); "
                "if not p then Printf('[EZ HARNESS] copy-with-status preview nil'); return end; "
                "Printf('[EZ HARNESS] copy-with-status preview seq=%s source=%s dest=%s status=%s supported=%s copied=%s local=%s contributors=%s', "
                "tostring(p.sequence_no or ''), tostring(p.source_cue_no or ''), tostring(p.dest_cue_no or ''), "
                "tostring(p.status or ''), tostring(p.supported), tostring(p.copied_line_count or 0), "
                "tostring(p.local_line_count or 0), tostring(p.contributor_count or 0)); "
                "for _, reason in ipairs(p.unsupported_reasons or {}) do Printf('[EZ HARNESS] unsupported: %s', tostring(reason)) end\""
            ),
        )
    if args.command == "validation-report":
        return (
            "validation-report",
            (
                'Lua "EZ.Version(); '
                "local h = EZ.GetPluginHealth(); "
                "if h then "
                "Printf('[EZ HARNESS] validation ez=%s build=%s hitmaker=%s loaded=%s', "
                "tostring(h.ez_version or ''), tostring(h.ez_build or ''), "
                "tostring(h.hitmaker_version or ''), tostring(h.hitmaker_loaded)); "
                'end"'
            ),
        )
    return None


def _run_terminal_feedback(args: argparse.Namespace) -> dict[str, object]:
    probe = _terminal_feedback_probe(args)
    host = str(args.terminal_host or args.ma3_host or "127.0.0.1").strip() or "127.0.0.1"
    if probe is None:
        return {
            "status": "unsupported",
            "host": host,
            "probe": None,
            "command": None,
            "output": "",
            "reason": f"No native terminal feedback probe is defined for {args.command}.",
        }
    probe_name, command_text = probe
    with MA3TerminalSession(
        host=host,
        timeout_seconds=float(args.terminal_timeout),
    ) as session:
        output = session.send_command(command_text)
    return {
        "status": "ok",
        "host": host,
        "probe": probe_name,
        "command": command_text,
        "output": output,
    }


def _print_output(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return
    print(str(payload), flush=True)


def _run_smoke(bridge: MA3OSCBridge) -> dict[str, Any]:
    ping = bridge.ping()
    version = bridge.get_version_info()
    health = bridge.get_plugin_health()
    timecodes = [_timecode_payload(item) for item in bridge.list_timecodes()]
    current_song_range = _sequence_range_payload(bridge.get_current_song_sequence_range())
    sequences = [_sequence_payload(item) for item in bridge.list_sequences()]

    browse: dict[str, Any] = {
        "timecodes": timecodes,
        "current_song_range": current_song_range,
        "sequences": sequences,
    }
    if timecodes:
        first_timecode_no = int(timecodes[0]["number"])
        track_groups = [
            _track_group_payload(item)
            for item in bridge.list_track_groups(timecode_no=first_timecode_no)
        ]
        browse["track_groups"] = track_groups
        if track_groups:
            first_group_no = int(track_groups[0]["number"])
            browse["tracks"] = [
                _track_payload(item)
                for item in bridge.list_tracks(
                    timecode_no=first_timecode_no,
                    track_group_no=first_group_no,
                )
            ]
    return {
        "ping": ping,
        "version": version,
        "health": health,
        "browse": browse,
    }


def _run_validation_report(
    bridge: MA3OSCBridge,
    *,
    args: argparse.Namespace,
    target: dict[str, Any],
    output_dir: Path,
    expected_root: Path | None,
    no_compare: bool,
    receive_duration_seconds: float,
    receive_trigger_command: str,
    command_transport: Any,
) -> dict[str, Any]:
    smoke = _run_smoke(bridge)
    health_check = _run_health_check(
        bridge,
        expected_root=expected_root,
        no_compare=no_compare,
    )
    receive_capture = None
    if float(receive_duration_seconds) > 0:
        receive_capture = _run_receive_capture(
            bridge,
            duration_seconds=float(receive_duration_seconds),
            ping_first=False,
            trigger_command=str(receive_trigger_command or ""),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.json"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    commands = (
        [] if command_transport is None else list(getattr(command_transport, "commands", []))
    )
    terminal_feedback = _run_terminal_feedback(args) if bool(args.terminal_feedback) else None
    _write_transcript_with_terminal_feedback(
        transcript_path,
        target=target,
        commands=commands,
        bridge=bridge,
        terminal_feedback=terminal_feedback,
    )
    browse = smoke["browse"]
    track_groups = browse.get("track_groups") or []
    tracks = browse.get("tracks") or []
    timecodes = browse.get("timecodes") or []
    sequences = browse.get("sequences") or []
    report = {
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "target": dict(target),
        "status": "pass" if health_check["status"] == "pass" else "fail",
        "smoke": smoke,
        "health_check": health_check,
        "receive_capture": receive_capture,
        "summary": {
            "timecode_count": len(timecodes),
            "sequence_count": len(sequences),
            "has_current_song_range": browse.get("current_song_range") is not None,
            "first_timecode_no": timecodes[0]["number"] if timecodes else None,
            "first_timecode_track_group_count": len(track_groups),
            "first_track_group_track_count": len(tracks),
        },
        "terminal_feedback": terminal_feedback,
        "transcript_path": str(transcript_path),
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
    }
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(summary_md_path, report=report, transcript_path=transcript_path)
    return report


def _run_receive_capture(
    bridge: MA3OSCBridge,
    *,
    duration_seconds: float,
    ping_first: bool,
    trigger_command: str,
) -> dict[str, Any]:
    if ping_first:
        bridge.ping()
    start_index = len(bridge.messages)
    trigger_text = str(trigger_command or "").strip()
    if trigger_text:
        bridge._ensure_command_ready()  # noqa: SLF001
        bridge._send_command(trigger_text)  # noqa: SLF001
    sleep(max(0.05, float(duration_seconds)))
    captured_messages = bridge.messages[start_index:]
    transport_updates = [
        message.fields
        for message in captured_messages
        if str(getattr(message, "message_type", "")) == "transport"
    ]
    return {
        "duration_seconds": float(duration_seconds),
        "trigger_command": trigger_text or None,
        "message_count": len(captured_messages),
        "message_keys": [message.key for message in captured_messages],
        "transport_update_count": len(transport_updates),
        "latest_transport_update": transport_updates[-1] if transport_updates else None,
    }


def _run_stream(
    bridge: MA3OSCBridge,
    *,
    duration_seconds: float,
    ping_first: bool,
    trigger_command: str,
) -> int:
    if ping_first:
        bridge.ping()
    start_index = len(bridge.messages)
    trigger_text = str(trigger_command or "").strip()
    if trigger_text:
        bridge._ensure_command_ready()  # noqa: SLF001
        bridge._send_command(trigger_text)  # noqa: SLF001

    deadline = None
    if float(duration_seconds) > 0:
        deadline = monotonic() + float(duration_seconds)

    emitted = start_index
    try:
        while True:
            messages = bridge.messages
            for message in messages[emitted:]:
                print(json.dumps(_message_payload(message), sort_keys=True), flush=True)
            emitted = len(messages)
            if deadline is not None and monotonic() >= deadline:
                return 0
            sleep(0.05)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bridge, target = _bridge_from_args(args)
    command_transport = getattr(bridge, "_command_transport", None)
    terminal_feedback: dict[str, object] | None = None
    try:
        if args.command == "ping":
            payload = bridge.ping()
        elif args.command == "version":
            payload = bridge.get_version_info()
        elif args.command == "health":
            payload = bridge.get_plugin_health()
        elif args.command == "health-check":
            payload = _run_health_check(
                bridge,
                expected_root=args.expected_root,
                no_compare=bool(args.no_compare),
            )
        elif args.command == "validation-report":
            payload = _run_validation_report(
                bridge,
                args=args,
                target=target,
                output_dir=args.output_dir,
                expected_root=args.expected_root,
                no_compare=bool(args.no_compare),
                receive_duration_seconds=float(args.receive_duration_seconds),
                receive_trigger_command=str(args.receive_trigger_command or ""),
                command_transport=command_transport,
            )
        elif args.command == "receive-capture":
            payload = _run_receive_capture(
                bridge,
                duration_seconds=float(args.duration_seconds),
                ping_first=bool(args.ping_first),
                trigger_command=str(args.trigger_command or ""),
            )
        elif args.command == "stream":
            return _run_stream(
                bridge,
                duration_seconds=float(args.duration_seconds),
                ping_first=bool(args.ping_first),
                trigger_command=str(args.trigger_command or ""),
            )
        elif args.command == "reload":
            bridge.reload_plugins()
            payload = {"result": "sent", "command": "RP"}
        elif args.command == "timecodes":
            payload = [_timecode_payload(item) for item in bridge.list_timecodes()]
        elif args.command == "create-timecode":
            payload = _timecode_payload(
                bridge.create_timecode_next_available(
                    preferred_name=None if args.name is None else str(args.name),
                )
            )
        elif args.command == "track-groups":
            payload = [
                _track_group_payload(item)
                for item in bridge.list_track_groups(timecode_no=int(args.timecode_no))
            ]
        elif args.command == "create-track-group":
            payload = _track_group_payload(
                bridge.create_track_group_next_available(
                    timecode_no=int(args.timecode_no),
                    preferred_name=None if args.name is None else str(args.name),
                )
            )
        elif args.command == "tracks":
            payload = [
                _track_payload(item)
                for item in bridge.list_tracks(
                    timecode_no=args.timecode_no,
                    track_group_no=args.track_group_no,
                )
            ]
        elif args.command == "create-track":
            payload = _track_payload(
                bridge.create_track(
                    timecode_no=int(args.timecode_no),
                    track_group_no=int(args.track_group_no),
                    preferred_name=None if args.name is None else str(args.name),
                )
            )
        elif args.command == "events":
            payload = [
                _event_payload(item) for item in bridge.list_track_events(str(args.track_coord))
            ]
        elif args.command == "create-static-preset":
            payload = _preset_payload(
                bridge.create_static_preset(
                    preset_type_no=int(args.preset_type),
                    preset_no=int(args.preset_no),
                    store_mode=str(args.store_mode),
                    preset_name=str(args.name),
                    selection_command=str(args.selection_command),
                    value_command=str(args.value_command),
                )
            )
        elif args.command == "create-phaser-preset":
            payload = _preset_payload(
                bridge.create_phaser_preset(
                    preset_type_no=int(args.preset_type),
                    preset_no=int(args.preset_no),
                    store_mode=str(args.store_mode),
                    preset_name=str(args.name),
                    selection_command=str(args.selection_command),
                    step_preset_refs=[str(step) for step in args.steps],
                    speed_bpm=args.speed_bpm,
                )
            )
        elif args.command == "create-phaser-fixture-set":
            payload = _run_create_phaser_fixture_set(
                bridge,
                selection_command=str(args.selection_command),
                speed_bpm=float(args.speed_bpm),
                look_21_speed_bpm=float(args.look_21_speed_bpm),
            )
        elif args.command == "create-recipe-preset":
            payload = _preset_payload(
                bridge.create_recipe_preset(
                    preset_type_no=int(args.preset_type),
                    preset_no=int(args.preset_no),
                    store_mode=str(args.store_mode),
                    preset_name=str(args.name),
                    selection_command=str(args.selection_command),
                    source_preset_ref=str(args.source_preset_ref),
                    selection_mode=str(args.selection_mode),
                )
            )
        elif args.command == "edit-static-preset":
            payload = _preset_payload(
                bridge.edit_static_preset(
                    preset_type_no=int(args.preset_type),
                    preset_no=int(args.preset_no),
                    store_mode=str(args.store_mode),
                    preset_name=str(args.name),
                    selection_command=str(args.selection_command),
                    value_command=str(args.value_command),
                )
            )
        elif args.command == "edit-phaser-preset":
            payload = _preset_payload(
                bridge.edit_phaser_preset(
                    preset_type_no=int(args.preset_type),
                    preset_no=int(args.preset_no),
                    store_mode=str(args.store_mode),
                    preset_name=str(args.name),
                    selection_command=str(args.selection_command),
                    step_preset_refs=[str(step) for step in args.steps],
                    speed_bpm=args.speed_bpm,
                )
            )
        elif args.command == "edit-recipe-preset":
            payload = _preset_payload(
                bridge.edit_recipe_preset(
                    preset_type_no=int(args.preset_type),
                    preset_no=int(args.preset_no),
                    store_mode=str(args.store_mode),
                    preset_name=str(args.name),
                    selection_command=str(args.selection_command),
                    source_preset_ref=str(args.source_preset_ref),
                    selection_mode=str(args.selection_mode),
                )
            )
        elif args.command == "sequences":
            payload = [
                _sequence_payload(item)
                for item in bridge.list_sequences(start_no=args.start_no, end_no=args.end_no)
            ]
        elif args.command == "create-sequence-next-available":
            payload = _sequence_payload(
                bridge.create_sequence_next_available(
                    preferred_name=None if args.name is None else str(args.name),
                )
            )
        elif args.command == "create-sequence-in-current-song-range":
            payload = _sequence_payload(
                bridge.create_sequence_in_current_song_range(
                    preferred_name=None if args.name is None else str(args.name),
                )
            )
        elif args.command == "sequence-cues":
            payload = bridge.list_sequence_cues(sequence_no=int(args.sequence_no))
        elif args.command == "analyze-cue-recipe-state":
            payload = bridge.analyze_cue_recipe_state(
                sequence_no=int(args.sequence_no),
                cue_no=str(args.cue_no),
            )
        elif args.command == "preview-recipe-cue-only":
            payload = bridge.preview_recipe_cue_only(
                sequence_no=int(args.sequence_no),
                source_cue_no=str(args.source_cue_no),
                target_cue_no=str(args.target_cue_no),
            )
        elif args.command == "preview-copy-cue-with-status":
            payload = bridge.preview_copy_cue_with_status(
                sequence_no=int(args.sequence_no),
                source_cue_no=str(args.source_cue_no),
                dest_cue_no=str(args.dest_cue_no),
            )
        elif args.command == "datapool-children":
            payload = bridge.list_datapool_objects(path=args.path)
        elif args.command == "datapool-object":
            payload = bridge.describe_datapool_object(path=args.path)
            payload.pop("dump", None)
            payload.pop("property_items", None)
            payload.pop("property_count", None)
            payload.pop("properties_truncated", None)
        elif args.command == "datapool-report":
            payload = _run_datapool_report(
                bridge,
                path=args.path,
                preset_ref=args.preset_ref,
                depth=int(args.depth),
            )
        elif args.command == "list-presets":
            payload = _run_list_presets(
                bridge,
                preset_type_no=int(args.preset_type),
            )
        elif args.command == "describe-preset":
            payload = _run_describe_preset(
                bridge,
                preset_ref=args.preset_ref,
            )
        elif args.command == "preview-replace-preset-when-group":
            payload = _run_preview_replace_preset_when_group(
                bridge,
                preset_type_no=int(args.preset_type),
                source_preset_ref=str(args.source_preset_ref),
                dest_preset_ref=str(args.dest_preset_ref),
                group_filter=str(args.group_filter),
                sequence_numbers=str(args.sequence_numbers),
            )
        elif args.command == "replace-preset-when-group":
            payload = _run_replace_preset_when_group(
                bridge,
                preset_type_no=int(args.preset_type),
                source_preset_ref=str(args.source_preset_ref),
                dest_preset_ref=str(args.dest_preset_ref),
                group_filter=str(args.group_filter),
                sequence_numbers=str(args.sequence_numbers),
            )
        elif args.command == "analyze-cue-recipe-state":
            payload = _run_analyze_cue_recipe_state(
                bridge,
                sequence_no=int(args.sequence_no),
                cue_no=str(args.cue_no),
            )
        elif args.command == "preview-recipe-cue-only":
            payload = _run_preview_recipe_cue_only(
                bridge,
                sequence_no=int(args.sequence_no),
                source_cue_no=str(args.source_cue_no),
                target_cue_no=str(args.target_cue_no),
            )
        elif args.command == "apply-recipe-cue-only":
            payload = _run_apply_recipe_cue_only(
                bridge,
                sequence_no=int(args.sequence_no),
                source_cue_no=str(args.source_cue_no),
                target_cue_no=str(args.target_cue_no),
            )
        elif args.command == "copy-cue-with-status":
            payload = _run_copy_cue_with_status(
                bridge,
                sequence_no=int(args.sequence_no),
                source_cue_no=str(args.source_cue_no),
                dest_cue_no=str(args.dest_cue_no),
            )
        elif args.command == "current-song-range":
            payload = _sequence_range_payload(bridge.get_current_song_sequence_range())
        elif args.command == "smoke":
            payload = _run_smoke(bridge)
        else:
            raise SystemExit(f"Unsupported command: {args.command}")

        result = {
            "target": target,
            "command": args.command,
            "result": payload,
        }
        if args.command != "validation-report" and bool(args.terminal_feedback):
            terminal_feedback = _run_terminal_feedback(args)
        if terminal_feedback is not None:
            result["terminal_feedback"] = terminal_feedback
        if args.command in {"datapool-report", "describe-preset"} and not args.json:
            print(str(payload["text"]), flush=True)
        else:
            _print_output(result, as_json=bool(args.json))
        return 0
    finally:
        if args.transcript_out is not None and args.command != "validation-report":
            commands = (
                []
                if command_transport is None
                else list(getattr(command_transport, "commands", []))
            )
            _write_transcript_with_terminal_feedback(
                args.transcript_out,
                target=target,
                commands=commands,
                bridge=bridge,
                terminal_feedback=terminal_feedback,
            )
        bridge.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
