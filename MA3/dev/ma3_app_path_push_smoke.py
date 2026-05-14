#!/usr/bin/env python3
"""Canonical app-path MA3 push smoke for one bounded live-target proof.
Exists to capture real widget/app push evidence without widening the harness surface.
Connects the app-flow send action to one explicit MA3 target track and reports the write result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from time import monotonic, sleep
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_harness_common import build_bridge  # noqa: E402
from echozero.application.presentation.inspector_contract import (  # noqa: E402
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.shared.enums import LayerKind  # noqa: E402
from echozero.application.shared.ranges import TimeRange  # noqa: E402
from echozero.application.timeline.intents import CreateEvent  # noqa: E402
from echozero.application.timeline.ma3_push_intents import (  # noqa: E402
    CreateMA3Sequence,
    MA3PushApplyMode,
    MA3SequenceCreationMode,
)
from echozero.testing.app_flow import AppFlowHarness  # noqa: E402
from echozero.ui.qt.timeline.widget_action_ma3_push_mixin import (  # noqa: E402
    _ManualPushRoutePopupResult,
    TimelineWidgetMA3PushActionMixin,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one canonical bounded app-path MA3 push smoke against a live target.",
    )
    parser.add_argument("--ma3-host", default=None)
    parser.add_argument("--ma3-port", type=int, default=None)
    parser.add_argument("--command-path", default=None)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--settings-path", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--target-track-coord", required=True)
    parser.add_argument(
        "--sequence-mode",
        choices=("none", "next-available", "current-song-range"),
        default="none",
        help="How to prepare the target when it has no assigned MA3 sequence.",
    )
    parser.add_argument(
        "--apply-mode",
        choices=("merge", "overwrite"),
        default="merge",
    )
    parser.add_argument("--cue-number", type=float, default=901.0)
    parser.add_argument("--event-start", type=float, default=9.01)
    parser.add_argument("--event-end", type=float, default=9.51)
    parser.add_argument("--wait-seconds", type=float, default=2.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--keep-working-dir", action="store_true")
    return parser


def _layer_contract_action(harness: AppFlowHarness, layer_id: object, action_id: str):
    contract = build_timeline_inspector_contract(
        harness.widget.presentation,
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=layer_id),
    )
    return next(
        action
        for section in contract.context_sections
        for action in section.actions
        if action.action_id == action_id and action.params.get("direction") == "push"
    )


def _sequence_action_for_mode(
    *,
    mode: str,
    preferred_name: str,
) -> CreateMA3Sequence | None:
    if mode == "none":
        return None
    creation_mode = (
        MA3SequenceCreationMode.NEXT_AVAILABLE
        if mode == "next-available"
        else MA3SequenceCreationMode.CURRENT_SONG_RANGE
    )
    return CreateMA3Sequence(
        creation_mode=creation_mode,
        preferred_name=preferred_name,
    )


def _wait_for_push_completion(harness: AppFlowHarness, *, timeout: float) -> tuple[str, str]:
    deadline = monotonic() + max(0.1, timeout)
    while monotonic() < deadline:
        flow = harness.presentation().manual_push_flow
        status = str(flow.operation_status or "idle").strip().lower()
        message = str(flow.operation_message or "").strip()
        if status in {"success", "error", "idle"} and not str(flow.operation_id or "").strip():
            return status, message
        harness.widget._on_runtime_tick()
        harness._app.processEvents()
        sleep(0.02)
    flow = harness.presentation().manual_push_flow
    return (
        str(flow.operation_status or "idle").strip().lower(),
        str(flow.operation_message or "").strip(),
    )


def _event_payload(event: Any) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "label": event.label,
        "start": event.start,
        "end": event.end,
        "cmd": event.cmd,
        "cue_number": event.cue_number,
        "cue_ref": event.cue_ref,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bridge, target = build_bridge(
        ma3_host=args.ma3_host,
        ma3_port=args.ma3_port,
        command_path=args.command_path,
        settings_path=args.settings_path,
        listen_host=str(args.listen_host or "0.0.0.0"),
        listen_port=int(args.listen_port),
        timeout=float(args.timeout),
    )

    target_track_coord = str(args.target_track_coord).strip()
    if not target_track_coord:
        raise SystemExit("--target-track-coord is required")

    target_track_before = next(
        (track for track in bridge.list_tracks() if track.coord == target_track_coord),
        None,
    )
    if target_track_before is None:
        raise SystemExit(f"Target MA3 track was not found: {target_track_coord}")

    cue_number = float(args.cue_number)
    remote_events_before = bridge.list_track_events(target_track_coord)
    if any(float(event.cue_number or 0) == cue_number for event in remote_events_before):
        raise SystemExit(
            f"Target track {target_track_coord} already contains cue {cue_number:g}; "
            "pick a different --cue-number for a bounded smoke write."
        )
    if target_track_before.sequence_no is None and args.sequence_mode == "none":
        raise SystemExit(
            f"Target track {target_track_coord} has no assigned sequence; "
            "pass --sequence-mode next-available or current-song-range."
        )

    artifacts_root = REPO_ROOT / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix="echozero_ma3_app_path_push_", dir=str(artifacts_root))
    )
    keep_working_dir = bool(args.keep_working_dir)
    harness = AppFlowHarness(sync_bridge=bridge, working_dir_root=temp_root / "working")
    try:
        state = harness.enable_sync()
        layer_title = f"MA3 Push Smoke {cue_number:g}"
        presentation = harness.runtime.add_layer(LayerKind.EVENT, layer_title)
        harness.widget.set_presentation(presentation)
        harness._app.processEvents()
        layer_id = harness.presentation().layers[0].layer_id
        harness.dispatch(
            CreateEvent(
                layer_id=layer_id,
                take_id=None,
                time_range=TimeRange(float(args.event_start), float(args.event_end)),
                label=f"Smoke Cue {cue_number:g}",
                cue_number=cue_number,
            )
        )

        original_open_popup = TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup
        TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup = (
            lambda *_a, **_k: _ManualPushRoutePopupResult(
                target_track_coord=target_track_coord,
                sequence_action=_sequence_action_for_mode(
                    mode=str(args.sequence_mode),
                    preferred_name=layer_title,
                ),
                apply_mode=MA3PushApplyMode(str(args.apply_mode)),
            )
        )
        try:
            harness.widget._trigger_contract_action(
                _layer_contract_action(
                    harness,
                    layer_id,
                    "transfer.workspace_open",
                )
            )
        finally:
            TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup = original_open_popup

        operation_status, operation_message = _wait_for_push_completion(
            harness,
            timeout=float(args.wait_seconds),
        )
        target_track_after = next(
            track for track in bridge.list_tracks() if track.coord == target_track_coord
        )
        remote_events_after = bridge.list_track_events(target_track_coord)
        pushed_event = next(
            (event for event in remote_events_after if float(event.cue_number or 0) == cue_number),
            None,
        )
        result: dict[str, Any] = {
            "target": target,
            "sync_state": {
                "connected": bool(state.connected),
                "mode": getattr(state.mode, "value", str(state.mode)),
            },
            "push_target": {
                "track_coord": target_track_coord,
                "track_name": target_track_after.name,
                "apply_mode": str(args.apply_mode),
                "sequence_mode": str(args.sequence_mode),
                "sequence_before": target_track_before.sequence_no,
                "sequence_after": target_track_after.sequence_no,
                "event_count_before": len(remote_events_before),
                "event_count_after": len(remote_events_after),
            },
            "push_event": {
                "cue_number": cue_number,
                "label": f"Smoke Cue {cue_number:g}",
                "present_after_push": pushed_event is not None,
                "remote_snapshot": None if pushed_event is None else _event_payload(pushed_event),
            },
            "push_operation": {
                "status": operation_status,
                "message": operation_message,
                "saved_route": harness.presentation().layers[0].sync_target_label,
            },
            "recent_ma3_messages": harness.runtime.recent_ma3_osc_messages(limit=12),
            "working_dir_kept": keep_working_dir,
        }
        if keep_working_dir:
            result["working_dir_root"] = str(temp_root)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if operation_status == "success" and pushed_event is not None else 1
    finally:
        harness.shutdown()
        if not keep_working_dir:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
