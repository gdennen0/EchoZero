#!/usr/bin/env python3
"""Canonical app-path MA3 smoke for non-destructive live localhost validation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MA3.dev.ma3_harness_common import build_bridge  # noqa: E402
from echozero.application.shared.enums import LayerKind  # noqa: E402
from echozero.application.shared.ranges import TimeRange  # noqa: E402
from echozero.application.timeline.intents import CreateEvent, OpenPullFromMA3Dialog  # noqa: E402
from echozero.testing.app_flow import AppFlowHarness  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one canonical non-destructive app-path MA3 smoke against a live target.",
    )
    parser.add_argument("--ma3-host", default=None)
    parser.add_argument("--ma3-port", type=int, default=None)
    parser.add_argument("--command-path", default=None)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--settings-path", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--keep-working-dir", action="store_true")
    return parser


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

    temp_root = Path(
        tempfile.mkdtemp(prefix="echozero_ma3_app_path_", dir=str(REPO_ROOT / "artifacts"))
    )
    keep_working_dir = bool(args.keep_working_dir)
    harness = AppFlowHarness(sync_bridge=bridge, working_dir_root=temp_root / "working")
    try:
        state = harness.enable_sync()
        presentation = harness.runtime.add_layer(LayerKind.EVENT, "MA3 Pull Smoke Target")
        harness.widget.set_presentation(presentation)
        harness._app.processEvents()
        layer_id = harness.presentation().layers[0].layer_id
        harness.dispatch(
            CreateEvent(
                layer_id=layer_id,
                take_id=None,
                time_range=TimeRange(0.25, 0.5),
            )
        )
        harness.dispatch(OpenPullFromMA3Dialog())
        flow = harness.presentation().manual_pull_flow
        result: dict[str, Any] = {
            "target": target,
            "sync_state": {
                "connected": bool(state.connected),
                "mode": getattr(state.mode, "value", str(state.mode)),
            },
            "pull_workspace": {
                "workspace_active": bool(flow.workspace_active),
                "selected_timecode_no": flow.selected_timecode_no,
                "timecode_count": len(flow.available_timecodes),
                "track_count": len(flow.available_tracks),
                "source_track_count": len(flow.selected_source_track_coords),
                "available_target_count": len(flow.available_target_layers),
            },
            "recent_ma3_messages": harness.runtime.recent_ma3_osc_messages(limit=12),
            "working_dir_kept": keep_working_dir,
        }
        if keep_working_dir:
            result["working_dir_root"] = str(temp_root)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        else:
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        harness.shutdown()
        if not keep_working_dir:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
