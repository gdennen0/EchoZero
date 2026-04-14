from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from echozero.testing.gui_lane_b import run_scenario_file

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_gui_lane_b")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_lane_b_runner_executes_starter_scenario_and_writes_trace():
    temp_root = _repo_local_temp_root()
    output_dir = temp_root / "artifacts"

    try:
        trace = run_scenario_file(
            scenario_path=Path("tests/gui/scenarios/e2e_core.json"),
            output_dir=output_dir,
        )

        assert trace
        assert [step["status"] for step in trace] == ["passed"] * len(trace)

        push_step = next(step for step in trace if step["action"] == "open_push_surface")
        pull_step = next(step for step in trace if step["action"] == "open_pull_surface")
        enable_step = next(step for step in trace if step["action"] == "enable_sync")
        disable_step = next(step for step in trace if step["action"] == "disable_sync")
        classify_step = next(step for step in trace if step["action"] == "classify_drum_events")

        assert push_step["snapshot"]["push_mode_active"] is True
        assert push_step["snapshot"]["batch_transfer_plan_id"] is not None
        assert pull_step["snapshot"]["pull_workspace_active"] is True
        assert pull_step["snapshot"]["batch_transfer_plan_id"] is not None
        assert enable_step["snapshot"]["sync_connected"] is True
        assert enable_step["snapshot"]["sync_mode"] == "ma3"
        assert disable_step["snapshot"]["sync_connected"] is False
        assert disable_step["snapshot"]["sync_mode"] == "none"
        assert any(layer["title"] == "Drums" for layer in classify_step["snapshot"]["layers"])
        assert any(layer["title"] == "Drum_Classified_Events" for layer in classify_step["snapshot"]["layers"])

        trace_path = output_dir / "trace.json"
        screenshot_path = output_dir / "lane-b-final.png"

        assert trace_path.exists()
        assert screenshot_path.exists()

        written_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert written_trace == trace
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
