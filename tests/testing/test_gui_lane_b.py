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
        push_apply_step = next(step for step in trace if step["label"] == "Apply push transfer plan")
        pull_step = next(step for step in trace if step["action"] == "open_pull_surface")
        pull_apply_step = next(step for step in trace if step["label"] == "Apply pull transfer plan")
        enable_step = next(step for step in trace if step["action"] == "enable_sync")
        disable_step = next(step for step in trace if step["action"] == "disable_sync")
        classify_step = next(step for step in trace if step["action"] == "classify_drum_events")

        assert push_step["snapshot"]["push_mode_active"] is True
        assert push_step["snapshot"]["batch_transfer_plan_id"] is not None
        assert push_apply_step["status"] == "passed"
        assert push_apply_step["snapshot"]["batch_transfer_plan_id"] == push_step["snapshot"]["batch_transfer_plan_id"]

        assert pull_step["snapshot"]["pull_workspace_active"] is True
        assert pull_step["snapshot"]["batch_transfer_plan_id"] is not None
        assert pull_apply_step["status"] == "passed"
        assert pull_apply_step["snapshot"]["batch_transfer_plan_id"] == pull_step["snapshot"]["batch_transfer_plan_id"]
        assert pull_apply_step["snapshot"]["batch_transfer_plan_id"] != push_step["snapshot"]["batch_transfer_plan_id"]

        assert enable_step["snapshot"]["sync_connected"] is True
        assert enable_step["snapshot"]["sync_mode"] == "ma3"
        assert disable_step["snapshot"]["sync_connected"] is False
        assert disable_step["snapshot"]["sync_mode"] == "none"
        assert any(layer["title"] == "Drums" for layer in classify_step["snapshot"]["layers"])
        assert any(layer["title"] == "Drum_Classified_Events" for layer in classify_step["snapshot"]["layers"])

        trace_path = output_dir / "trace.json"
        screenshot_path = output_dir / "lane-b-final.png"
        post_stems_path = output_dir / "lane-b-post-stems.png"
        post_drum_events_path = output_dir / "lane-b-post-drum-events.png"
        post_classification_path = output_dir / "lane-b-post-classification.png"
        post_push_apply_path = output_dir / "lane-b-post-push-apply.png"

        assert trace_path.exists()
        assert screenshot_path.exists()
        assert post_stems_path.exists()
        assert post_drum_events_path.exists()
        assert post_classification_path.exists()
        assert post_push_apply_path.exists()

        written_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert written_trace == trace
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_lane_b_runner_writes_simulated_video_manifest(monkeypatch):
    temp_root = _repo_local_temp_root()
    output_dir = temp_root / "artifacts"

    def fake_write_frame_video(frame_paths, output_path, *, fps):
        assert frame_paths
        assert fps == 6
        output_path.write_bytes(b"fake-mp4")
        return output_path

    monkeypatch.setattr("echozero.testing.gui_lane_b._write_frame_video", fake_write_frame_video)

    try:
        trace = run_scenario_file(
            scenario_path=Path("tests/gui/scenarios/e2e_core.json"),
            output_dir=output_dir,
            record_video=True,
            fps=6,
        )

        assert trace
        assert (output_dir / "gui-lane-b-simulated.mp4").exists()
        artifacts = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
        assert artifacts["video"] == "gui-lane-b-simulated.mp4"
        assert artifacts["frame_count"] >= len(trace)
        assert artifacts["proof_classification"] == "simulated_gui_capture"
        assert artifacts["operator_demo_valid"] is False
        assert artifacts["analysis_mode"] == "mock"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_lane_b_runner_can_disable_mock_analysis(monkeypatch):
    temp_root = _repo_local_temp_root()
    output_dir = temp_root / "artifacts"
    captured: dict[str, object] = {}

    class _FakeHarness:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        widget = None
        launcher = type("_L", (), {"confirm_close": None})()
        runtime = type("_R", (), {"_is_dirty": False})()

        def shutdown(self):
            return None

    monkeypatch.setattr("echozero.testing.gui_lane_b.AppFlowHarness", _FakeHarness)
    monkeypatch.setattr("echozero.testing.gui_lane_b._render_for_hit_testing", lambda harness: None)
    monkeypatch.setattr("echozero.testing.gui_lane_b.load_scenario", lambda path: type("S", (), {"name": "real", "steps": []})())

    try:
        trace = run_scenario_file(
            scenario_path=Path("tests/gui/scenarios/e2e_core.json"),
            output_dir=output_dir,
            use_mock_analysis=False,
        )

        assert trace == []
        assert captured["analysis_service"] is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
