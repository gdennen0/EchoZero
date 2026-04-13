from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from echozero.application.timeline.intents import ToggleLayerExpanded
from echozero.testing.app_flow import AppFlowHarness

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_app_flow_harness")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_app_flow_harness_dispatch_and_launcher_actions():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")

    try:
        layer_id = harness.presentation().layers[0].layer_id

        harness.dispatch(ToggleLayerExpanded(layer_id))
        assert harness.is_dirty is True

        save_path = harness.queue_save_path(temp_root / "saved-project.ez")
        harness.trigger_action("save_as")

        assert harness.project_path == save_path
        assert save_path.exists()
        assert harness.is_dirty is False

        harness.trigger_action("new")
        assert harness.project_path is None
        assert harness.is_dirty is False
        assert harness.presentation().title == "EchoZero Project"
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_sync_with_simulated_ma3_connects_bridge():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3=True, working_dir_root=temp_root / "working-sync")

    try:
        state = harness.enable_sync()
        assert harness.ma3_bridge is not None
        assert state.connected is True
        assert state.mode.value == "ma3"
        assert harness.ma3_bridge.connected is True
        assert harness.ma3_bridge.connect_calls == 1

        disabled = harness.disable_sync()
        assert disabled.connected is False
        assert disabled.mode.value == "none"
        assert harness.ma3_bridge.connected is False
        assert harness.ma3_bridge.disconnect_calls == 1
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)
