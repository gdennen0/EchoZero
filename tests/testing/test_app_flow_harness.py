from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from echozero.application.timeline.intents import ToggleLayerExpanded
from echozero.testing.app_flow import AppFlowHarness
from echozero.testing.analysis_mocks import write_test_wav

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_app_flow_harness")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_app_flow_harness_dispatch_and_launcher_actions():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "appflow-harness.wav")
        harness.runtime.add_song_from_path("Harness Song", audio_path)

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


def test_app_flow_harness_osc_loopback_helpers():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3_osc=True, working_dir_root=temp_root / "working-osc")

    try:
        harness.send_ma3_osc("/ma3/exec", 7, "flash")
        capture = harness.wait_for_ma3_osc("/ma3/exec", timeout=1.0)

        assert harness.ma3_osc_loopback is not None
        assert harness.ma3_osc_loopback.is_running is True
        assert capture is not None
        assert capture.args == (7, "flash")
        assert [message.path for message in harness.ma3_osc_messages()] == ["/ma3/exec"]

        harness.clear_ma3_osc()
        assert harness.ma3_osc_messages() == []
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_shutdown_stops_osc_loopback():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3_osc=True, working_dir_root=temp_root / "working-osc-stop")

    loopback = harness.ma3_osc_loopback
    assert loopback is not None
    thread = loopback.thread

    harness.shutdown()

    try:
        assert loopback.is_running is False
        assert thread is not None
        assert thread.is_alive() is False
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
