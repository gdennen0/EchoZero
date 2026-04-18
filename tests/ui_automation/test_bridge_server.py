from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from ui_automation import LiveEchoZeroAutomationProvider

from echozero.testing.app_flow import AppFlowHarness
from echozero.testing.analysis_mocks import write_test_wav
from echozero.ui.qt.automation_bridge import AutomationBridgeServer

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_echozero_bridge_server")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_bridge_server_exposes_health_and_snapshot():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")
    bridge = AutomationBridgeServer(
        runtime=harness.runtime,
        widget=harness.widget,
        launcher=harness.launcher,
        app=harness._app,
        port=0,
    )
    bridge.start()

    try:
        backend = LiveEchoZeroAutomationProvider(f"http://{bridge.address[0]}:{bridge.address[1]}").attach()
        health = _run_with_qt_events(lambda: backend.health())
        snapshot = _run_with_qt_events(lambda: backend.snapshot())

        assert health["ok"] is True
        assert snapshot.app == "EchoZero"
        assert any(target.target_id == "shell.timeline" for target in snapshot.targets)
    finally:
        bridge.stop()
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_bridge_server_preserves_focus_and_object_snapshot_fields():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")
    bridge = AutomationBridgeServer(
        runtime=harness.runtime,
        widget=harness.widget,
        launcher=harness.launcher,
        app=harness._app,
        port=0,
    )
    bridge.start()

    try:
        backend = LiveEchoZeroAutomationProvider(f"http://{bridge.address[0]}:{bridge.address[1]}").attach()
        clicked = _run_with_qt_events(lambda: backend.click("shell.timeline"))
        snapshot = _run_with_qt_events(lambda: backend.snapshot())

        assert clicked.focused_target_id == "shell.timeline"
        assert snapshot.focused_target_id == "shell.timeline"
        assert snapshot.focused_object_id is None
        assert snapshot.objects == ()
    finally:
        bridge.stop()
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_bridge_server_drives_project_lifecycle_actions():
    temp_root = _repo_local_temp_root()
    save_path = temp_root / "projects" / "bridge-lifecycle.ez"
    audio_path = write_test_wav(temp_root / "fixtures" / "bridge-lifecycle.wav")
    harness = AppFlowHarness(
        working_dir_root=temp_root / "working",
        initial_project_name="Bridge Lifecycle Start",
    )
    bridge = AutomationBridgeServer(
        runtime=harness.runtime,
        widget=harness.widget,
        launcher=harness.launcher,
        app=harness._app,
        port=0,
    )
    bridge.start()

    try:
        backend = LiveEchoZeroAutomationProvider(f"http://{bridge.address[0]}:{bridge.address[1]}").attach()

        after_new = _run_with_qt_events(
            lambda: backend.invoke("app.new", params={"name": "Bridge Lifecycle Project"})
        )
        assert after_new.artifacts["project_title"] == "Bridge Lifecycle Project"

        after_import = _run_with_qt_events(
            lambda: backend.invoke(
                "add_song_from_path",
                params={"title": "Bridge Song", "audio_path": str(audio_path)},
            )
        )
        assert any(target.label == "Bridge Song" for target in after_import.targets)

        after_save = _run_with_qt_events(
            lambda: backend.invoke("app.save_as", params={"path": str(save_path)})
        )
        assert save_path.exists()
        assert after_save.artifacts["project_title"] == "Bridge Lifecycle Project"

        reopened = _run_with_qt_events(
            lambda: backend.invoke("app.open", params={"path": str(save_path)})
        )
        assert reopened.artifacts["project_title"] == "Bridge Lifecycle Project"
        assert any(target.label == "Bridge Song" for target in reopened.targets)
    finally:
        bridge.stop()
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)
def _run_with_qt_events(callback):
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result["value"] = callback()
        except BaseException as exc:  # pragma: no cover - surfaced to test
            error["value"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    deadline = time.time() + 10.0
    app = QApplication.instance()
    while not done.is_set():
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for bridge client call")
        if app is not None:
            app.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    if "value" in error:
        raise error["value"]
    return result.get("value")
