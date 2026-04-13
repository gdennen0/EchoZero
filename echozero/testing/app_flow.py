from __future__ import annotations

import os
import tempfile
from collections import deque
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from echozero.application.shared.enums import SyncMode
from echozero.testing.ma3 import OSCLoopback, OSCMessageCapture, SimulatedMA3Bridge
from echozero.ui.qt.app_shell import AppRuntimeProfile, AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.widget import TimelineWidget
from run_echozero import LauncherController

_APPFLOW_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/echozero_appflow")


class AppFlowHarness:
    def __init__(
        self,
        *,
        simulate_ma3: bool = False,
        simulate_ma3_osc: bool = False,
        working_dir_root: Path | None = None,
        initial_project_name: str = "EchoZero Test Project",
    ) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        self._app = QApplication.instance() or QApplication([])
        self._temp_root_handle = None
        if working_dir_root is None:
            _APPFLOW_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            self._temp_root_handle = tempfile.TemporaryDirectory(
                prefix="echozero_appflow_",
                dir=str(_APPFLOW_TEMP_ROOT),
            )
            working_dir_root = Path(self._temp_root_handle.name) / "working"
        self._working_dir_root = Path(working_dir_root)
        self._working_dir_root.mkdir(parents=True, exist_ok=True)
        self._dialog_root = self._working_dir_root.parent
        self.ma3_bridge = SimulatedMA3Bridge() if simulate_ma3 else None
        self.ma3_osc_loopback = OSCLoopback().start() if simulate_ma3_osc else None
        runtime = build_app_shell(
            profile=AppRuntimeProfile.TEST,
            sync_bridge=self.ma3_bridge,
            working_dir_root=self._working_dir_root,
            initial_project_name=initial_project_name,
        )
        if not isinstance(runtime, AppShellRuntime):
            raise TypeError("AppFlowHarness requires canonical AppShellRuntime")
        self.runtime = runtime
        self.widget = TimelineWidget(
            self.runtime.presentation(),
            on_intent=self.runtime.dispatch,
            runtime_audio=self.runtime.runtime_audio,
        )
        self.widget.resize(1440, 720)
        self.launcher = LauncherController(runtime=self.runtime, widget=self.widget)
        self.launcher.install()
        self._open_paths: deque[Path] = deque()
        self._save_paths: deque[Path] = deque()
        self.launcher._choose_open_path = self._choose_open_path  # type: ignore[method-assign]
        self.launcher._choose_save_path = self._choose_save_path  # type: ignore[method-assign]
        self.widget.show()
        self._app.processEvents()

    def presentation(self):
        return self.runtime.presentation()

    def dispatch(self, intent):
        presentation = self.runtime.dispatch(intent)
        self.widget.set_presentation(presentation)
        self._app.processEvents()
        return presentation

    def trigger_action(self, action_id: str):
        action_map = {
            "new": "new_project",
            "open": "open_project",
            "save": "save_project",
            "save_as": "save_project_as",
        }
        action_key = action_map.get(action_id, action_id)
        action = self.launcher.actions[action_key]
        action.trigger()
        self._app.processEvents()
        return self.runtime.presentation()

    def queue_open_path(self, path: str | Path) -> Path:
        queued = Path(path)
        self._open_paths.append(queued)
        return queued

    def queue_save_path(self, path: str | Path) -> Path:
        queued = Path(path)
        self._save_paths.append(queued)
        return queued

    def enable_sync(self, mode: SyncMode = SyncMode.MA3):
        state = self.runtime.enable_sync(mode)
        self._app.processEvents()
        return state

    def disable_sync(self):
        state = self.runtime.disable_sync()
        self._app.processEvents()
        return state

    def send_ma3_osc(self, path: str, *args: object) -> None:
        if self.ma3_osc_loopback is None:
            raise RuntimeError("MA3 OSC loopback is not enabled")
        self.ma3_osc_loopback.send(path, *args)

    def wait_for_ma3_osc(self, path: str, timeout: float = 1.0) -> OSCMessageCapture | None:
        if self.ma3_osc_loopback is None:
            raise RuntimeError("MA3 OSC loopback is not enabled")
        return self.ma3_osc_loopback.wait_for(path, timeout=timeout)

    def ma3_osc_messages(self) -> list[OSCMessageCapture]:
        if self.ma3_osc_loopback is None:
            raise RuntimeError("MA3 OSC loopback is not enabled")
        return self.ma3_osc_loopback.captures()

    def clear_ma3_osc(self) -> None:
        if self.ma3_osc_loopback is None:
            raise RuntimeError("MA3 OSC loopback is not enabled")
        self.ma3_osc_loopback.clear()

    @property
    def project_path(self) -> Path | None:
        return self.runtime.project_path

    @property
    def is_dirty(self) -> bool:
        return self.runtime.is_dirty

    def close(self) -> None:
        self.widget.close()
        self._app.processEvents()

    def shutdown(self) -> None:
        try:
            self.close()
            self.runtime.shutdown()
        finally:
            if self.ma3_osc_loopback is not None:
                self.ma3_osc_loopback.stop()
            if self._temp_root_handle is not None:
                self._temp_root_handle.cleanup()
                self._temp_root_handle = None

    def _choose_open_path(self) -> Path | None:
        if self._open_paths:
            return self._open_paths.popleft()
        current = self.project_path
        if current is not None:
            return current
        return None

    def _choose_save_path(self) -> Path | None:
        if self._save_paths:
            return self._save_paths.popleft()
        current = self.project_path
        if current is not None:
            return current
        return self._dialog_root / "appflow-project.ez"
