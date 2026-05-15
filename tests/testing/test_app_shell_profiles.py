from __future__ import annotations

import inspect
import shutil
import uuid
from pathlib import Path

from echozero.application.shared.enums import SyncMode
import echozero.ui.qt.launcher_surface as launcher_surface
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.launcher_surface import LauncherSurface, build_launcher_surface

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_app_shell_profiles")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_build_app_shell_builds_canonical_runtime():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "runtime")

    try:
        assert isinstance(runtime, AppShellRuntime)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_launcher_surface_builds_canonical_runtime_surface(monkeypatch):
    temp_root = _repo_local_temp_root()

    class FakeWidget:
        def __init__(
            self,
            presentation,
            *,
            on_intent,
            runtime_audio,
            initial_header_width,
            app_settings_service=None,
        ) -> None:
            self.presentation = presentation
            self.on_intent = on_intent
            self.runtime_audio = runtime_audio
            self.initial_header_width = initial_header_width
            self.app_settings_service = app_settings_service

        def setObjectName(self, _name: str) -> None:
            pass

        def resize(self, _width: int, _height: int) -> None:
            pass

        def setWindowTitle(self, _title: str) -> None:
            pass

        def setWindowModified(self, _modified: bool) -> None:
            pass

        def setWindowFilePath(self, _path: str) -> None:
            pass

        def set_window_title_sync_callback(self, _callback) -> None:
            pass

        def addAction(self, _action) -> None:
            pass

    monkeypatch.setattr(launcher_surface, "TimelineWidget", FakeWidget)
    monkeypatch.setattr(launcher_surface.LauncherController, "install", lambda self: None)
    surface = build_launcher_surface(working_dir_root=temp_root / "surface")

    try:
        assert isinstance(surface, LauncherSurface)
        assert isinstance(surface.runtime, AppShellRuntime)
    finally:
        surface.runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_build_app_shell_has_no_profile_split():
    signature = inspect.signature(build_app_shell)
    assert "profile" not in signature.parameters


def test_app_shell_module_does_not_route_through_demo_app():
    source = Path("/Users/march/Documents/GitHub/EchoZero/echozero/ui/qt/app_shell.py").read_text(
        encoding="utf-8"
    )
    assert "timeline.demo_app" not in source


def test_runtime_surfaces_recent_ma3_osc_messages() -> None:
    temp_root = _repo_local_temp_root()

    class _FakeMessage:
        def __init__(
            self, message_type: str, change: str, *, timestamp: float, fields: dict[str, object]
        ) -> None:
            self.message_type = message_type
            self.change = change
            self.timestamp = timestamp
            self.fields = fields
            self.raw_payload = "raw"

    class _FakeBridge:
        def __init__(self) -> None:
            self.messages = [
                _FakeMessage("connection", "ping", timestamp=1.0, fields={"status": "ok"}),
                _FakeMessage(
                    "transport", "scrubbed", timestamp=2.0, fields={"tc": 112, "to_seconds": 9.5}
                ),
            ]

    runtime = build_app_shell(
        working_dir_root=temp_root / "runtime-messages",
        sync_bridge=_FakeBridge(),
    )

    try:
        rows = runtime.recent_ma3_osc_messages(limit=1)
        assert len(rows) == 1
        assert rows[0]["message_type"] == "transport"
        assert rows[0]["change"] == "scrubbed"
        assert rows[0]["fields"] == {"tc": 112, "to_seconds": 9.5}
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_runtime_clears_recent_ma3_osc_messages() -> None:
    temp_root = _repo_local_temp_root()

    class _FakeMessage:
        message_type = "connection"
        change = "ping"
        timestamp = 1.0
        fields = {"status": "ok"}
        raw_payload = "raw"

    class _FakeBridge:
        def __init__(self) -> None:
            self._messages = [_FakeMessage()]

        @property
        def messages(self) -> list[_FakeMessage]:
            return list(self._messages)

        def clear_messages(self) -> None:
            self._messages.clear()

    runtime = build_app_shell(
        working_dir_root=temp_root / "runtime-clear-messages",
        sync_bridge=_FakeBridge(),
    )

    try:
        assert runtime.recent_ma3_osc_messages()
        runtime.clear_ma3_osc_messages()
        assert runtime.recent_ma3_osc_messages() == []
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_runtime_prefers_low_latency_transport_poll_only_when_ma3_connected() -> None:
    temp_root = _repo_local_temp_root()

    class _FakeBridge:
        pass

    runtime = build_app_shell(
        working_dir_root=temp_root / "runtime-sync-cadence",
        sync_bridge=_FakeBridge(),
    )

    try:
        runtime.session.sync_state.mode = SyncMode.NONE
        runtime.session.sync_state.connected = False
        assert runtime.prefers_low_latency_transport_poll() is False
        runtime.session.sync_state.mode = SyncMode.MA3
        runtime.session.sync_state.connected = True
        assert runtime.prefers_low_latency_transport_poll() is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)
