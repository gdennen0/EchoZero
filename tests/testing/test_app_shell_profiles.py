from __future__ import annotations

import inspect
import shutil
import uuid
from pathlib import Path

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
        def __init__(self, presentation, *, on_intent, runtime_audio) -> None:
            self.presentation = presentation
            self.on_intent = on_intent
            self.runtime_audio = runtime_audio

        def setObjectName(self, _name: str) -> None:
            pass

        def resize(self, _width: int, _height: int) -> None:
            pass

        def setWindowTitle(self, _title: str) -> None:
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
