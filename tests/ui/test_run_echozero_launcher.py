from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import echozero.ui.qt.launcher_surface as launcher_surface
import run_echozero
from PyQt6.QtWidgets import QMessageBox


class FakeRuntimeAudio:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeWidget:
    instances: list["FakeWidget"] = []

    def __init__(self, presentation, *, on_intent, runtime_audio) -> None:
        self.presentation = presentation
        self.on_intent = on_intent
        self.runtime_audio = runtime_audio
        self.resize_calls: list[tuple[int, int]] = []
        self.window_titles: list[str] = []
        self.show_calls = 0
        self.close_calls = 0
        self.presentation_updates: list[object] = []
        self.actions: list[object] = []
        self.close_events: list[object] = []
        FakeWidget.instances.append(self)

    def resize(self, width: int, height: int) -> None:
        self.resize_calls.append((width, height))

    def setWindowTitle(self, title: str) -> None:
        self.window_titles.append(title)

    def show(self) -> None:
        self.show_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def set_presentation(self, presentation) -> None:
        self.presentation = presentation
        self.presentation_updates.append(presentation)

    def addAction(self, action) -> None:
        self.actions.append(action)

    def closeEvent(self, event) -> None:
        self.close_events.append(event)
        event.accept()


class FakeSignal:
    def __init__(self) -> None:
        self._callbacks: list[object] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in list(self._callbacks):
            callback()


class FakeAction:
    def __init__(self, text: str, parent) -> None:
        self.text = text
        self.parent = parent
        self.shortcut = None
        self.triggered = FakeSignal()

    def setShortcut(self, shortcut) -> None:
        self.shortcut = shortcut

    def trigger(self) -> None:
        self.triggered.emit()


class FakeCloseEvent:
    def __init__(self) -> None:
        self.accepted = False
        self.ignored = False

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


class FakeQTimer:
    shots: list[tuple[int, object]] = []

    @classmethod
    def singleShot(cls, milliseconds: int, callback) -> None:
        cls.shots.append((milliseconds, callback))


class FakeQApplication:
    instance_value = None
    init_args: list[list[str]] = []
    exec_result = 0
    exec_error: Exception | None = None
    instance_calls = 0
    quit_calls = 0

    def __init__(self, args: list[str]) -> None:
        self.args = list(args)
        type(self).init_args.append(self.args)

    @classmethod
    def instance(cls):
        cls.instance_calls += 1
        return cls.instance_value

    def exec(self) -> int:
        if type(self).exec_error is not None:
            raise type(self).exec_error
        return type(self).exec_result

    def quit(self) -> None:
        type(self).quit_calls += 1


class FakeAutomationBridgeServer:
    instances: list["FakeAutomationBridgeServer"] = []

    def __init__(self, *, runtime, widget, launcher, app, port: int) -> None:
        self.runtime = runtime
        self.widget = widget
        self.launcher = launcher
        self.app = app
        self.port = port
        self.started = False
        self.stopped = False
        type(self).instances.append(self)

    @property
    def address(self) -> tuple[str, int]:
        return ("127.0.0.1", 43210 if self.port == 0 else self.port)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def _install_launcher_fakes(monkeypatch, *, exec_result: int = 0, exec_error: Exception | None = None):
    FakeWidget.instances = []
    FakeQTimer.shots = []
    FakeQApplication.instance_value = None
    FakeQApplication.init_args = []
    FakeQApplication.exec_result = exec_result
    FakeQApplication.exec_error = exec_error
    FakeQApplication.instance_calls = 0
    FakeQApplication.quit_calls = 0
    FakeAutomationBridgeServer.instances = []

    runtime_audio = FakeRuntimeAudio()
    runtime = SimpleNamespace(
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
        runtime_audio=runtime_audio,
        shutdown=lambda: runtime_audio.shutdown(),
    )
    build_calls: list[dict[str, object]] = []

    def fake_build_launcher_surface(*, working_dir_root=None, **_kwargs):
        build_calls.append(
            {
                "working_dir_root": working_dir_root,
            }
        )
        widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
        widget.resize(1440, 720)
        widget.setWindowTitle("EchoZero")
        launcher = SimpleNamespace(
            actions={
                "new_project": object(),
                "open_project": object(),
                "save_project": object(),
                "save_project_as": object(),
            }
        )
        widget._launcher_actions = launcher.actions
        return SimpleNamespace(runtime=runtime, widget=widget, controller=launcher)

    log_calls: list[Path | None] = []

    monkeypatch.setattr(run_echozero, "build_launcher_surface", fake_build_launcher_surface)
    monkeypatch.setattr(run_echozero, "QApplication", FakeQApplication)
    monkeypatch.setattr(run_echozero, "QTimer", FakeQTimer)
    monkeypatch.setattr(run_echozero, "AutomationBridgeServer", FakeAutomationBridgeServer)
    monkeypatch.setattr(
        run_echozero,
        "install_runtime_logging",
        lambda log_dir=None: log_calls.append(log_dir) or Path("C:/logs/session.log"),
    )
    return runtime, runtime_audio, build_calls, log_calls


def test_run_echozero_main_wires_widget_and_smoke_timer(monkeypatch):
    _, runtime_audio, build_calls, log_calls = _install_launcher_fakes(monkeypatch, exec_result=27)

    result = run_echozero.main(["--smoke-exit-seconds", "1.25", "--style", "fusion"])

    widget = FakeWidget.instances[0]
    assert result == 27
    assert build_calls == [
        {
            "working_dir_root": Path(tempfile.gettempdir()) / "EchoZero" / "smoke-working",
        }
    ]
    assert log_calls == [None]
    assert widget.window_titles == ["EchoZero"]
    assert widget.resize_calls == [(1440, 720)]
    assert widget.show_calls == 1
    assert sorted(widget._launcher_actions.keys()) == [
        "new_project",
        "open_project",
        "save_project",
        "save_project_as",
    ]
    assert len(FakeQTimer.shots) == 1
    milliseconds, callback = FakeQTimer.shots[0]
    assert milliseconds == 1250
    callback()
    assert widget.close_calls == 0
    assert FakeQApplication.quit_calls == 1
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_skips_timer_when_not_requested(monkeypatch):
    _, runtime_audio, build_calls, log_calls = _install_launcher_fakes(monkeypatch, exec_result=5)

    result = run_echozero.main([])

    widget = FakeWidget.instances[0]
    assert result == 5
    assert build_calls == [
        {
            "working_dir_root": None,
        }
    ]
    assert log_calls == [None]
    assert widget.window_titles == ["EchoZero"]
    assert widget.resize_calls == [(1440, 720)]
    assert widget.show_calls == 1
    assert FakeQTimer.shots == []
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_shuts_down_audio_on_exec_failure(monkeypatch):
    _, runtime_audio, build_calls, log_calls = _install_launcher_fakes(monkeypatch, exec_error=RuntimeError("boom"))

    try:
        run_echozero.main([])
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert build_calls == [
        {
            "working_dir_root": None,
        }
    ]
    assert log_calls == [None]
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_starts_and_stops_automation_bridge(monkeypatch, capsys):
    _, runtime_audio, build_calls, log_calls = _install_launcher_fakes(monkeypatch, exec_result=11)

    result = run_echozero.main(["--automation-port", "0"])

    assert result == 11
    assert build_calls == [{"working_dir_root": None}]
    assert log_calls == [None]
    assert runtime_audio.shutdown_calls == 1
    assert len(FakeAutomationBridgeServer.instances) == 1
    bridge = FakeAutomationBridgeServer.instances[0]
    assert bridge.port == 0
    assert bridge.started is True
    assert bridge.stopped is True
    assert "automation_bridge=http://127.0.0.1:43210" in capsys.readouterr().out


def test_run_echozero_main_writes_automation_info_file(monkeypatch, tmp_path):
    _, _, _, _ = _install_launcher_fakes(monkeypatch, exec_result=0)
    info_path = tmp_path / "automation" / "bridge.txt"

    result = run_echozero.main(["--automation-port", "0", "--automation-info-file", str(info_path)])

    assert result == 0
    assert info_path.read_text(encoding="utf-8").strip() == "http://127.0.0.1:43210"


def test_launcher_actions_invoke_canonical_runtime_methods(monkeypatch):
    runtime_audio = FakeRuntimeAudio()
    calls: list[tuple[str, object | None]] = []
    runtime = SimpleNamespace(
        runtime_audio=runtime_audio,
        project_path=Path("C:/projects/current.ez"),
        is_dirty=False,
    )
    current_presentation = {"value": "initial"}

    def presentation():
        return current_presentation["value"]

    def new_project():
        calls.append(("new_project", None))
        current_presentation["value"] = "after-new"

    def open_project(path):
        calls.append(("open_project", Path(path)))
        runtime.project_path = Path(path)
        current_presentation["value"] = "after-open"

    def save_project():
        calls.append(("save_project", None))

    def save_project_as(path):
        calls.append(("save_project_as", Path(path)))
        runtime.project_path = Path(path)

    runtime.presentation = presentation
    runtime.dispatch = lambda intent: intent
    runtime.new_project = new_project
    runtime.open_project = open_project
    runtime.save_project = save_project
    runtime.save_project_as = save_project_as

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("C:/projects/opened.ez", run_echozero.PROJECT_FILE_FILTER),
    )
    monkeypatch.setattr(
        launcher_surface.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("C:/projects/saved-as.ez", run_echozero.PROJECT_FILE_FILTER),
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["new_project"].trigger()
    widget._launcher_actions["open_project"].trigger()
    widget._launcher_actions["save_project"].trigger()
    runtime.project_path = None
    widget._launcher_actions["save_project"].trigger()
    widget._launcher_actions["save_project_as"].trigger()

    assert calls == [
        ("new_project", None),
        ("open_project", Path("C:/projects/opened.ez")),
        ("save_project", None),
        ("save_project_as", Path("C:/projects/saved-as.ez")),
        ("save_project_as", Path("C:/projects/saved-as.ez")),
    ]
    assert widget.presentation_updates[:2] == ["after-new", "after-open"]


def test_launcher_close_prompt_cancel_prevents_close(monkeypatch):
    runtime = SimpleNamespace(runtime_audio=FakeRuntimeAudio(), is_dirty=True, project_path=Path("C:/projects/current.ez"))
    runtime.presentation = lambda: "presentation"
    runtime.dispatch = lambda intent: intent
    runtime.save_project = lambda: None

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()
    event = FakeCloseEvent()

    widget.closeEvent(event)

    assert event.ignored is True
    assert event.accepted is False
    assert widget.close_events == []


def test_launcher_close_prompt_save_uses_current_path(monkeypatch):
    calls: list[str] = []
    runtime = SimpleNamespace(runtime_audio=FakeRuntimeAudio(), is_dirty=True, project_path=Path("C:/projects/current.ez"))
    runtime.presentation = lambda: "presentation"
    runtime.dispatch = lambda intent: intent
    runtime.save_project = lambda: calls.append("save_project")

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Save,
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()
    event = FakeCloseEvent()

    widget.closeEvent(event)

    assert calls == ["save_project"]
    assert event.accepted is True
    assert event.ignored is False
    assert len(widget.close_events) == 1


def test_launcher_close_prompt_discard_closes(monkeypatch):
    runtime = SimpleNamespace(runtime_audio=FakeRuntimeAudio(), is_dirty=True, project_path=Path("C:/projects/current.ez"))
    runtime.presentation = lambda: "presentation"
    runtime.dispatch = lambda intent: intent

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()
    event = FakeCloseEvent()

    widget.closeEvent(event)

    assert event.accepted is True
    assert event.ignored is False
    assert len(widget.close_events) == 1


def test_launcher_demo_mode_actions_noop_safely(monkeypatch):
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        is_dirty=False,
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )
    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)

    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    assert widget._launcher_actions["new_project"].trigger() is None
    assert widget._launcher_actions["open_project"].trigger() is None
    assert widget._launcher_actions["save_project"].trigger() is None
    assert widget._launcher_actions["save_project_as"].trigger() is None
