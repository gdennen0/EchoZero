from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import run_echozero
import run_timeline_demo
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


def _install_launcher_fakes(monkeypatch, *, exec_result: int = 0, exec_error: Exception | None = None):
    FakeWidget.instances = []
    FakeQTimer.shots = []
    FakeQApplication.instance_value = None
    FakeQApplication.init_args = []
    FakeQApplication.exec_result = exec_result
    FakeQApplication.exec_error = exec_error
    FakeQApplication.instance_calls = 0
    FakeQApplication.quit_calls = 0

    runtime_audio = FakeRuntimeAudio()
    demo = SimpleNamespace(
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
        runtime_audio=runtime_audio,
    )
    build_calls: list[dict[str, object]] = []

    def fake_build_app_shell(*, use_demo_fixture=False, sync_bridge=None, sync_service=None, working_dir_root=None):
        build_calls.append(
            {
                "use_demo_fixture": use_demo_fixture,
                "sync_bridge": sync_bridge,
                "sync_service": sync_service,
                "working_dir_root": working_dir_root,
            }
        )
        return demo

    monkeypatch.setattr(run_echozero, "build_app_shell", fake_build_app_shell)
    monkeypatch.setattr(run_echozero, "TimelineWidget", FakeWidget)
    monkeypatch.setattr(run_echozero, "QApplication", FakeQApplication)
    monkeypatch.setattr(run_echozero, "QTimer", FakeQTimer)
    monkeypatch.setattr(run_echozero, "QAction", FakeAction)
    return demo, runtime_audio, build_calls


def test_run_echozero_main_wires_widget_and_smoke_timer(monkeypatch):
    _, runtime_audio, build_calls = _install_launcher_fakes(monkeypatch, exec_result=27)

    result = run_echozero.main(["--smoke-exit-seconds", "1.25", "--style", "fusion"])

    widget = FakeWidget.instances[0]
    assert result == 27
    assert build_calls == [
        {
            "use_demo_fixture": False,
            "sync_bridge": None,
            "sync_service": None,
            "working_dir_root": Path(tempfile.gettempdir()) / "EchoZero" / "smoke-working",
        }
    ]
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
    _, runtime_audio, build_calls = _install_launcher_fakes(monkeypatch, exec_result=5)

    result = run_echozero.main([])

    widget = FakeWidget.instances[0]
    assert result == 5
    assert build_calls == [
        {
            "use_demo_fixture": False,
            "sync_bridge": None,
            "sync_service": None,
            "working_dir_root": None,
        }
    ]
    assert widget.window_titles == ["EchoZero"]
    assert widget.resize_calls == [(1440, 720)]
    assert widget.show_calls == 1
    assert FakeQTimer.shots == []
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_shuts_down_audio_on_exec_failure(monkeypatch):
    _, runtime_audio, build_calls = _install_launcher_fakes(monkeypatch, exec_error=RuntimeError("boom"))

    try:
        run_echozero.main([])
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert build_calls == [
        {
            "use_demo_fixture": False,
            "sync_bridge": None,
            "sync_service": None,
            "working_dir_root": None,
        }
    ]
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_passes_demo_fixture_flag(monkeypatch):
    _, runtime_audio, build_calls = _install_launcher_fakes(monkeypatch, exec_result=9)

    result = run_echozero.main(["--use-demo-fixture"])

    assert result == 9
    assert build_calls == [
        {
            "use_demo_fixture": True,
            "sync_bridge": None,
            "sync_service": None,
            "working_dir_root": None,
        }
    ]
    assert runtime_audio.shutdown_calls == 1


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

    monkeypatch.setattr(run_echozero, "QAction", FakeAction)
    monkeypatch.setattr(
        run_echozero.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("C:/projects/opened.ez", run_echozero.PROJECT_FILE_FILTER),
    )
    monkeypatch.setattr(
        run_echozero.QFileDialog,
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

    monkeypatch.setattr(run_echozero, "QAction", FakeAction)
    monkeypatch.setattr(
        run_echozero.QMessageBox,
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

    monkeypatch.setattr(run_echozero, "QAction", FakeAction)
    monkeypatch.setattr(
        run_echozero.QMessageBox,
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

    monkeypatch.setattr(run_echozero, "QAction", FakeAction)
    monkeypatch.setattr(
        run_echozero.QMessageBox,
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

    monkeypatch.setattr(run_echozero, "QAction", FakeAction)

    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    assert widget._launcher_actions["new_project"].trigger() is None
    assert widget._launcher_actions["open_project"].trigger() is None
    assert widget._launcher_actions["save_project"].trigger() is None
    assert widget._launcher_actions["save_project_as"].trigger() is None


def test_run_timeline_demo_delegates_to_run_echozero_main_with_demo_fixture_flag(monkeypatch):
    forwarded_args: list[list[str]] = []

    def fake_main(argv):
        forwarded_args.append(list(argv))
        return 17

    monkeypatch.setattr(run_timeline_demo, "_run_echozero_main", fake_main)

    result = run_timeline_demo.main(["--style", "fusion"])

    assert result == 17
    assert forwarded_args == [["--style", "fusion", "--use-demo-fixture"]]


def test_run_timeline_demo_does_not_duplicate_demo_fixture_flag(monkeypatch):
    forwarded_args: list[list[str]] = []

    def fake_main(argv):
        forwarded_args.append(list(argv))
        return 23

    monkeypatch.setattr(run_timeline_demo, "_run_echozero_main", fake_main)

    result = run_timeline_demo.main(["--use-demo-fixture", "--style", "fusion"])

    assert result == 23
    assert forwarded_args == [["--use-demo-fixture", "--style", "fusion"]]
