from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import echozero.ui.qt.launcher_surface as launcher_surface
import run_echozero
from echozero.application.settings import AppSettingsLaunchOverrides
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
        self.external_presentation_updates: list[object] = []
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

    def apply_external_presentation_update(self, presentation) -> None:
        self.presentation = presentation
        self.external_presentation_updates.append(presentation)
        self.presentation_updates.append(presentation)

    def addAction(self, action) -> None:
        self.actions.append(action)

    def set_runtime_audio_controller(self, runtime_audio) -> None:
        self.runtime_audio = runtime_audio

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
        self._props: dict[str, object] = {}
        self._stylesheet = ""
        self._palette = None
        type(self).init_args.append(self.args)

    @classmethod
    def instance(cls):
        cls.instance_calls += 1
        return cls.instance_value

    def property(self, name: str) -> object | None:
        return self._props.get(name)

    def setProperty(self, name: str, value: object) -> None:
        self._props[name] = value

    def setPalette(self, palette) -> None:
        self._palette = palette

    def setStyleSheet(self, stylesheet: str) -> None:
        self._stylesheet = stylesheet

    def styleSheet(self) -> str:
        return self._stylesheet

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


class FakeAppSettingsService:
    def __init__(self) -> None:
        self.audio_output_config = SimpleNamespace(sample_rate=48000)
        self.ma3_config = SimpleNamespace(
            is_enabled=False,
            receive=SimpleNamespace(
                enabled=False,
                host="127.0.0.1",
                port=0,
                path="/ez/message",
            ),
            send=SimpleNamespace(
                enabled=False,
                host="127.0.0.1",
                port=None,
                path="/cmd",
            ),
        )
        self.launch_overrides: list[AppSettingsLaunchOverrides] = []

    def resolve_audio_output_config(self):
        return self.audio_output_config

    def resolve_ma3_osc_runtime_config(self, *, launch_overrides):
        self.launch_overrides.append(launch_overrides)
        return self.ma3_config


class FakeOscUdpSendTransport:
    instances: list["FakeOscUdpSendTransport"] = []

    def __init__(self, host: str, port: int, *, path: str = "/") -> None:
        self.host = host
        self.port = port
        self.path = path
        type(self).instances.append(self)


class FakeMA3OSCBridge:
    instances: list["FakeMA3OSCBridge"] = []

    def __init__(
        self,
        *,
        listen_host,
        listen_port,
        listen_path="/ez/message",
        command_transport=None,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.listen_path = listen_path
        self.command_transport = command_transport
        type(self).instances.append(self)

    def shutdown(self) -> None:
        return None


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
    FakeOscUdpSendTransport.instances = []
    FakeMA3OSCBridge.instances = []

    runtime_audio = FakeRuntimeAudio()
    app_settings_service = FakeAppSettingsService()
    runtime = SimpleNamespace(
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
        runtime_audio=runtime_audio,
        shutdown=lambda: runtime_audio.shutdown(),
    )
    build_calls: list[dict[str, object]] = []
    surface_calls: list[dict[str, object]] = []

    def fake_build_launcher_surface(*, working_dir_root=None, **_kwargs):
        build_calls.append(
            {
                "working_dir_root": working_dir_root,
            }
        )
        surface_calls.append(
            {
                "working_dir_root": working_dir_root,
                **_kwargs,
            }
        )
        widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
        widget.resize(1440, 720)
        widget.setWindowTitle("EchoZero")
        launcher = SimpleNamespace(
            actions={
                "undo": object(),
                "redo": object(),
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
    monkeypatch.setattr(run_echozero, "build_default_app_settings_service", lambda: app_settings_service)
    monkeypatch.setattr(run_echozero, "OscUdpSendTransport", FakeOscUdpSendTransport)
    monkeypatch.setattr(run_echozero, "MA3OSCBridge", FakeMA3OSCBridge)
    monkeypatch.setattr(
        run_echozero,
        "install_runtime_logging",
        lambda log_dir=None: log_calls.append(log_dir) or Path("C:/logs/session.log"),
    )
    return runtime, runtime_audio, build_calls, log_calls, app_settings_service, surface_calls


def test_run_echozero_main_wires_widget_and_smoke_timer(monkeypatch):
    _, runtime_audio, build_calls, log_calls, _, _ = _install_launcher_fakes(monkeypatch, exec_result=27)

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
        "redo",
        "save_project",
        "save_project_as",
        "undo",
    ]
    assert len(FakeQTimer.shots) == 1
    milliseconds, callback = FakeQTimer.shots[0]
    assert milliseconds == 1250
    callback()
    assert widget.close_calls == 0
    assert FakeQApplication.quit_calls == 1
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_skips_timer_when_not_requested(monkeypatch):
    _, runtime_audio, build_calls, log_calls, _, _ = _install_launcher_fakes(monkeypatch, exec_result=5)

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
    _, runtime_audio, build_calls, log_calls, _, _ = _install_launcher_fakes(monkeypatch, exec_error=RuntimeError("boom"))

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
    _, runtime_audio, build_calls, log_calls, _, _ = _install_launcher_fakes(monkeypatch, exec_result=11)

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
    _, _, _, _, _, _ = _install_launcher_fakes(monkeypatch, exec_result=0)
    info_path = tmp_path / "automation" / "bridge.txt"

    result = run_echozero.main(["--automation-port", "0", "--automation-info-file", str(info_path)])

    assert result == 0
    assert info_path.read_text(encoding="utf-8").strip() == "http://127.0.0.1:43210"


def test_run_echozero_main_uses_app_settings_service_for_launcher_and_ma3(monkeypatch):
    _, _, _, _, app_settings_service, surface_calls = _install_launcher_fakes(
        monkeypatch,
        exec_result=0,
    )
    app_settings_service.ma3_config = SimpleNamespace(
        is_enabled=True,
        receive=SimpleNamespace(
            enabled=True,
            host="127.0.0.1",
            port=7100,
            path="/ez/message",
        ),
        send=SimpleNamespace(
            enabled=True,
            host="10.0.0.2",
            port=9000,
            path="/cmd",
        ),
    )

    result = run_echozero.main(
        [
            "--ma3-osc-listen-port",
            "7100",
            "--ma3-osc-command-host",
            "10.0.0.2",
            "--ma3-osc-command-port",
            "9000",
        ]
    )

    assert result == 0
    assert app_settings_service.launch_overrides == [
        AppSettingsLaunchOverrides(
            ma3_osc_listen_host=None,
            ma3_osc_listen_port=7100,
            ma3_osc_command_host="10.0.0.2",
            ma3_osc_command_port=9000,
        )
    ]
    assert surface_calls[0]["app_settings_service"] is app_settings_service
    assert surface_calls[0]["audio_output_config"] is app_settings_service.audio_output_config
    assert len(FakeOscUdpSendTransport.instances) == 1
    assert FakeOscUdpSendTransport.instances[0].host == "10.0.0.2"
    assert FakeOscUdpSendTransport.instances[0].port == 9000
    assert FakeOscUdpSendTransport.instances[0].path == "/cmd"
    assert len(FakeMA3OSCBridge.instances) == 1
    assert FakeMA3OSCBridge.instances[0].listen_port == 7100
    assert FakeMA3OSCBridge.instances[0].listen_path == "/ez/message"


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

    def undo():
        calls.append(("undo", None))

    def redo():
        calls.append(("redo", None))

    runtime.presentation = presentation
    runtime.dispatch = lambda intent: intent
    runtime.new_project = new_project
    runtime.open_project = open_project
    runtime.save_project = save_project
    runtime.save_project_as = save_project_as
    runtime.undo = undo
    runtime.redo = redo

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

    widget._launcher_actions["undo"].trigger()
    widget._launcher_actions["redo"].trigger()
    widget._launcher_actions["new_project"].trigger()
    widget._launcher_actions["open_project"].trigger()
    widget._launcher_actions["save_project"].trigger()
    runtime.project_path = None
    widget._launcher_actions["save_project"].trigger()
    widget._launcher_actions["save_project_as"].trigger()

    assert calls == [
        ("undo", None),
        ("redo", None),
        ("new_project", None),
        ("open_project", Path("C:/projects/opened.ez")),
        ("save_project", None),
        ("save_project_as", Path("C:/projects/saved-as.ez")),
        ("save_project_as", Path("C:/projects/saved-as.ez")),
    ]
    assert widget.external_presentation_updates[2:4] == ["after-new", "after-open"]
    assert widget.presentation_updates[2:4] == ["after-new", "after-open"]


def test_launcher_exposes_enable_phone_review_service_when_runtime_supports_it(monkeypatch):
    info_messages: list[tuple[str, str]] = []
    phone_review_events: list[str] = []
    open_review_calls: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        is_dirty=False,
        project_path=Path("C:/projects/current.ez"),
        session=SimpleNamespace(active_song_version_id="ver_active"),
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )
    runtime.enable_phone_review_service = lambda: phone_review_events.append("enable") or SimpleNamespace(
        url="http://127.0.0.1:8421/",
        desktop_url="http://127.0.0.1:8421/",
        phone_url="http://192.168.1.44:8421/",
    )
    runtime.open_project_review_session = lambda **kwargs: open_review_calls.append(kwargs) or SimpleNamespace(
        url="http://127.0.0.1:8421/",
        desktop_url="http://127.0.0.1:8421/",
        phone_url="http://192.168.1.44:8421/",
    )

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        "echozero.ui.qt.launcher_review_actions.QMessageBox.information",
        lambda _parent, title, message: info_messages.append((title, message)),
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    assert "enable_phone_review_service" in widget._launcher_actions
    widget._launcher_actions["enable_phone_review_service"].trigger()
    assert info_messages == [
        (
            "Phone Review Service Enabled",
            "Live link enabled for the current EZ session.\n\n"
            "Phone URL:\nhttp://192.168.1.44:8421/",
        ),
    ]
    assert phone_review_events == ["enable"]
    assert open_review_calls == [
        {
            "song_version_id": "ver_active",
            "review_mode": "all_events",
            "item_limit": None,
        }
    ]


def test_launcher_new_project_prompts_to_save_dirty_changes(monkeypatch):
    calls: list[str] = []
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        is_dirty=True,
        project_path=Path("C:/projects/current.ez"),
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )
    runtime.save_project = lambda: calls.append("save_project")
    runtime.new_project = lambda: calls.append("new_project")

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Save,
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["new_project"].trigger()

    assert calls == ["save_project", "new_project"]


def test_launcher_open_project_cancel_keeps_current_project(monkeypatch):
    calls: list[tuple[str, object]] = []
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        is_dirty=True,
        project_path=Path("C:/projects/current.ez"),
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )
    runtime.open_project = lambda path: calls.append(("open_project", Path(path)))

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("C:/projects/opened.ez", run_echozero.PROJECT_FILE_FILTER),
    )
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["open_project"].trigger()

    assert calls == []


def test_launcher_save_project_as_appends_ez_extension(monkeypatch):
    calls: list[Path] = []
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        is_dirty=False,
        project_path=None,
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )
    runtime.save_project_as = lambda path: calls.append(Path(path))

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("C:/projects/saved-project", run_echozero.PROJECT_FILE_FILTER),
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["save_project_as"].trigger()

    assert calls == [Path("C:/projects/saved-project.ez")]


def test_launcher_open_project_reports_errors(monkeypatch):
    errors: list[tuple[str, str]] = []
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        is_dirty=False,
        project_path=Path("C:/projects/current.ez"),
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )

    def open_project(_path) -> None:
        raise RuntimeError("bad archive")

    runtime.open_project = open_project

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("C:/projects/opened.ez", run_echozero.PROJECT_FILE_FILTER),
    )
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["open_project"].trigger()

    assert errors == [("Open Project Failed", "Open Project failed.\n\nbad archive")]


def test_launcher_open_project_tracks_recent_projects(monkeypatch):
    class _RecentSettings:
        def __init__(self) -> None:
            self._recent: list[Path] = [Path("C:/projects/older.ez")]
            self.remember_calls: list[Path] = []
            self.forget_calls: list[Path] = []

        def recent_project_paths(self):
            return tuple(self._recent)

        def remember_recent_project_path(self, path, *, limit=10):
            candidate = Path(path)
            self.remember_calls.append(candidate)
            self._recent = [entry for entry in self._recent if entry != candidate]
            self._recent.insert(0, candidate)
            self._recent = self._recent[:limit]
            return tuple(self._recent)

        def forget_recent_project_path(self, path):
            candidate = Path(path)
            self.forget_calls.append(candidate)
            self._recent = [entry for entry in self._recent if entry != candidate]
            return tuple(self._recent)

    calls: list[Path] = []
    settings = _RecentSettings()
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        app_settings_service=settings,
        is_dirty=False,
        project_path=Path("C:/projects/current.ez"),
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )

    def open_project(path) -> None:
        candidate = Path(path)
        calls.append(candidate)
        runtime.project_path = candidate

    runtime.open_project = open_project

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("C:/projects/opened.ez", run_echozero.PROJECT_FILE_FILTER),
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget, app_settings_service=settings)
    launcher.install()

    widget._launcher_actions["open_project"].trigger()

    assert calls == [Path("C:/projects/opened.ez")]
    assert settings.remember_calls[-1] == Path("C:/projects/opened.ez")
    assert not settings.forget_calls
    assert "open_recent_project::0" in launcher._recent_menu_actions


def test_launcher_open_recent_project_path_forgets_failed_entry(monkeypatch):
    class _RecentSettings:
        def __init__(self) -> None:
            self._recent: list[Path] = [Path("C:/projects/broken.ez")]
            self.remember_calls: list[Path] = []
            self.forget_calls: list[Path] = []

        def recent_project_paths(self):
            return tuple(self._recent)

        def remember_recent_project_path(self, path, *, limit=10):
            candidate = Path(path)
            self.remember_calls.append(candidate)
            self._recent = [entry for entry in self._recent if entry != candidate]
            self._recent.insert(0, candidate)
            self._recent = self._recent[:limit]
            return tuple(self._recent)

        def forget_recent_project_path(self, path):
            candidate = Path(path)
            self.forget_calls.append(candidate)
            self._recent = [entry for entry in self._recent if entry != candidate]
            return tuple(self._recent)

    errors: list[tuple[str, str]] = []
    settings = _RecentSettings()
    runtime = SimpleNamespace(
        runtime_audio=FakeRuntimeAudio(),
        app_settings_service=settings,
        is_dirty=False,
        project_path=Path("C:/projects/current.ez"),
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )

    def open_project(_path) -> None:
        raise RuntimeError("missing project")

    runtime.open_project = open_project

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget, app_settings_service=settings)
    launcher.install()

    result = launcher.open_recent_project_path(Path("C:/projects/broken.ez"))

    assert result is False
    assert settings.remember_calls == []
    assert settings.forget_calls == [Path("C:/projects/broken.ez")]
    assert errors == [("Open Project Failed", "Open Project failed.\n\nmissing project")]


def test_launcher_preferences_action_opens_dialog(monkeypatch):
    runtime_audio = FakeRuntimeAudio()
    dialog_calls: list[tuple[object, object]] = []
    runtime = SimpleNamespace(
        runtime_audio=runtime_audio,
        app_settings_service=object(),
        is_dirty=False,
        project_path=None,
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )

    class FakePreferencesDialog:
        def __init__(self, service, *, parent=None) -> None:
            self.service = service
            self.parent = parent
            dialog_calls.append((service, parent))

        def exec(self) -> int:
            return 1

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(launcher_surface, "PreferencesDialog", FakePreferencesDialog)

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["preferences"].trigger()

    assert dialog_calls == [(runtime.app_settings_service, widget)]


def test_launcher_osc_settings_action_opens_dialog(monkeypatch):
    runtime_audio = FakeRuntimeAudio()
    dialog_calls: list[tuple[object, object]] = []
    runtime = SimpleNamespace(
        runtime_audio=runtime_audio,
        app_settings_service=object(),
        is_dirty=False,
        project_path=None,
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
    )

    class FakeOscSettingsDialog:
        def __init__(self, service, *, parent=None) -> None:
            self.service = service
            self.parent = parent
            dialog_calls.append((service, parent))

        def exec(self) -> int:
            return 1

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(launcher_surface, "OscSettingsDialog", FakeOscSettingsDialog)

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["osc_settings"].trigger()

    assert dialog_calls == [(runtime.app_settings_service, widget)]


def test_launcher_project_settings_action_updates_ma3_push_offset(monkeypatch):
    runtime_audio = FakeRuntimeAudio()
    calls: list[float] = []
    runtime = SimpleNamespace(
        runtime_audio=runtime_audio,
        app_settings_service=object(),
        is_dirty=False,
        project_path=None,
        presentation=lambda: "presentation",
        dispatch=lambda intent: intent,
        get_project_ma3_push_offset_seconds=lambda: -1.0,
        set_project_ma3_push_offset_seconds=lambda value: calls.append(float(value)),
    )

    class FakeProjectSettingsDialog:
        def __init__(self, *, current_offset_seconds: float, parent=None) -> None:
            self.current_offset_seconds = current_offset_seconds
            self.parent = parent
            self.ma3_push_offset_seconds = -0.25

        def exec(self) -> int:
            return 1

    monkeypatch.setattr(launcher_surface, "QAction", FakeAction)
    monkeypatch.setattr(
        launcher_surface,
        "ProjectSettingsDialog",
        FakeProjectSettingsDialog,
    )
    monkeypatch.setattr(
        launcher_surface.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warning dialog not expected")
        ),
    )

    widget = FakeWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    launcher = run_echozero.LauncherController(runtime=runtime, widget=widget)
    launcher.install()

    widget._launcher_actions["project_settings"].trigger()

    assert calls == [-0.25]


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


def test_launcher_actions_noop_safely_without_project_controls(monkeypatch):
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

    assert widget._launcher_actions["undo"].trigger() is None
    assert widget._launcher_actions["redo"].trigger() is None
    assert widget._launcher_actions["new_project"].trigger() is None
    assert widget._launcher_actions["open_project"].trigger() is None
    assert widget._launcher_actions["save_project"].trigger() is None
    assert widget._launcher_actions["save_project_as"].trigger() is None
