from __future__ import annotations

from types import SimpleNamespace

import run_echozero
import run_timeline_demo


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
        FakeWidget.instances.append(self)

    def resize(self, width: int, height: int) -> None:
        self.resize_calls.append((width, height))

    def setWindowTitle(self, title: str) -> None:
        self.window_titles.append(title)

    def show(self) -> None:
        self.show_calls += 1

    def close(self) -> None:
        self.close_calls += 1


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

    def fake_build_app_shell(*, use_demo_fixture=False, sync_bridge=None, sync_service=None):
        build_calls.append(
            {
                "use_demo_fixture": use_demo_fixture,
                "sync_bridge": sync_bridge,
                "sync_service": sync_service,
            }
        )
        return demo

    monkeypatch.setattr(run_echozero, "build_app_shell", fake_build_app_shell)
    monkeypatch.setattr(run_echozero, "TimelineWidget", FakeWidget)
    monkeypatch.setattr(run_echozero, "QApplication", FakeQApplication)
    monkeypatch.setattr(run_echozero, "QTimer", FakeQTimer)
    return demo, runtime_audio, build_calls


def test_run_echozero_main_wires_widget_and_smoke_timer(monkeypatch):
    _, runtime_audio, build_calls = _install_launcher_fakes(monkeypatch, exec_result=27)

    result = run_echozero.main(["--smoke-exit-seconds", "1.25", "--style", "fusion"])

    widget = FakeWidget.instances[0]
    assert result == 27
    assert build_calls == [{"use_demo_fixture": False, "sync_bridge": None, "sync_service": None}]
    assert widget.window_titles == ["EchoZero"]
    assert widget.resize_calls == [(1440, 720)]
    assert widget.show_calls == 1
    assert len(FakeQTimer.shots) == 1
    milliseconds, callback = FakeQTimer.shots[0]
    assert milliseconds == 1250
    callback()
    assert widget.close_calls == 1
    assert FakeQApplication.quit_calls == 1
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_skips_timer_when_not_requested(monkeypatch):
    _, runtime_audio, build_calls = _install_launcher_fakes(monkeypatch, exec_result=5)

    result = run_echozero.main([])

    widget = FakeWidget.instances[0]
    assert result == 5
    assert build_calls == [{"use_demo_fixture": False, "sync_bridge": None, "sync_service": None}]
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

    assert build_calls == [{"use_demo_fixture": False, "sync_bridge": None, "sync_service": None}]
    assert runtime_audio.shutdown_calls == 1


def test_run_echozero_main_passes_demo_fixture_flag(monkeypatch):
    _, runtime_audio, build_calls = _install_launcher_fakes(monkeypatch, exec_result=9)

    result = run_echozero.main(["--use-demo-fixture"])

    assert result == 9
    assert build_calls == [{"use_demo_fixture": True, "sync_bridge": None, "sync_service": None}]
    assert runtime_audio.shutdown_calls == 1


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
