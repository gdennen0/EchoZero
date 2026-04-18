from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from echozero.application.presentation.models import EventPresentation, LayerPresentation, TakeLanePresentation, TimelinePresentation
from echozero.application.shared.enums import LayerKind, PlaybackMode
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.intents import Pause, Play, Seek, SelectLayer, SetActivePlaybackTarget, Stop
from echozero.audio.engine import AudioEngine
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.runtime_audio import RuntimeAudioTimingSnapshot, TimelineRuntimeAudioController
from echozero.ui.qt.timeline.blocks.ruler import seek_time_for_x
from echozero.ui.qt.timeline.widget import TimelineWidget


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.blocksize = kwargs.get("blocksize")
        self.latency = kwargs.get("latency", 0.0)
        self.prime_output_buffers_using_stream_callback = kwargs.get(
            "prime_output_buffers_using_stream_callback"
        )
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def _fake_stream_factory(**kwargs):
    return FakeStream(**kwargs)


class FakeRuntimeAudio:
    def __init__(self):
        self.playing = False
        self.current_time = 0.0
        self.mix_states: list[TimelinePresentation] = []
        self.snapshot: RuntimeAudioTimingSnapshot | None = None

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        return None

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self.mix_states.append(presentation)

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def stop(self) -> None:
        self.playing = False
        self.current_time = 0.0

    def seek(self, position_seconds: float) -> None:
        self.current_time = position_seconds

    def current_time_seconds(self) -> float:
        return self.current_time

    def timing_snapshot(self) -> RuntimeAudioTimingSnapshot:
        if self.snapshot is not None:
            return self.snapshot
        return RuntimeAudioTimingSnapshot(
            audible_time_seconds=self.current_time,
            clock_time_seconds=self.current_time,
            snapshot_monotonic_seconds=None,
            is_playing=self.playing,
        )

    def is_playing(self) -> bool:
        return self.playing

    def presentation_signature(self, presentation: TimelinePresentation):
        return tuple(
            (
                str(layer.layer_id),
                layer.source_audio_path or layer.playback_source_ref or "",
            )
            for layer in presentation.layers
            if layer.source_audio_path or layer.playback_source_ref
        )


class RecordingRuntimeAudio:
    def __init__(self):
        self.calls: list[tuple[str, float | None]] = []
        self.playing = False
        self.current_time = 0.0

    def play(self) -> None:
        self.playing = True
        self.calls.append(("play", None))

    def pause(self) -> None:
        self.playing = False
        self.calls.append(("pause", None))

    def stop(self) -> None:
        self.playing = False
        self.current_time = 0.0
        self.calls.append(("stop", None))

    def seek(self, position_seconds: float) -> None:
        self.current_time = float(position_seconds)
        self.calls.append(("seek", position_seconds))

    def current_time_seconds(self) -> float:
        return self.current_time

    def is_playing(self) -> bool:
        return self.playing

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        return None

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self.calls.append(("mix", None))

    def presentation_signature(self, presentation: TimelinePresentation):
        return tuple(
            (
                str(layer.layer_id),
                layer.source_audio_path or layer.playback_source_ref or "",
            )
            for layer in presentation.layers
            if layer.source_audio_path or layer.playback_source_ref
        )


class CountingRuntimeAudio(FakeRuntimeAudio):
    def __init__(self):
        super().__init__()
        self.build_calls = 0
        self.mix_calls = 0

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        self.build_calls += 1

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        self.mix_calls += 1


def _audio_presentation() -> TimelinePresentation:
    base = build_demo_app().presentation()
    audio_layer = LayerPresentation(
        layer_id=LayerId("runtime_audio"),
        title="Runtime Audio",
        kind=LayerKind.AUDIO,
        source_audio_path="demo.wav",
    )
    return replace(
        base,
        layers=[audio_layer],
        active_playback_layer_id=audio_layer.layer_id,
        playhead=0.0,
        is_playing=False,
        current_time_label="00:00.00",
    )


def _event_slice_presentation() -> TimelinePresentation:
    base = build_demo_app().presentation()
    audio_layer = LayerPresentation(
        layer_id=LayerId("bed"),
        title="Bed",
        kind=LayerKind.AUDIO,
        source_audio_path="bed.wav",
    )
    kick_layer = LayerPresentation(
        layer_id=LayerId("kick_lane"),
        title="Kick",
        kind=LayerKind.EVENT,
        playback_enabled=True,
        playback_mode=PlaybackMode.EVENT_SLICE,
        playback_source_ref="kick.wav",
        events=[
            EventPresentation(
                event_id=EventId("kick_1"),
                start=0.5,
                end=0.6,
                label="Kick",
            ),
            EventPresentation(
                event_id=EventId("kick_2"),
                start=1.0,
                end=1.1,
                label="Kick",
            ),
        ],
    )
    return replace(
        base,
        layers=[audio_layer, kick_layer],
        playhead=0.0,
        is_playing=False,
        current_time_label="00:00.00",
    )


def test_runtime_controller_updates_mix_state_while_playing():
    presentation = _audio_presentation()
    engine = AudioEngine(stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda path: (np.ones(4410, dtype=np.float32), 44100),
    )
    controller.build_for_presentation(presentation)
    controller.play()

    updated = replace(
        presentation,
        layers=[replace(presentation.layers[0], gain_db=-6.0)],
    )
    controller.apply_mix_state(updated)

    engine_layer = engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)
    assert engine.transport.is_playing is True
    assert engine_layer is not None
    assert round(engine_layer.volume, 3) == round(10 ** (-6.0 / 20.0), 3)
    controller.shutdown()


def test_runtime_controller_compensates_for_reported_output_latency_while_playing():
    presentation = _audio_presentation()
    engine = AudioEngine(
        stream_factory=lambda **kwargs: FakeStream(**kwargs | {"latency": 0.1})
    )
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda path: (np.ones(4410, dtype=np.float32), 44100),
    )
    controller.build_for_presentation(presentation)
    controller.play()

    engine.seek_seconds(1.0)

    assert controller.current_time_seconds() == pytest.approx(0.9)
    controller.shutdown()


def test_runtime_controller_exposes_backend_timing_snapshot(monkeypatch):
    presentation = _audio_presentation()
    engine = AudioEngine(stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda path: (np.ones(4410, dtype=np.float32), 44100),
    )
    controller.build_for_presentation(presentation)
    controller.play()

    monotonic_now = {"value": 100.0}
    monkeypatch.setattr("echozero.audio.engine.time.monotonic", lambda: monotonic_now["value"])

    outdata = np.zeros((256, 1), dtype=np.float32)
    engine._audio_callback(
        outdata,
        256,
        {"currentTime": 5.0, "outputBufferDacTime": 5.1},
        None,
    )

    snapshot = controller.timing_snapshot()

    assert snapshot.is_playing is True
    assert snapshot.audible_time_seconds == pytest.approx(engine.audible_time_seconds)
    assert snapshot.clock_time_seconds == pytest.approx(engine.clock.position_seconds)
    assert snapshot.snapshot_monotonic_seconds == pytest.approx(100.0)
    controller.shutdown()


def test_runtime_controller_snapshot_state_reports_backend_session_and_target():
    presentation = _audio_presentation()
    engine = AudioEngine(stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda path: (np.ones(4410, dtype=np.float32), 44100),
    )

    state = controller.snapshot_state(presentation)

    assert state.backend_name == "sounddevice"
    assert state.output_sample_rate == 44100
    assert state.output_channels == 1
    assert state.active_layer_id == presentation.active_playback_layer_id
    assert state.active_take_id is None
    assert len(state.active_sources) == 1
    assert state.active_sources[0].source_ref == "demo.wav"
    controller.shutdown()


def test_runtime_controller_selected_event_lane_becomes_only_active_playback_source():
    presentation = replace(
        _event_slice_presentation(),
        active_playback_layer_id=LayerId("kick_lane"),
    )
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "bed.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([1.0, 0.5], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(int(0.5 * 44100), 2)

    engine_layer = engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)
    assert engine_layer is not None
    np.testing.assert_array_almost_equal(mixed, np.array([1.0, 0.5], dtype=np.float32))
    controller.shutdown()


def test_runtime_controller_requires_explicit_playback_target_when_selection_is_missing():
    presentation = _event_slice_presentation()
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "bed.wav":
            return np.array([0.25, 0.1], dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([0.75, -0.25], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    engine_layer = engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)
    assert engine_layer is None
    controller.shutdown()


def test_runtime_controller_selected_layer_switches_active_source_without_stopping_transport():
    base = replace(
        _event_slice_presentation(),
        active_playback_layer_id=LayerId("bed"),
    )
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "bed.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([1.0, 0.5], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(base)
    controller.play()
    controller.apply_mix_state(
        replace(
            base,
            active_playback_layer_id=LayerId("kick_lane"),
        )
    )

    assert controller.is_playing() is True
    assert engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID) is not None
    mixed = engine.mixer.read_mix(int(0.5 * 44100), 2)
    np.testing.assert_array_almost_equal(mixed, np.array([1.0, 0.5], dtype=np.float32))
    controller.shutdown()


def test_runtime_controller_uses_selected_take_audio_for_monitored_layer():
    base = build_demo_app().presentation()
    alt_take = TakeLanePresentation(
        take_id=TakeId("take_alt"),
        name="Alt",
        kind=LayerKind.AUDIO,
        source_audio_path="alt.wav",
        playback_source_ref="alt.wav",
    )
    monitored_layer = LayerPresentation(
        layer_id=LayerId("stems"),
        title="Stems",
        kind=LayerKind.AUDIO,
        source_audio_path="main.wav",
        playback_source_ref="main.wav",
        takes=[alt_take],
    )
    presentation = replace(
        base,
        layers=[monitored_layer],
        active_playback_layer_id=LayerId("stems"),
        active_playback_take_id=alt_take.take_id,
    )
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "main.wav":
            return np.array([0.1, 0.2], dtype=np.float32), 44100
        if path == "alt.wav":
            return np.array([0.8, -0.4], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    engine_layer = engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)
    assert engine_layer is not None
    np.testing.assert_array_almost_equal(engine_layer.buffer[:2], np.array([0.8, -0.4], dtype=np.float32))
    controller.shutdown()


def test_demo_dispatch_routes_transport_intents_into_runtime_audio():
    demo = build_demo_app()
    runtime_audio = RecordingRuntimeAudio()
    demo.runtime_audio = runtime_audio

    demo.dispatch(Play())
    demo.dispatch(Seek(4.25))
    demo.dispatch(Pause())
    demo.dispatch(Stop())

    assert runtime_audio.calls[:4] == [
        ("play", None),
        ("seek", 4.25),
        ("pause", None),
        ("stop", None),
    ]


def test_demo_dispatch_selection_does_not_reroute_runtime_audio():
    demo = build_demo_app()
    runtime_audio = RecordingRuntimeAudio()
    demo.runtime_audio = runtime_audio
    layer_id = demo.presentation().layers[0].layer_id

    demo.dispatch(SelectLayer(layer_id))

    assert runtime_audio.calls == []


def test_demo_dispatch_routes_playback_target_updates_runtime_audio():
    demo = build_demo_app()
    runtime_audio = RecordingRuntimeAudio()
    demo.runtime_audio = runtime_audio
    layer_id = demo.presentation().layers[0].layer_id

    demo.dispatch(SetActivePlaybackTarget(layer_id))

    assert runtime_audio.calls == [("mix", None)]


def test_widget_runtime_tick_tracks_provider_smoothly_without_seek_dispatch():
    app = QApplication.instance() or QApplication([])
    presentation = _audio_presentation()
    runtime_audio = FakeRuntimeAudio()
    dispatched: list[object] = []
    widget = TimelineWidget(
        presentation,
        on_intent=lambda intent: dispatched.append(intent) or presentation,
        runtime_audio=runtime_audio,
    )
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        runtime_audio.playing = True
        runtime_audio.current_time = 1.015
        widget._on_runtime_tick()
        first = widget.presentation.playhead

        runtime_audio.current_time = 1.033
        widget._on_runtime_tick()
        second = widget.presentation.playhead

        assert dispatched == []
        assert first == 1.015
        assert second == 1.033
        assert 0.0 < second - first < 0.03
        assert widget.presentation.current_time_label == "00:01.03"
    finally:
        widget.close()
        app.processEvents()


def test_widget_runtime_tick_extrapolates_from_backend_timing_snapshot(monkeypatch):
    app = QApplication.instance() or QApplication([])
    presentation = _audio_presentation()
    runtime_audio = FakeRuntimeAudio()
    dispatched: list[object] = []
    widget = TimelineWidget(
        presentation,
        on_intent=lambda intent: dispatched.append(intent) or presentation,
        runtime_audio=runtime_audio,
    )
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        runtime_audio.playing = True
        runtime_audio.current_time = 1.0
        runtime_audio.snapshot = RuntimeAudioTimingSnapshot(
            audible_time_seconds=1.0,
            clock_time_seconds=1.25,
            snapshot_monotonic_seconds=500.0,
            is_playing=True,
        )
        monkeypatch.setattr(time, "monotonic", lambda: 500.08)

        widget._on_runtime_tick()

        assert dispatched == []
        assert widget.presentation.playhead == pytest.approx(1.08)
        assert widget.presentation.current_time_label == "00:01.08"
    finally:
        widget.close()
        app.processEvents()


def test_widget_dispatch_preserves_runtime_playhead_on_audio_route_update():
    app = QApplication.instance() or QApplication([])
    presentation = _audio_presentation()
    runtime_audio = FakeRuntimeAudio()
    runtime_audio.playing = True
    runtime_audio.current_time = 4.25

    def _on_intent(intent):
        layer = presentation.layers[0]
        return replace(
            presentation,
            layers=[layer],
            active_playback_layer_id=intent.layer_id if isinstance(intent, SetActivePlaybackTarget) else presentation.active_playback_layer_id,
            active_playback_take_id=intent.take_id if isinstance(intent, SetActivePlaybackTarget) else presentation.active_playback_take_id,
            playhead=0.0,
            is_playing=False,
            current_time_label="00:00.00",
        )

    widget = TimelineWidget(presentation, on_intent=_on_intent, runtime_audio=runtime_audio)
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        widget._dispatch(SetActivePlaybackTarget(presentation.layers[0].layer_id))

        assert widget.presentation.playhead == 4.25
        assert widget.presentation.is_playing is True
        assert widget.presentation.current_time_label == "00:04.25"
    finally:
        widget.close()
        app.processEvents()


def test_widget_dispatch_uses_exact_runtime_clock_time_when_pausing():
    app = QApplication.instance() or QApplication([])
    presentation = _audio_presentation()
    runtime_audio = FakeRuntimeAudio()
    runtime_audio.playing = True
    runtime_audio.current_time = 4.257

    def _on_intent(intent):
        if isinstance(intent, Pause):
            runtime_audio.pause()
            return replace(
                presentation,
                playhead=4.25,
                is_playing=False,
                current_time_label="00:04.25",
            )
        return presentation

    widget = TimelineWidget(presentation, on_intent=_on_intent, runtime_audio=runtime_audio)
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        widget._dispatch(Pause())

        assert widget.presentation.playhead == 4.257
        assert widget.presentation.is_playing is False
        assert widget.presentation.current_time_label == "00:04.26"
    finally:
        widget.close()
        app.processEvents()


def test_widget_set_presentation_avoids_rebuilding_runtime_layers_when_sources_unchanged():
    app = QApplication.instance() or QApplication([])
    presentation = _audio_presentation()
    runtime_audio = CountingRuntimeAudio()
    widget = TimelineWidget(
        presentation,
        on_intent=lambda intent: presentation,
        runtime_audio=runtime_audio,
    )
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        assert runtime_audio.build_calls == 1
        assert runtime_audio.mix_calls == 0

        next_presentation = replace(
            presentation,
            layers=[replace(presentation.layers[0], is_selected=True)],
        )
        widget.set_presentation(next_presentation)

        assert runtime_audio.build_calls == 1
        assert runtime_audio.mix_calls == 1
    finally:
        widget.close()
        app.processEvents()


def test_widget_set_presentation_routes_runtime_audio_without_rebuild_when_playback_target_changes():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _event_slice_presentation(),
        active_playback_layer_id=LayerId("bed"),
    )
    runtime_audio = CountingRuntimeAudio()
    widget = TimelineWidget(
        presentation,
        on_intent=lambda intent: presentation,
        runtime_audio=runtime_audio,
    )
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        assert runtime_audio.build_calls == 1
        assert runtime_audio.mix_calls == 0

        next_presentation = replace(
            presentation,
            active_playback_layer_id=LayerId("kick_lane"),
        )
        widget.set_presentation(next_presentation)

        assert runtime_audio.build_calls == 1
        assert runtime_audio.mix_calls == 1
    finally:
        widget.close()
        app.processEvents()


def test_widget_uses_precise_runtime_timer_with_8ms_interval():
    app = QApplication.instance() or QApplication([])
    presentation = _audio_presentation()
    widget = TimelineWidget(presentation)
    try:
        assert widget._runtime_timer.timerType() == Qt.TimerType.PreciseTimer
        assert widget._runtime_timer.interval() == 8
    finally:
        widget.close()
        app.processEvents()


def test_audio_engine_keeps_injected_streams_on_low_latency_with_unspecified_callback_blocksize():
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    engine.play()

    assert engine.is_active is True
    assert engine.reported_output_latency_seconds == 0.0
    assert engine._stream is not None
    assert engine._stream.blocksize == 0
    assert engine._stream.latency == "low"
    assert engine._stream.prime_output_buffers_using_stream_callback is True
    engine.shutdown()


def test_widget_runtime_ticks_do_not_snap_backward_during_audio_route_churn():
    app = QApplication.instance() or QApplication([])
    base_presentation = _audio_presentation()
    runtime_audio = FakeRuntimeAudio()
    runtime_audio.playing = True
    state = {"presentation": base_presentation}

    def _on_intent(intent):
        current = state["presentation"]
        updated = replace(
            current,
            active_playback_layer_id=intent.layer_id if isinstance(intent, SetActivePlaybackTarget) else current.active_playback_layer_id,
            active_playback_take_id=intent.take_id if isinstance(intent, SetActivePlaybackTarget) else current.active_playback_take_id,
            playhead=0.0,
            is_playing=False,
            current_time_label="00:00.00",
        )
        state["presentation"] = updated
        return updated

    widget = TimelineWidget(state["presentation"], on_intent=_on_intent, runtime_audio=runtime_audio)
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        samples: list[float] = []

        runtime_audio.current_time = 4.000
        widget._on_runtime_tick()
        samples.append(widget.presentation.playhead)

        runtime_audio.current_time = 4.042
        widget._on_runtime_tick()
        samples.append(widget.presentation.playhead)

        runtime_audio.current_time = 4.018
        widget._dispatch(SetActivePlaybackTarget(base_presentation.layers[0].layer_id))
        samples.append(widget.presentation.playhead)
        widget._on_runtime_tick()
        samples.append(widget.presentation.playhead)

        runtime_audio.current_time = 4.019
        widget._dispatch(SetActivePlaybackTarget(base_presentation.layers[0].layer_id))
        samples.append(widget.presentation.playhead)
        widget._on_runtime_tick()
        samples.append(widget.presentation.playhead)

        runtime_audio.current_time = 4.083
        widget._on_runtime_tick()
        samples.append(widget.presentation.playhead)

        assert samples == sorted(samples)
        assert widget.presentation.playhead == 4.083
        assert widget.presentation.current_time_label == "00:04.08"
    finally:
        widget.close()
        app.processEvents()


def test_widget_seek_churn_keeps_seek_anchor_through_stale_runtime_samples():
    app = QApplication.instance() or QApplication([])
    base_presentation = _audio_presentation()
    runtime_audio = FakeRuntimeAudio()
    state = {"presentation": base_presentation}

    def _on_intent(intent):
        current = state["presentation"]
        layer = current.layers[0]
        if isinstance(intent, Play):
            runtime_audio.play()
            updated = replace(
                current,
                is_playing=True,
                playhead=1.0,
                current_time_label="00:01.00",
            )
        elif isinstance(intent, Seek):
            runtime_audio.seek(intent.position)
            updated = replace(
                current,
                is_playing=True,
                playhead=float(intent.position),
                current_time_label="00:00.75",
            )
        elif isinstance(intent, SetActivePlaybackTarget):
            updated = replace(
                current,
                active_playback_layer_id=intent.layer_id,
                active_playback_take_id=intent.take_id,
                playhead=0.0,
                is_playing=False,
                current_time_label="00:00.00",
            )
        else:
            updated = current
        state["presentation"] = updated
        return updated

    widget = TimelineWidget(state["presentation"], on_intent=_on_intent, runtime_audio=runtime_audio)
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        runtime_audio.current_time = 1.000
        widget._dispatch(Play())
        assert widget.presentation.playhead == 1.0

        runtime_audio.current_time = 1.040
        widget._on_runtime_tick()
        assert widget.presentation.playhead == 1.04

        widget._dispatch(Seek(0.75))
        assert widget.presentation.playhead == 0.75
        assert widget.presentation.current_time_label == "00:00.75"

        runtime_audio.current_time = 0.710
        widget._on_runtime_tick()
        assert widget.presentation.playhead == 0.75

        runtime_audio.current_time = 0.720
        widget._dispatch(SetActivePlaybackTarget(base_presentation.layers[0].layer_id))
        assert widget.presentation.playhead == 0.75

        runtime_audio.current_time = 0.710
        widget._dispatch(SetActivePlaybackTarget(base_presentation.layers[0].layer_id))
        assert widget.presentation.playhead == 0.75

        runtime_audio.current_time = 0.810
        widget._on_runtime_tick()
        assert widget.presentation.playhead == 0.81
        assert widget.presentation.current_time_label == "00:00.81"
    finally:
        widget.close()
        app.processEvents()


def test_widget_dispatch_preserves_local_scroll_when_seek_updates_arrive_with_stale_scroll_state():
    app = QApplication.instance() or QApplication([])
    initial = replace(_audio_presentation(), scroll_x=972.0, pixels_per_second=180.0)

    def _on_intent(intent):
        if isinstance(intent, Seek):
            # Simulate app response that forgot viewport state.
            return replace(initial, playhead=float(intent.position), scroll_x=0.0)
        return initial

    widget = TimelineWidget(initial, on_intent=_on_intent, runtime_audio=None)
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        widget._dispatch(Seek(60.0))

        assert widget.presentation.playhead == 60.0
        assert widget.presentation.scroll_x == 972.0
    finally:
        widget.close()
        app.processEvents()


def test_widget_zoom_keeps_anchor_time_under_cursor():
    app = QApplication.instance() or QApplication([])
    initial = replace(_audio_presentation(), scroll_x=840.0, pixels_per_second=180.0)
    widget = TimelineWidget(initial, runtime_audio=None)
    widget._runtime_timer.stop()
    try:
        widget.resize(1280, 360)
        widget.show()
        app.processEvents()

        anchor_x = 760.0
        content_start_x = float(widget._canvas._header_width)
        before_time = seek_time_for_x(
            anchor_x,
            scroll_x=widget.presentation.scroll_x,
            pixels_per_second=widget.presentation.pixels_per_second,
            content_start_x=content_start_x,
        )

        widget._zoom_from_input(120, anchor_x)

        after_time = seek_time_for_x(
            anchor_x,
            scroll_x=widget.presentation.scroll_x,
            pixels_per_second=widget.presentation.pixels_per_second,
            content_start_x=content_start_x,
        )

        assert widget.presentation.pixels_per_second > initial.pixels_per_second
        assert abs(after_time - before_time) < 0.02
    finally:
        widget.close()
        app.processEvents()
