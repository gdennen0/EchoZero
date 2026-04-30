"""Widget-oriented runtime-audio support cases.
Exists to keep runtime timing, seek churn, and widget rebuild coverage separate from controller support tests.
Connects the compatibility wrapper to the bounded runtime-audio widget slice.
"""

from tests.ui.runtime_audio_shared_support import *  # noqa: F401,F403
from echozero.ui.FEEL import TIMELINE_RUNTIME_TICK_IDLE_MS

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
            selected_layer_id=(
                intent.layer_id
                if isinstance(intent, SetLayerMute)
                else presentation.selected_layer_id
            ),
            selected_take_id=(
                presentation.selected_take_id
            ),
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

        widget._dispatch(SetLayerMute(layer_id=presentation.layers[0].layer_id, muted=True))

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


def test_widget_set_presentation_routes_runtime_audio_without_rebuild_when_selection_changes():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _event_slice_presentation(),
        selected_layer_id=LayerId("bed"),
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
            selected_layer_id=LayerId("kick_lane"),
        )
        widget.set_presentation(next_presentation)

        assert runtime_audio.build_calls == 1
        assert runtime_audio.mix_calls == 1
    finally:
        widget.close()
        app.processEvents()


def test_widget_set_presentation_rebuilds_runtime_layers_when_event_slice_source_changes():
    app = QApplication.instance() or QApplication([])
    presentation = _event_slice_presentation()
    runtime_audio = SignatureAwareRuntimeAudio()
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

        changed_source = replace(
            presentation,
            layers=[
                presentation.layers[0],
                replace(
                    presentation.layers[1],
                    playback_source_ref="kick_alt.wav",
                ),
            ],
        )
        widget.set_presentation(changed_source)

        assert runtime_audio.build_calls == 2
        assert runtime_audio.mix_calls == 0

        soloed_source = replace(
            changed_source,
            layers=[changed_source.layers[0], replace(changed_source.layers[1], gain_db=-6.0)],
        )
        widget.set_presentation(soloed_source)

        assert runtime_audio.build_calls == 2
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
        assert widget._runtime_timer.interval() == TIMELINE_RUNTIME_TICK_IDLE_MS
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


def test_runtime_controller_uses_unified_sounddevice_backend_for_audio_layers():
    app = QApplication.instance() or QApplication([])
    presentation = _audio_presentation()
    controller = TimelineRuntimeAudioController(
        audio_loader=lambda _path: (np.ones(4410, dtype=np.float32), 44100),
    )
    try:
        controller.build_for_presentation(presentation)
        state = controller.snapshot_state(presentation)

        assert state.backend_name == "sounddevice"
        assert (
            controller.engine.mixer.get_layer(TimelineRuntimeAudioController._PRIMARY_TRACK_ID)
            is not None
        )
    finally:
        controller.shutdown()
        app.processEvents()


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
            selected_layer_id=(
                intent.layer_id
                if isinstance(intent, SetLayerMute)
                else current.selected_layer_id
            ),
            selected_take_id=(
                current.selected_take_id
            ),
            playhead=0.0,
            is_playing=False,
            current_time_label="00:00.00",
        )
        state["presentation"] = updated
        return updated

    widget = TimelineWidget(
        state["presentation"], on_intent=_on_intent, runtime_audio=runtime_audio
    )
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
        widget._dispatch(SetLayerMute(layer_id=base_presentation.layers[0].layer_id, muted=True))
        samples.append(widget.presentation.playhead)
        widget._on_runtime_tick()
        samples.append(widget.presentation.playhead)

        runtime_audio.current_time = 4.019
        widget._dispatch(SetLayerMute(layer_id=base_presentation.layers[0].layer_id, muted=False))
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
        elif isinstance(intent, SetLayerMute):
            updated = replace(
                current,
                layers=[
                    replace(layer, muted=bool(intent.muted))
                    if layer.layer_id == intent.layer_id
                    else layer
                    for layer in current.layers
                ],
                playhead=0.0,
                is_playing=False,
                current_time_label="00:00.00",
            )
        else:
            updated = current
        state["presentation"] = updated
        return updated

    widget = TimelineWidget(
        state["presentation"], on_intent=_on_intent, runtime_audio=runtime_audio
    )
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
        widget._dispatch(SetLayerMute(layer_id=base_presentation.layers[0].layer_id, muted=True))
        assert widget.presentation.playhead == 0.75

        runtime_audio.current_time = 0.710
        widget._dispatch(SetLayerMute(layer_id=base_presentation.layers[0].layer_id, muted=False))
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


def test_widget_external_presentation_update_preserves_local_viewport_for_same_timeline():
    app = QApplication.instance() or QApplication([])
    initial = replace(_audio_presentation(), scroll_x=972.0, pixels_per_second=180.0)
    widget = TimelineWidget(initial, runtime_audio=None)
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        updated = replace(
            initial,
            playhead=12.0,
            scroll_x=0.0,
            pixels_per_second=100.0,
        )
        widget.apply_external_presentation_update(updated)

        assert widget.presentation.playhead == 12.0
        assert widget.presentation.scroll_x == 972.0
        assert widget.presentation.pixels_per_second == 180.0
    finally:
        widget.close()
        app.processEvents()


def test_widget_external_presentation_update_resets_viewport_for_new_timeline():
    app = QApplication.instance() or QApplication([])
    initial = replace(_audio_presentation(), scroll_x=972.0, pixels_per_second=180.0)
    widget = TimelineWidget(initial, runtime_audio=None)
    widget._runtime_timer.stop()
    try:
        widget.resize(1200, 320)
        widget.show()
        app.processEvents()

        updated = replace(
            initial,
            timeline_id=type(initial.timeline_id)("timeline_new"),
            scroll_x=144.0,
            pixels_per_second=120.0,
        )
        widget.apply_external_presentation_update(updated)

        assert widget.presentation.timeline_id == updated.timeline_id
        assert widget.presentation.scroll_x == 144.0
        assert widget.presentation.pixels_per_second == 120.0
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

__all__ = [name for name in globals() if name.startswith("test_")]
