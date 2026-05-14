"""Controller-oriented runtime-audio support cases.
Exists to isolate controller and demo-dispatch coverage from widget timing support tests.
Connects the compatibility wrapper to the bounded runtime-audio controller slice.
"""

import json
import threading
from pathlib import Path

from tests.ui.runtime_audio_shared_support import *  # noqa: F401,F403


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

    engine_layer = engine.mixer.get_layer(TimelineRuntimeAudioController._PRIMARY_TRACK_ID)
    assert engine.transport.is_playing is True
    assert engine_layer is not None
    assert round(engine_layer.volume, 3) == round(10 ** (-6.0 / 20.0), 3)
    controller.shutdown()


def test_runtime_controller_compensates_for_reported_output_latency_while_playing():
    presentation = _audio_presentation()
    engine = AudioEngine(stream_factory=lambda **kwargs: FakeStream(**kwargs | {"latency": 0.1}))
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
    assert state.active_layer_id == presentation.selected_layer_id
    assert state.active_take_id is None
    assert len(state.active_sources) == 1
    assert state.active_sources[0].source_ref == "demo.wav"
    assert state.diagnostics.output_device == "default"
    assert state.diagnostics.last_transition == ""
    controller.shutdown()


def test_runtime_controller_snapshot_state_reports_engine_diagnostics():
    presentation = _audio_presentation()
    engine = AudioEngine(
        stream_factory=lambda **kwargs: FakeStream(**kwargs | {"latency": 0.2}),
        stream_latency="low",
        stream_blocksize=512,
        prime_output_buffers_using_stream_callback=False,
        output_device="Built-in Output",
    )
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda path: (np.ones(4410, dtype=np.float32), 44100),
    )

    controller.build_for_presentation(presentation)
    controller.play()

    outdata = np.zeros((256, 1), dtype=np.float32)
    engine._audio_callback(outdata, 256, None, "underflow")
    state = controller.snapshot_state(presentation)

    assert state.diagnostics.glitch_count == 1
    assert state.diagnostics.last_audio_status == "underflow"
    assert state.diagnostics.output_device == "Built-in Output"
    assert state.diagnostics.stream_latency == "low"
    assert state.diagnostics.stream_blocksize == 512
    assert state.diagnostics.prime_output_buffers_using_stream_callback is False
    assert state.diagnostics.last_transition == "play"
    assert state.diagnostics.last_track_sync_reason == "track-signature-changed"
    assert state.diagnostics.structural_rebuild_count >= 1
    assert state.diagnostics.max_structural_rebuild_ms >= 0.0
    controller.shutdown()


def test_runtime_controller_seek_while_playing_keeps_transport_running():
    presentation = _audio_presentation()
    engine = AudioEngine(stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda path: (np.ones(44100, dtype=np.float32), 44100),
    )

    controller.build_for_presentation(presentation)
    controller.play()
    controller.seek(4.25)

    assert controller.is_playing() is True
    assert engine.transport.is_playing is True
    assert engine.clock.position_seconds == pytest.approx(4.25)
    controller.shutdown()


def test_runtime_controller_mix_sync_never_triggers_structural_decode_reload():
    presentation = _event_slice_presentation()
    load_calls: list[str] = []

    def _loader(path: str):
        load_calls.append(path)
        if path == "bed.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([1.0, 0.5], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(audio_loader=_loader)
    try:
        controller.build_for_presentation(presentation)
        assert set(load_calls) == {"bed.wav", "kick.wav"}
        assert len(load_calls) == 2

        changed = replace(
            presentation,
            layers=[
                presentation.layers[0],
                replace(
                    presentation.layers[1],
                    events=[
                        *presentation.layers[1].events,
                        EventPresentation(
                            event_id=EventId("kick_3"),
                            start=1.4,
                            end=1.5,
                            label="Kick",
                        ),
                    ],
                ),
            ],
        )
        controller.apply_mix_state(changed)

        assert set(load_calls) == {"bed.wav", "kick.wav"}
        assert len(load_calls) == 2
        assert controller._last_track_sync_reason == "mix-state-pending-structure-sync"

        controller.sync_structure_state(changed)
        state = controller.snapshot_state(changed)
        assert state.diagnostics.structural_rebuild_count >= 2
        assert state.diagnostics.last_structural_rebuild_ms >= 0.0
        assert (
            state.diagnostics.max_structural_rebuild_ms
            >= state.diagnostics.last_structural_rebuild_ms
        )
    finally:
        controller.shutdown()


def test_runtime_controller_decodes_selected_audio_source_on_build():
    presentation = _audio_presentation()
    load_calls: list[str] = []

    def _loader(path: str):
        load_calls.append(path)
        return np.ones(4410, dtype=np.float32), 44100

    controller = TimelineRuntimeAudioController(audio_loader=_loader)
    try:
        signature = controller.presentation_signature(presentation)

        assert load_calls == []

        controller.build_for_presentation(presentation)

        assert signature == (("runtime_audio", "audio:demo.wav|outputs_1_2"),)
        assert load_calls == ["demo.wav"]
    finally:
        controller.shutdown()


def test_runtime_controller_state_queries_do_not_decode_or_raise_for_missing_event_assets():
    presentation = replace(
        _event_slice_presentation(),
        selected_layer_id=LayerId("kick_lane"),
    )
    load_calls: list[str] = []

    def _loader(path: str):
        load_calls.append(path)
        raise FileNotFoundError(path)

    controller = TimelineRuntimeAudioController(audio_loader=_loader)
    try:
        signature = controller.presentation_signature(presentation)
        state = controller.snapshot_state(presentation)

        assert signature == (
            ("bed", "audio:bed.wav|outputs_1_2"),
            ("kick_lane", "event:kick.wav:0.500000:0:0,1.000000:0:0|outputs_1_2"),
        )
        assert {(source.layer_id, source.source_ref) for source in state.active_sources} == {
            ("bed", "bed.wav"),
            ("kick_lane", "kick.wav"),
        }
        assert load_calls == []
    finally:
        controller.shutdown()


def test_runtime_controller_can_prefer_sounddevice_backend_for_audio_layers():
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


def test_runtime_controller_preserves_stereo_audio_layer_channels():
    presentation = _audio_presentation()
    engine = AudioEngine(sample_rate=44100, channels=2, stream_factory=_fake_stream_factory)
    stereo = np.column_stack(
        (
            np.ones(128, dtype=np.float32),
            -np.ones(128, dtype=np.float32),
        )
    )
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (stereo, 44100),
    )

    controller.build_for_presentation(presentation)
    mixed = engine.mixer.read_mix(0, 128, channels=2)

    assert mixed.shape == (128, 2)
    np.testing.assert_array_equal(mixed[:, 0], np.ones(128, dtype=np.float32))
    np.testing.assert_array_equal(mixed[:, 1], -np.ones(128, dtype=np.float32))
    controller.shutdown()


def test_runtime_controller_routes_song_and_timecode_layers_to_separate_output_pairs():
    base = build_demo_app().presentation()
    song_layer = LayerPresentation(
        layer_id=LayerId("song_layer"),
        title="Song",
        kind=LayerKind.AUDIO,
        source_audio_path="song.wav",
    )
    timecode_layer = LayerPresentation(
        layer_id=LayerId("timecode_layer"),
        title="Timecode",
        kind=LayerKind.AUDIO,
        source_audio_path="ltc.wav",
        output_bus="outputs_3_4",
    )
    presentation = replace(
        base,
        layers=[song_layer, timecode_layer],
        selected_layer_id=song_layer.layer_id,
    )
    engine = AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "song.wav":
            return np.array([0.25, -0.25], dtype=np.float32), 44100
        if path == "ltc.wav":
            return np.array([0.75, -0.75], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(0, 2, channels=4)
    np.testing.assert_array_equal(
        mixed,
        np.array(
            [
                [0.25, 0.25, 0.75, 0.75],
                [-0.25, -0.25, -0.75, -0.75],
            ],
            dtype=np.float32,
        ),
    )
    assert engine.mixer.get_layer(TimelineRuntimeAudioController._PRIMARY_TRACK_ID) is None
    assert engine.mixer.get_layer("__ez_route__song_layer") is not None
    assert engine.mixer.get_layer("__ez_route__timecode_layer") is not None

    state = controller.snapshot_state(presentation)
    assert {source.layer_id for source in state.active_sources} == {
        "song_layer",
        "timecode_layer",
    }
    controller.shutdown()


def test_runtime_controller_reconfigure_device_rebuilds_and_restores_playback_state():
    presentation = _audio_presentation()
    engines = [
        AudioEngine(sample_rate=44100, channels=2, stream_factory=_fake_stream_factory),
        AudioEngine(sample_rate=48000, channels=2, stream_factory=_fake_stream_factory),
    ]

    controller = TimelineRuntimeAudioController(
        engine_factory=lambda: engines.pop(0),
        audio_loader=lambda _path: (np.ones(44100, dtype=np.float32), 44100),
    )
    controller.build_for_presentation(presentation)
    controller.play()
    controller.seek(1.25)

    result = controller.reconfigure_device(
        device_spec={"output_device": "next", "sample_rate": 48000},
    )
    state = controller.snapshot_state(presentation)

    assert result["reason"] == "device-change"
    assert result["device_reinit_count"] == 1
    assert controller.engine.sample_rate == 48000
    assert controller.is_playing() is True
    assert controller.current_time_seconds() == pytest.approx(1.25)
    assert state.diagnostics.device_reinit_count == 1
    assert state.diagnostics.last_device_reinit_reason == "device-change"
    controller.shutdown()


def test_runtime_controller_recomputes_routing_after_device_channel_changes():
    base = build_demo_app().presentation()
    song_layer = LayerPresentation(
        layer_id=LayerId("song_layer"),
        title="Song",
        kind=LayerKind.AUDIO,
        source_audio_path="song.wav",
    )
    timecode_layer = LayerPresentation(
        layer_id=LayerId("timecode_layer"),
        title="Timecode",
        kind=LayerKind.AUDIO,
        source_audio_path="ltc.wav",
        output_bus="outputs_3_4",
    )
    presentation = replace(
        base,
        layers=[song_layer, timecode_layer],
        selected_layer_id=song_layer.layer_id,
    )
    engines = [
        AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory),
        AudioEngine(sample_rate=44100, channels=2, stream_factory=_fake_stream_factory),
        AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory),
    ]

    def _loader(path: str):
        if path == "song.wav":
            return np.array([0.25, -0.25], dtype=np.float32), 44100
        if path == "ltc.wav":
            return np.array([0.75, -0.75], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(
        engine_factory=lambda: engines.pop(0),
        audio_loader=_loader,
    )
    controller.build_for_presentation(presentation)
    wide_mix = controller.engine.mixer.read_mix(0, 2, channels=4)

    controller.reconfigure_device(device_spec={"channels": 2})
    narrow_mix = controller.engine.mixer.read_mix(0, 2, channels=2)
    narrow_state = controller.snapshot_state(presentation)

    controller.reconfigure_device(device_spec={"channels": 4})
    restored_mix = controller.engine.mixer.read_mix(0, 2, channels=4)

    np.testing.assert_array_equal(
        wide_mix[:, 2:4],
        np.array([[0.75, 0.75], [-0.75, -0.75]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        narrow_mix,
        np.array([[0.25, 0.25], [-0.25, -0.25]], dtype=np.float32),
    )
    assert "routes-exceed-hardware" in narrow_state.diagnostics.route_resolution_summary
    np.testing.assert_array_equal(restored_mix, wide_mix)
    controller.shutdown()


def test_runtime_controller_routes_active_take_when_multichannel_mode_is_enabled():
    base = build_demo_app().presentation()
    alt_take = TakeLanePresentation(
        take_id=TakeId("take_alt"),
        name="Alt",
        kind=LayerKind.AUDIO,
        source_audio_path="alt.wav",
        playback_source_ref="alt.wav",
    )
    song_layer = LayerPresentation(
        layer_id=LayerId("song_layer"),
        title="Song",
        kind=LayerKind.AUDIO,
        source_audio_path="main.wav",
        playback_source_ref="main.wav",
        takes=[alt_take],
    )
    timecode_layer = LayerPresentation(
        layer_id=LayerId("timecode_layer"),
        title="Timecode",
        kind=LayerKind.AUDIO,
        source_audio_path="ltc.wav",
        output_bus="outputs_3_4",
    )
    presentation = replace(
        base,
        layers=[song_layer, timecode_layer],
        selected_layer_id=song_layer.layer_id,
        selected_take_id=alt_take.take_id,
    )
    engine = AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "main.wav":
            return np.array([0.25, -0.25], dtype=np.float32), 44100
        if path == "alt.wav":
            return np.array([0.5, -0.5], dtype=np.float32), 44100
        if path == "ltc.wav":
            return np.array([0.75, -0.75], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(0, 2, channels=4)
    np.testing.assert_array_equal(
        mixed,
        np.array(
            [
                [0.5, 0.5, 0.75, 0.75],
                [-0.5, -0.5, -0.75, -0.75],
            ],
            dtype=np.float32,
        ),
    )
    controller.shutdown()


def test_runtime_controller_routes_layer_to_wide_output_span():
    base = build_demo_app().presentation()
    song_layer = LayerPresentation(
        layer_id=LayerId("song_layer"),
        title="Song",
        kind=LayerKind.AUDIO,
        source_audio_path="song.wav",
        output_bus="outputs_1_4",
    )
    presentation = replace(
        base,
        layers=[song_layer],
        selected_layer_id=song_layer.layer_id,
        playback_output_channels=4,
    )
    engine = AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory)

    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (np.array([0.25, -0.25], dtype=np.float32), 44100),
    )
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(0, 2, channels=4)
    np.testing.assert_array_equal(
        mixed,
        np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [-0.25, -0.25, -0.25, -0.25],
            ],
            dtype=np.float32,
        ),
    )
    routed = engine.mixer.get_layer("__ez_route__song_layer")
    assert routed is not None
    assert routed.output_bus == "outputs_1_4"
    controller.shutdown()


def test_runtime_controller_preserves_explicit_route_when_device_channel_count_shrinks():
    base = build_demo_app().presentation()
    song_layer = LayerPresentation(
        layer_id=LayerId("song_layer"),
        title="Song",
        kind=LayerKind.AUDIO,
        source_audio_path="song.wav",
        output_bus="outputs_7_8",
    )
    presentation = replace(
        base,
        layers=[song_layer],
        selected_layer_id=song_layer.layer_id,
        playback_output_channels=4,
    )
    engine = AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory)

    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (np.array([0.25, -0.25], dtype=np.float32), 44100),
    )
    controller.build_for_presentation(presentation)

    primary = engine.mixer.get_layer(TimelineRuntimeAudioController._PRIMARY_TRACK_ID)
    assert primary is None
    routed = engine.mixer.get_layer("__ez_route__song_layer")
    assert routed is not None
    assert routed.output_bus == "outputs_7_8"

    mixed = engine.mixer.read_mix(0, 2, channels=4)
    np.testing.assert_array_equal(
        mixed,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    controller.shutdown()


def test_runtime_controller_does_not_leak_ltc_to_main_pair_when_presentation_channels_are_stale():
    base = build_demo_app().presentation()
    song_layer = LayerPresentation(
        layer_id=LayerId("song_layer"),
        title="Song",
        kind=LayerKind.AUDIO,
        source_audio_path="song.wav",
    )
    timecode_layer = LayerPresentation(
        layer_id=LayerId("timecode_layer"),
        title="Timecode",
        kind=LayerKind.AUDIO,
        source_audio_path="ltc.wav",
        output_bus="outputs_3_4",
    )
    presentation = replace(
        base,
        layers=[song_layer, timecode_layer],
        selected_layer_id=song_layer.layer_id,
        playback_output_channels=2,
    )
    engine = AudioEngine(sample_rate=44100, channels=2, stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "song.wav":
            return np.array([0.25, -0.25], dtype=np.float32), 44100
        if path == "ltc.wav":
            return np.array([0.75, -0.75], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(0, 2, channels=2)
    np.testing.assert_array_equal(
        mixed,
        np.array(
            [
                [0.25, 0.25],
                [-0.25, -0.25],
            ],
            dtype=np.float32,
        ),
    )
    routed_timecode = engine.mixer.get_layer("__ez_route__timecode_layer")
    assert routed_timecode is not None
    assert routed_timecode.output_bus == "outputs_3_4"
    controller.shutdown()


def test_runtime_controller_keeps_active_event_lane_when_routed_layers_are_present():
    base = _event_slice_presentation()
    presentation = replace(
        base,
        layers=[
            replace(base.layers[0], output_bus="outputs_1_2"),
            base.layers[1],
        ],
        selected_layer_id=LayerId("kick_lane"),
    )
    engine = AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "bed.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([1.0, 0.5], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(int(0.5 * 44100), 2, channels=4)
    np.testing.assert_array_equal(
        mixed,
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.75, 0.75, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert engine.mixer.get_layer("__ez_route__bed") is not None
    assert engine.mixer.get_layer("__ez_route__kick_lane") is not None

    state = controller.snapshot_state(presentation)
    assert {source.layer_id for source in state.active_sources} == {"bed", "kick_lane"}
    controller.shutdown()


def test_runtime_controller_preview_clip_plays_sliced_audio_on_preview_engine():
    engine = AudioEngine(stream_factory=_fake_stream_factory)
    decoded = np.arange(10, dtype=np.float32)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (decoded, 10),
    )

    played = controller.preview_clip("kick.wav", start_seconds=0.2, end_seconds=0.6)

    assert played is True
    preview_layer = getattr(engine, "_overlay_buffer", None)
    assert preview_layer is not None
    np.testing.assert_array_equal(preview_layer, np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    assert not np.shares_memory(preview_layer, decoded)
    assert engine.overlay_active is True
    controller.shutdown()


def test_runtime_controller_preview_clip_tears_down_preview_stream_after_end():
    engine = AudioEngine(sample_rate=10, stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (np.arange(10, dtype=np.float32), 10),
    )

    played = controller.preview_clip("kick.wav", start_seconds=0.0, end_seconds=0.2)

    assert played is True
    outdata = np.zeros((256, 1), dtype=np.float32)
    engine._audio_callback(outdata, 256, None, None)
    engine._audio_callback(outdata, 256, None, None)

    controller.current_time_seconds()

    assert engine.overlay_active is False
    controller.shutdown()


def test_runtime_controller_snapshot_exposes_preview_audio_runtime_sensor_events():
    presentation = _audio_presentation()
    engine = AudioEngine(sample_rate=10, stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (np.arange(10, dtype=np.float32), 10),
    )

    assert controller.preview_clip("kick.wav", start_seconds=0.0, end_seconds=0.4)

    state = controller.snapshot_state(presentation)
    events = state.diagnostics.recent_audio_runtime_events
    kinds = {str(event.get("kind")) for event in events}

    assert "preview-start" in kinds
    assert "overlay-start" in kinds
    assert any(event.get("source") == "audio_engine" for event in events)
    assert any(event.get("source") == "playback_controller" for event in events)
    controller.shutdown()


def test_runtime_controller_audio_diagnostics_capture_writes_bundle(tmp_path):
    presentation = _audio_presentation()
    engine = AudioEngine(sample_rate=10, channels=1, stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (np.arange(10, dtype=np.float32), 10),
    )

    started = controller.start_audio_diagnostics_capture(
        output_dir=tmp_path,
        include_audio_buffers=True,
        max_audio_blocks=4,
    )
    assert started["active"] is True

    assert controller.preview_clip("kick.wav", start_seconds=0.0, end_seconds=0.4)
    outdata = np.zeros((4, 1), dtype=np.float32)
    engine._audio_callback(outdata, 4, None, None)
    controller.snapshot_state(presentation)

    stopped = controller.stop_audio_diagnostics_capture()

    assert stopped["active"] is False
    assert int(stopped["audio_block_count"]) >= 1
    assert Path(str(stopped["json_path"])).exists()
    assert Path(str(stopped["npy_path"])).exists()
    assert Path(str(stopped["wav_path"])).exists()
    payload = json.loads(Path(str(stopped["json_path"])).read_text(encoding="utf-8"))
    event_kinds = {str(event.get("kind")) for event in payload["runtime_sensor_events"]}
    assert "preview-start" in event_kinds
    assert "overlay-start" in event_kinds
    assert payload["device_config"]["sample_rate"] == 10
    controller.shutdown()


def test_runtime_controller_rapid_preview_replacement_is_declick_safe_and_non_mutating():
    engine = AudioEngine(sample_rate=48000, channels=1, stream_factory=_fake_stream_factory)
    decoded = np.concatenate(
        (
            np.ones(512, dtype=np.float32),
            -np.ones(512, dtype=np.float32),
        )
    )
    original = decoded.copy()
    controller = TimelineRuntimeAudioController(
        engine=engine,
        audio_loader=lambda _path: (decoded, 48000),
    )

    assert controller.preview_clip("events.wav", start_seconds=0.0, end_seconds=512 / 48000)
    first = np.zeros((128, 1), dtype=np.float32)
    engine._audio_callback(first, 128, None, None)

    assert controller.preview_clip("events.wav", start_seconds=512 / 48000, end_seconds=1024 / 48000)
    replaced = np.zeros((128, 1), dtype=np.float32)
    continued = np.zeros((128, 1), dtype=np.float32)
    engine._audio_callback(replaced, 128, None, None)
    engine._audio_callback(continued, 128, None, None)

    np.testing.assert_array_equal(decoded, original)
    staged = getattr(engine, "_overlay_buffer", None)
    assert staged is not None
    np.testing.assert_array_equal(staged, decoded[512:1024])
    joined = np.concatenate((first, replaced, continued), axis=0)
    assert float(np.max(np.abs(np.diff(joined, axis=0)))) <= 0.18
    controller.shutdown()


def test_runtime_controller_mixes_all_playable_layers_by_default():
    presentation = replace(
        _event_slice_presentation(),
        selected_layer_id=LayerId("kick_lane"),
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

    assert engine.mixer.get_layer(TimelineRuntimeAudioController._PRIMARY_TRACK_ID) is None
    assert engine.mixer.get_layer("__ez_route__bed") is not None
    assert engine.mixer.get_layer("__ez_route__kick_lane") is not None
    np.testing.assert_array_almost_equal(mixed, np.array([1.0, 0.75], dtype=np.float32))
    controller.shutdown()


def test_runtime_controller_keeps_song_and_stems_sample_aligned_at_shared_anchor():
    base = build_demo_app().presentation()
    layer_ids = (
        "song_layer",
        "stem_vocals",
        "stem_drums",
        "stem_bass",
        "stem_other",
    )
    source_paths = {
        "song_layer": "song.wav",
        "stem_vocals": "vocals.wav",
        "stem_drums": "drums.wav",
        "stem_bass": "bass.wav",
        "stem_other": "other.wav",
    }
    amplitudes = {
        "song.wav": 0.10,
        "vocals.wav": 0.12,
        "drums.wav": 0.14,
        "bass.wav": 0.16,
        "other.wav": 0.18,
    }
    layers = [
        LayerPresentation(
            layer_id=LayerId(layer_id),
            title=layer_id,
            kind=LayerKind.AUDIO,
            source_audio_path=source_paths[layer_id],
        )
        for layer_id in layer_ids
    ]
    presentation = replace(
        base,
        layers=layers,
        selected_layer_id=layers[0].layer_id,
    )
    engine = AudioEngine(sample_rate=100, stream_factory=_fake_stream_factory)
    anchor_sample = 400
    total_samples = 1200

    def _loader(path: str):
        if path not in amplitudes:
            raise AssertionError(path)
        buffer = np.zeros(total_samples, dtype=np.float32)
        buffer[anchor_sample] = amplitudes[path]
        return buffer, 100

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    expected_mix = np.zeros(9, dtype=np.float32)
    expected_mix[4] = sum(amplitudes.values())
    mixed = engine.mixer.read_mix(anchor_sample - 4, 9)
    np.testing.assert_array_almost_equal(mixed, expected_mix)

    for layer_id in layer_ids:
        routed_layer = engine.mixer.get_layer(f"__ez_route__{layer_id}")
        assert routed_layer is not None
        assert int(np.argmax(routed_layer.buffer)) == anchor_sample

    controller.shutdown()


def test_runtime_controller_resamples_mixed_sample_rate_layers_before_engine_mix():
    from echozero.audio.layer import resample_buffer

    base = build_demo_app().presentation()
    layers = [
        LayerPresentation(
            layer_id=LayerId("song_layer"),
            title="Song",
            kind=LayerKind.AUDIO,
            source_audio_path="song.wav",
        ),
        LayerPresentation(
            layer_id=LayerId("stem_layer"),
            title="Stem",
            kind=LayerKind.AUDIO,
            source_audio_path="stem.wav",
        ),
    ]
    presentation = replace(
        base,
        layers=layers,
        selected_layer_id=layers[0].layer_id,
    )
    engine = AudioEngine(sample_rate=48000, stream_factory=_fake_stream_factory)
    duration_seconds = 2.0
    song_sample_rate = 48000
    stem_sample_rate = 44100
    rng = np.random.default_rng(12345)
    song_buffer = (
        rng.standard_normal(int(duration_seconds * song_sample_rate)).astype(np.float32) * 0.1
    )
    stem_buffer = resample_buffer(song_buffer, song_sample_rate, stem_sample_rate)

    def _loader(path: str):
        if path == "song.wav":
            return song_buffer, song_sample_rate
        if path == "stem.wav":
            return stem_buffer, stem_sample_rate
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    routed_song = engine.mixer.get_layer("__ez_route__song_layer")
    routed_stem = engine.mixer.get_layer("__ez_route__stem_layer")
    assert routed_song is not None
    assert routed_stem is not None
    assert routed_song.sample_rate == engine.sample_rate
    assert routed_stem.sample_rate == engine.sample_rate
    assert routed_song.duration_samples == len(song_buffer)
    assert routed_stem.duration_samples == len(song_buffer)
    assert float(np.corrcoef(routed_song.buffer, routed_stem.buffer)[0, 1]) > 0.95

    controller.shutdown()


def test_runtime_controller_plays_layers_without_explicit_playback_target():
    presentation = _event_slice_presentation()
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "bed.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([0.75, -0.25], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(int(0.5 * 44100), 2)
    assert engine.mixer.get_layer("__ez_route__bed") is not None
    assert engine.mixer.get_layer("__ez_route__kick_lane") is not None
    np.testing.assert_array_almost_equal(mixed, np.array([1.0, 0.0], dtype=np.float32))
    controller.shutdown()


def test_runtime_controller_switches_playback_target_without_stopping_transport():
    base = replace(
        _event_slice_presentation(),
        selected_layer_id=LayerId("bed"),
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
    before = engine.mixer.read_mix(int(0.5 * 44100), 2)
    controller.apply_mix_state(
        replace(
            base,
            selected_layer_id=LayerId("kick_lane"),
        )
    )
    after = engine.mixer.read_mix(int(0.5 * 44100), 2)

    assert controller.is_playing() is True
    assert engine.mixer.get_layer("__ez_route__bed") is not None
    assert engine.mixer.get_layer("__ez_route__kick_lane") is not None
    np.testing.assert_array_almost_equal(before, np.array([1.0, 0.75], dtype=np.float32))
    np.testing.assert_array_almost_equal(after, before)
    controller.shutdown()


def test_runtime_controller_mute_and_solo_controls_update_effective_mix_without_rebuild():
    base = _event_slice_presentation()
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "bed.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([1.0, 0.5], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(base)
    mixed_default = engine.mixer.read_mix(int(0.5 * 44100), 32)

    muted_bed = replace(
        base,
        layers=[
            replace(base.layers[0], muted=True),
            base.layers[1],
        ],
    )
    controller.apply_mix_state(muted_bed)
    _ = engine.mixer.read_mix(int(0.5 * 44100), 1024)
    mixed_bed_muted = engine.mixer.read_mix(int(0.5 * 44100), 32)

    soloed_kick = replace(
        base,
        layers=[
            base.layers[0],
            replace(base.layers[1], soloed=True),
        ],
    )
    controller.apply_mix_state(soloed_kick)
    _ = engine.mixer.read_mix(int(0.5 * 44100), 1024)
    mixed_kick_solo = engine.mixer.read_mix(int(0.5 * 44100), 32)

    assert controller._last_track_sync_reason == "mix-state-applied"
    np.testing.assert_allclose(
        mixed_default[:2],
        np.array([1.0, 0.75], dtype=np.float32),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        mixed_bed_muted[:2],
        np.array([1.0, 0.5], dtype=np.float32),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        mixed_kick_solo[:2],
        np.array([1.0, 0.5], dtype=np.float32),
        atol=1e-4,
    )
    controller.shutdown()


def test_runtime_controller_solo_monitor_overrides_muted_generated_audio_layer():
    base = build_demo_app().presentation()
    layers = [
        LayerPresentation(
            layer_id=LayerId("song_layer"),
            title="Song",
            kind=LayerKind.AUDIO,
            source_audio_path="song.wav",
        ),
        LayerPresentation(
            layer_id=LayerId("stem_drums"),
            title="Drums",
            kind=LayerKind.AUDIO,
            source_audio_path="drums.wav",
            muted=True,
            soloed=True,
        ),
    ]
    presentation = replace(
        base,
        layers=layers,
        selected_layer_id=LayerId("stem_drums"),
    )
    engine = AudioEngine(stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "song.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "drums.wav":
            return np.full(44100, 0.75, dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    mixed = engine.mixer.read_mix(0, 32)

    routed_drums = engine.mixer.get_layer("__ez_route__stem_drums")
    assert routed_drums is not None
    assert routed_drums.muted is False
    np.testing.assert_allclose(mixed[:2], np.array([0.75, 0.75], dtype=np.float32))
    controller.shutdown()


def test_runtime_controller_route_change_applies_immediately_during_mix_sync():
    base = _event_slice_presentation()
    presentation = replace(
        base,
        layers=[
            replace(base.layers[0], output_bus="outputs_1_2"),
            base.layers[1],
        ],
    )
    engine = AudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory)

    def _loader(path: str):
        if path == "bed.wav":
            return np.full(44100, 0.25, dtype=np.float32), 44100
        if path == "kick.wav":
            return np.array([1.0, 0.5], dtype=np.float32), 44100
        raise AssertionError(path)

    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    controller.build_for_presentation(presentation)

    rerouted = replace(
        presentation,
        layers=[
            replace(presentation.layers[0], output_bus="outputs_3_4"),
            presentation.layers[1],
        ],
    )
    controller.apply_mix_state(rerouted)

    assert controller._last_track_sync_reason == "mix-state-applied"
    routed = engine.mixer.get_layer("__ez_route__bed")
    assert routed is not None
    assert routed.output_bus == "outputs_3_4"
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
        selected_layer_id=LayerId("stems"),
        selected_take_id=alt_take.take_id,
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

    engine_layer = engine.mixer.get_layer(TimelineRuntimeAudioController._PRIMARY_TRACK_ID)
    assert engine_layer is not None
    np.testing.assert_array_almost_equal(
        engine_layer.buffer[:2], np.array([0.8, -0.4], dtype=np.float32)
    )
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


def test_demo_dispatch_routes_mix_update_intents_to_runtime_audio():
    demo = build_demo_app()
    runtime_audio = RecordingRuntimeAudio()
    demo.runtime_audio = runtime_audio
    layer_id = demo.presentation().layers[0].layer_id

    demo.dispatch(SetLayerMute(layer_id=layer_id, muted=True))

    assert runtime_audio.calls == [("mix", None)]


def test_runtime_controller_structural_sync_queues_async_while_playing(monkeypatch):
    base = _event_slice_presentation()
    changed = replace(
        base,
        layers=[
            base.layers[0],
            replace(
                base.layers[1],
                events=[
                    *base.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_3"),
                        start=2.0,
                        end=2.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
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

    started = threading.Event()
    release = threading.Event()
    original_prepare = controller._prepare_structure_track_plan_async

    def _blocking_prepare(presentation):
        started.set()
        assert release.wait(timeout=2.0)
        return original_prepare(presentation)

    monkeypatch.setattr(controller, "_prepare_structure_track_plan_async", _blocking_prepare)
    before_track = engine.mixer.get_layer("__ez_route__kick_lane")
    assert before_track is not None
    before_duration = before_track.duration_samples

    controller.sync_structure_state(changed)
    assert started.wait(timeout=1.0) is True
    assert controller._last_track_sync_reason == "structure-async-queued"
    mid_track = engine.mixer.get_layer("__ez_route__kick_lane")
    assert mid_track is not None
    assert mid_track.duration_samples == before_duration

    release.set()
    deadline = time.monotonic() + 2.0
    while controller._pending_structure_futures and time.monotonic() < deadline:
        controller.drain_pending_structure_sync()
        time.sleep(0.01)

    after_track = engine.mixer.get_layer("__ez_route__kick_lane")
    assert after_track is not None
    assert after_track.duration_samples > before_duration
    assert controller._last_track_sync_reason == "structure-async-applied"
    controller.shutdown()


def test_runtime_controller_structural_sync_does_not_relatch_transport_while_playing(monkeypatch):
    base = _event_slice_presentation()
    changed = replace(
        base,
        layers=[
            base.layers[0],
            replace(
                base.layers[1],
                events=[
                    *base.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_3"),
                        start=2.0,
                        end=2.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
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
    controller.seek(1.25)

    seek_calls = 0
    play_calls = 0
    original_seek = engine.seek_seconds
    original_play = engine.play

    def _counted_seek(seconds: float) -> None:
        nonlocal seek_calls
        seek_calls += 1
        original_seek(seconds)

    def _counted_play() -> None:
        nonlocal play_calls
        play_calls += 1
        original_play()

    monkeypatch.setattr(
        AudioEngine,
        "seek_seconds",
        lambda self, seconds: _counted_seek(seconds),
    )
    monkeypatch.setattr(
        AudioEngine,
        "play",
        lambda self: _counted_play(),
    )

    controller.sync_structure_state(changed)
    deadline = time.monotonic() + 2.0
    while controller._pending_structure_futures and time.monotonic() < deadline:
        controller.drain_pending_structure_sync()
        time.sleep(0.01)

    assert seek_calls == 0
    assert play_calls == 0
    assert controller.is_playing() is True
    assert engine.transport.is_playing is True
    assert engine.clock.position_seconds >= 1.25
    controller.shutdown()


def test_runtime_controller_structural_sync_latest_wins_and_drops_stale(monkeypatch):
    base = _event_slice_presentation()
    changed_v1 = replace(
        base,
        layers=[
            base.layers[0],
            replace(
                base.layers[1],
                events=[
                    *base.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_3"),
                        start=2.0,
                        end=2.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
    )
    changed_v2 = replace(
        changed_v1,
        layers=[
            changed_v1.layers[0],
            replace(
                changed_v1.layers[1],
                events=[
                    *changed_v1.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_4"),
                        start=3.0,
                        end=3.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
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

    first_started = threading.Event()
    release_first = threading.Event()
    original_prepare = controller._prepare_structure_track_plan_async

    def _prepare_with_first_block(presentation):
        event_count = len(presentation.layers[1].events)
        if event_count == 3:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return original_prepare(presentation)

    monkeypatch.setattr(
        controller, "_prepare_structure_track_plan_async", _prepare_with_first_block
    )

    controller.sync_structure_state(changed_v1)
    assert first_started.wait(timeout=1.0) is True
    controller.sync_structure_state(changed_v2)
    release_first.set()

    deadline = time.monotonic() + 2.0
    while controller._pending_structure_futures and time.monotonic() < deadline:
        controller.drain_pending_structure_sync()
        time.sleep(0.01)

    assert controller._latest_ready_generation == controller._latest_requested_generation
    assert controller._coalesced_edit_count >= 1

    expected_engine = AudioEngine(stream_factory=_fake_stream_factory)
    expected = TimelineRuntimeAudioController(engine=expected_engine, audio_loader=_loader)
    expected.build_for_presentation(changed_v2)
    expected_track = expected_engine.mixer.get_layer("__ez_route__kick_lane")
    applied_track = engine.mixer.get_layer("__ez_route__kick_lane")
    assert expected_track is not None
    assert applied_track is not None
    assert applied_track.duration_samples == expected_track.duration_samples

    expected.shutdown()
    controller.shutdown()


def test_runtime_controller_structural_sync_is_immediate_when_not_playing():
    base = _event_slice_presentation()
    changed = replace(
        base,
        layers=[
            base.layers[0],
            replace(
                base.layers[1],
                events=[
                    *base.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_3"),
                        start=2.0,
                        end=2.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
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
    before_track = engine.mixer.get_layer("__ez_route__kick_lane")
    assert before_track is not None
    before_duration = before_track.duration_samples

    controller.sync_structure_state(changed)

    after_track = engine.mixer.get_layer("__ez_route__kick_lane")
    assert after_track is not None
    assert after_track.duration_samples > before_duration
    assert controller._pending_structure_futures == {}
    controller.shutdown()


def test_runtime_controller_shutdown_cancels_async_render_jobs(monkeypatch):
    base = _event_slice_presentation()
    changed = replace(
        base,
        layers=[
            base.layers[0],
            replace(
                base.layers[1],
                events=[
                    *base.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_3"),
                        start=2.0,
                        end=2.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
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

    started = threading.Event()
    release = threading.Event()
    original_prepare = controller._prepare_structure_track_plan_async

    def _blocking_prepare(presentation):
        started.set()
        assert release.wait(timeout=2.0)
        return original_prepare(presentation)

    monkeypatch.setattr(controller, "_prepare_structure_track_plan_async", _blocking_prepare)
    controller.sync_structure_state(changed)
    assert started.wait(timeout=1.0) is True

    controller.shutdown()
    release.set()
    assert controller._pending_structure_futures == {}


def test_runtime_controller_structural_storm_queues_without_blocking_and_keeps_glitch_count_flat(
    monkeypatch,
):
    base = _event_slice_presentation()
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

    original_prepare = controller._prepare_structure_track_plan_async

    def _slow_prepare(presentation):
        time.sleep(0.12)
        return original_prepare(presentation)

    monkeypatch.setattr(controller, "_prepare_structure_track_plan_async", _slow_prepare)
    max_call_ms = 0.0
    for index in range(4):
        changed = replace(
            base,
            layers=[
                base.layers[0],
                replace(
                    base.layers[1],
                    events=[
                        *base.layers[1].events,
                        EventPresentation(
                            event_id=EventId(f"storm_{index}"),
                            start=2.0 + (index * 0.25),
                            end=2.1 + (index * 0.25),
                            label="Kick",
                        ),
                    ],
                ),
            ],
        )
        started = time.perf_counter()
        controller.sync_structure_state(changed)
        max_call_ms = max(max_call_ms, (time.perf_counter() - started) * 1000.0)

    deadline = time.monotonic() + 3.0
    while controller._pending_structure_futures and time.monotonic() < deadline:
        controller.drain_pending_structure_sync()
        time.sleep(0.01)

    assert max_call_ms < 50.0
    assert engine.glitch_count == 0
    controller.shutdown()


def test_runtime_controller_shutdown_is_idempotent():
    controller = TimelineRuntimeAudioController(
        engine=AudioEngine(stream_factory=_fake_stream_factory),
        audio_loader=lambda _path: (np.ones(512, dtype=np.float32), 44100),
    )
    presentation = _audio_presentation()
    controller.build_for_presentation(presentation)

    controller.shutdown()
    controller.shutdown()

    assert controller._pending_structure_futures == {}
    assert controller._shutdown_state == "shutdown"


def test_runtime_controller_ignores_structure_queue_after_shutdown():
    base = _event_slice_presentation()
    changed = replace(
        base,
        layers=[
            base.layers[0],
            replace(
                base.layers[1],
                events=[
                    *base.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_3"),
                        start=2.0,
                        end=2.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
    )
    controller = TimelineRuntimeAudioController(
        engine=AudioEngine(stream_factory=_fake_stream_factory),
        audio_loader=lambda _path: (np.ones(512, dtype=np.float32), 44100),
    )
    controller.build_for_presentation(base)
    controller.shutdown()

    controller.sync_structure_state(changed)
    controller.drain_pending_structure_sync()

    assert controller._pending_structure_futures == {}
    assert controller._last_track_sync_reason == "structure-async-shutdown-ignored"


def test_runtime_controller_drain_after_shutdown_never_applies_completed_results(monkeypatch):
    base = _event_slice_presentation()
    changed = replace(
        base,
        layers=[
            base.layers[0],
            replace(
                base.layers[1],
                events=[
                    *base.layers[1].events,
                    EventPresentation(
                        event_id=EventId("kick_3"),
                        start=2.0,
                        end=2.1,
                        label="Kick",
                    ),
                ],
            ),
        ],
    )
    controller = TimelineRuntimeAudioController(
        engine=AudioEngine(stream_factory=_fake_stream_factory),
        audio_loader=lambda _path: (np.ones(512, dtype=np.float32), 44100),
    )
    controller.build_for_presentation(base)
    controller.play()

    started = threading.Event()
    release = threading.Event()
    original_prepare = controller._prepare_structure_track_plan_async

    def _blocking_prepare(presentation):
        started.set()
        assert release.wait(timeout=2.0)
        return original_prepare(presentation)

    monkeypatch.setattr(controller, "_prepare_structure_track_plan_async", _blocking_prepare)
    controller.sync_structure_state(changed)
    assert started.wait(timeout=1.0) is True

    controller.shutdown()
    release.set()
    controller.drain_pending_structure_sync()

    assert controller._pending_structure_futures == {}
    assert all(
        outcome in {"cancelled", "failed", "stale-dropped", "applied"}
        for outcome in controller._generation_terminal_outcomes.values()
    )


__all__ = [name for name in globals() if name.startswith("test_")]
