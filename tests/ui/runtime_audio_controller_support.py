"""Controller-oriented runtime-audio support cases.
Exists to isolate controller and demo-dispatch coverage from widget timing support tests.
Connects the compatibility wrapper to the bounded runtime-audio controller slice.
"""

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

    engine_layer = engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)
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
    assert state.active_layer_id == presentation.active_playback_layer_id
    assert state.active_take_id is None
    assert len(state.active_sources) == 1
    assert state.active_sources[0].source_ref == "demo.wav"
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


def test_runtime_controller_preview_clip_plays_sliced_audio_on_preview_engine():
    engine = AudioEngine(stream_factory=_fake_stream_factory)
    preview_engine = AudioEngine(sample_rate=10, stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        preview_engine=preview_engine,
        audio_loader=lambda _path: (np.arange(10, dtype=np.float32), 10),
    )

    played = controller.preview_clip("kick.wav", start_seconds=0.2, end_seconds=0.6)

    assert played is True
    preview_layer = preview_engine.mixer.get_layer(
        TimelineRuntimeAudioController._PREVIEW_LAYER_ID
    )
    assert preview_layer is not None
    np.testing.assert_array_equal(
        preview_layer.buffer, np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    )
    assert preview_engine.transport.is_playing is True
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


def test_demo_dispatch_routes_playback_target_updates_runtime_audio():
    demo = build_demo_app()
    runtime_audio = RecordingRuntimeAudio()
    demo.runtime_audio = runtime_audio
    layer_id = demo.presentation().layers[0].layer_id

    demo.dispatch(SetActivePlaybackTarget(layer_id))

    assert runtime_audio.calls == [("mix", None)]



__all__ = [name for name in globals() if name.startswith("test_")]
