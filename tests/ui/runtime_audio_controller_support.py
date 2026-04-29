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


def test_runtime_controller_force_qt_for_audio_blocks_sounddevice_fallback(monkeypatch):
    presentation = _audio_presentation()
    controller = TimelineRuntimeAudioController(
        audio_loader=lambda _path: (np.ones(4410, dtype=np.float32), 44100),
        use_qt_player=True,
        force_qt_for_continuous_audio=True,
    )
    monkeypatch.setattr(controller, "_ensure_qt_player", lambda: False)
    try:
        controller.build_for_presentation(presentation)
        controller.play()

        state = controller.snapshot_state(presentation)
        assert state.backend_name == "qt_multimedia"
        assert controller.engine.transport.is_playing is False
        assert (
            controller.engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)
            is None
        )
    finally:
        controller.shutdown()


def test_runtime_controller_qt_audio_routes_do_not_eager_decode_source(monkeypatch):
    presentation = _audio_presentation()
    load_calls: list[str] = []

    def _loader(path: str):
        load_calls.append(path)
        return np.ones(4410, dtype=np.float32), 44100

    controller = TimelineRuntimeAudioController(
        audio_loader=_loader,
        use_qt_player=True,
    )

    class _FakeQtPlayer:
        def setSource(self, _source):
            return None

        def play(self):
            return None

        def stop(self):
            return None

        def setPosition(self, _position_ms: int):
            return None

        def position(self) -> int:
            return 0

        def playbackState(self):
            return 0

    class _FakeQtAudioOutput:
        def setVolume(self, _volume: float):
            return None

        def device(self):
            return None

    controller._qt_player = _FakeQtPlayer()
    controller._qt_audio_output = _FakeQtAudioOutput()
    monkeypatch.setattr(controller, "_ensure_qt_player", lambda: True)
    try:
        signature = controller.presentation_signature(presentation)
        controller.build_for_presentation(presentation)

        assert signature == (("runtime_audio", "audio:demo.wav"),)
        assert load_calls == []
    finally:
        controller.shutdown()


def test_runtime_controller_can_prefer_sounddevice_backend_for_audio_layers():
    presentation = _audio_presentation()
    controller = TimelineRuntimeAudioController(
        audio_loader=lambda _path: (np.ones(4410, dtype=np.float32), 44100),
        use_qt_player=True,
        prefer_qt_for_continuous_audio=False,
    )
    try:
        controller.build_for_presentation(presentation)
        state = controller.snapshot_state(presentation)

        assert state.backend_name == "sounddevice"
        assert (
            controller.engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)
            is not None
        )
    finally:
        controller.shutdown()


def test_runtime_controller_qt_route_switch_fades_before_rebinding_source(monkeypatch):
    app = QApplication.instance() or QApplication([])
    from PyQt6.QtMultimedia import QMediaPlayer

    base = _audio_presentation()
    alt_layer = LayerPresentation(
        layer_id=LayerId("alt_audio"),
        title="Alt Audio",
        kind=LayerKind.AUDIO,
        source_audio_path="alt.wav",
    )
    switched = replace(
        base,
        layers=[base.layers[0], alt_layer],
        active_playback_layer_id=alt_layer.layer_id,
    )
    events: list[tuple[str, object]] = []

    class _FakeQtPlayer:
        def __init__(self):
            self._position = 0
            self._state = QMediaPlayer.PlaybackState.StoppedState

        def setSource(self, source):
            events.append(("source", source.toLocalFile()))

        def play(self):
            self._state = QMediaPlayer.PlaybackState.PlayingState
            events.append(("play", None))

        def stop(self):
            self._state = QMediaPlayer.PlaybackState.StoppedState
            events.append(("stop", None))

        def setPosition(self, position_ms: int):
            self._position = position_ms
            events.append(("seek", position_ms))

        def position(self) -> int:
            return self._position

        def playbackState(self):
            return self._state

    class _FakeQtAudioOutput:
        def __init__(self):
            self._volume = 1.0

        def setVolume(self, volume: float):
            self._volume = float(volume)
            events.append(("volume", round(self._volume, 3)))

        def volume(self) -> float:
            return self._volume

        def device(self):
            return None

    controller = TimelineRuntimeAudioController(
        audio_loader=lambda _path: (np.ones(4410, dtype=np.float32), 44100),
        use_qt_player=True,
    )
    controller._qt_player = _FakeQtPlayer()
    controller._qt_audio_output = _FakeQtAudioOutput()
    controller._qt_transition_interval_ms = 0
    controller._qt_transition_steps = 2
    monkeypatch.setattr(controller, "_ensure_qt_player", lambda: True)
    try:
        controller.build_for_presentation(base)
        controller.play()
        events.clear()

        controller.apply_mix_state(switched)

        assert ("source", "alt.wav") not in events

        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
            if ("source", "alt.wav") in events:
                source_index = events.index(("source", "alt.wav"))
                if any(
                    event[0] == "volume" and isinstance(event[1], float) and event[1] > 0.0
                    for event in events[source_index + 1 :]
                ):
                    break

        assert ("source", "alt.wav") in events
        source_index = events.index(("source", "alt.wav"))
        assert any(event == ("volume", 0.0) for event in events[: source_index + 1])
        assert any(event == ("play", None) for event in events[source_index:])
        assert any(
            event[0] == "volume" and isinstance(event[1], float) and event[1] > 0.0
            for event in events[source_index + 1 :]
        )
    finally:
        controller.shutdown()
        app.processEvents()


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
        active_playback_layer_id=song_layer.layer_id,
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
    assert engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID) is None
    assert engine.mixer.get_layer("__ez_route__song_layer") is not None
    assert engine.mixer.get_layer("__ez_route__timecode_layer") is not None

    state = controller.snapshot_state(presentation)
    assert {source.layer_id for source in state.active_sources} == {
        "song_layer",
        "timecode_layer",
    }
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
        active_playback_layer_id=song_layer.layer_id,
        active_playback_take_id=alt_take.take_id,
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


def test_runtime_controller_preview_clip_tears_down_preview_stream_after_end():
    engine = AudioEngine(stream_factory=_fake_stream_factory)
    preview_engine = AudioEngine(sample_rate=10, stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        preview_engine=preview_engine,
        audio_loader=lambda _path: (np.arange(10, dtype=np.float32), 10),
    )

    played = controller.preview_clip("kick.wav", start_seconds=0.0, end_seconds=0.2)

    assert played is True
    outdata = np.zeros((256, 1), dtype=np.float32)
    preview_engine._audio_callback(outdata, 256, None, None)
    preview_engine._audio_callback(outdata, 256, None, None)

    controller.current_time_seconds()

    assert preview_engine.is_active is False
    assert (
        preview_engine.mixer.get_layer(TimelineRuntimeAudioController._PREVIEW_LAYER_ID)
        is None
    )
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
