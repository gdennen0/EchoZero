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
    preview_engine = AudioEngine(sample_rate=10, stream_factory=_fake_stream_factory)
    controller = TimelineRuntimeAudioController(
        engine=engine,
        preview_engine=preview_engine,
        audio_loader=lambda _path: (np.arange(10, dtype=np.float32), 10),
    )

    played = controller.preview_clip("kick.wav", start_seconds=0.2, end_seconds=0.6)

    assert played is True
    preview_layer = preview_engine.mixer.get_layer(
        TimelineRuntimeAudioController._PREVIEW_TRACK_ID
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
        preview_engine.mixer.get_layer(TimelineRuntimeAudioController._PREVIEW_TRACK_ID)
        is None
    )
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
    mixed_default = engine.mixer.read_mix(int(0.5 * 44100), 2)

    muted_bed = replace(
        base,
        layers=[
            replace(base.layers[0], muted=True),
            base.layers[1],
        ],
    )
    controller.apply_mix_state(muted_bed)
    mixed_bed_muted = engine.mixer.read_mix(int(0.5 * 44100), 2)

    soloed_kick = replace(
        base,
        layers=[
            base.layers[0],
            replace(base.layers[1], soloed=True),
        ],
    )
    controller.apply_mix_state(soloed_kick)
    mixed_kick_solo = engine.mixer.read_mix(int(0.5 * 44100), 2)

    assert controller._last_track_sync_reason == "mix-state-applied"
    np.testing.assert_array_almost_equal(mixed_default, np.array([1.0, 0.75], dtype=np.float32))
    np.testing.assert_array_almost_equal(mixed_bed_muted, np.array([1.0, 0.5], dtype=np.float32))
    np.testing.assert_array_almost_equal(mixed_kick_solo, np.array([1.0, 0.5], dtype=np.float32))
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



__all__ = [name for name in globals() if name.startswith("test_")]
