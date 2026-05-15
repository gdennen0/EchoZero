"""Audio declick and timecode seam proof cases.
Exists to pin transition-safe playback behavior without real audio hardware.
Connects synthetic callback output to runtime diagnostics and display authority contracts.
"""

from __future__ import annotations

import numpy as np

from echozero.audio.engine import AudioEngine, _declick_ramp_samples
from echozero.audio.layer import AudioLayer
from tests.audio_engine_shared_support import fake_stream_factory

_SAMPLE_RATE = 48000
_FRAMES = 128
_DELTA_LIMIT = 0.18


def _callback(engine: AudioEngine, frames: int = _FRAMES) -> np.ndarray:
    out = np.zeros((frames, engine.output_channels), dtype=np.float32)
    engine._audio_callback(out, frames, None, None)
    return out.copy()


def _max_delta(*chunks: np.ndarray) -> float:
    joined = np.concatenate(chunks, axis=0)
    if joined.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(joined, axis=0))))


def _constant_track(engine: AudioEngine, track_id: str = "bed", value: float = 1.0) -> AudioLayer:
    samples = np.full(_SAMPLE_RATE, value, dtype=np.float32)
    return engine.create_track(track_id, samples, _SAMPLE_RATE, output_bus="outputs_1_2")


def test_transport_transitions_are_declick_safe_for_synthetic_step_waveform() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks([_constant_track(engine)])

    engine.play()
    play_chunk = _callback(engine)
    engine.pause()
    pause_chunk = _callback(engine)
    engine.play()
    resume_chunk = _callback(engine)
    engine.stop()
    stop_chunk = _callback(engine)

    assert _max_delta(play_chunk) <= _DELTA_LIMIT
    assert _max_delta(pause_chunk) <= _DELTA_LIMIT
    assert _max_delta(resume_chunk) <= _DELTA_LIMIT
    assert _max_delta(stop_chunk) <= _DELTA_LIMIT
    assert engine.last_ramp_reason in {"play", "pause", "stop", "overlay-stop"}
    engine.shutdown()


def test_toggle_play_pause_uses_transport_declick_path() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks([_constant_track(engine)])

    engine.toggle_play_pause()
    assert engine.last_discontinuity_reason == "toggle-play"
    play_a = _callback(engine)
    play_b = _callback(engine)

    engine.toggle_play_pause()
    assert engine.last_discontinuity_reason == "toggle-pause"
    pause_a = _callback(engine)
    pause_b = _callback(engine)

    assert _max_delta(play_a, play_b) <= _DELTA_LIMIT
    assert _max_delta(pause_a, pause_b) <= _DELTA_LIMIT
    assert engine.last_ramp_reason == "toggle-pause"
    engine.shutdown()


def test_callback_declicks_pause_state_even_if_request_races_callback() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks([_constant_track(engine)])

    try:
        engine.play()
        previous = _callback(engine)
        while engine.ramp_samples_remaining > 0:
            previous = _callback(engine)

        engine.transport.pause()
        paused = _callback(engine)

        assert _max_delta(previous, paused) <= _DELTA_LIMIT
        assert engine.last_ramp_reason == "transport-state-changed"
    finally:
        engine.shutdown()


def test_callback_declicks_play_state_even_if_pending_request_was_consumed_early() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks([_constant_track(engine)])

    try:
        _callback(engine)
        engine.request_declick()
        early_silence = _callback(engine)
        engine.transport.play()
        resumed = _callback(engine)

        assert _max_delta(early_silence, resumed) <= _DELTA_LIMIT
        assert engine.last_ramp_reason == "transport-state-changed"
    finally:
        engine.shutdown()


def test_paused_mute_declicks_from_last_audible_tail_to_silence() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks([_constant_track(engine)])

    try:
        engine.play()
        previous = _callback(engine)
        while engine.ramp_samples_remaining > 0:
            previous = _callback(engine)

        engine.pause()
        engine.apply_track_mix_updates({"bed": (True, 1.0, None)})
        engine.play()
        resumed = _callback(engine)
        settled = _callback(engine)

        assert _max_delta(previous[-1:], resumed, settled) <= _DELTA_LIMIT
        assert float(np.max(np.abs(settled[-32:]))) <= 1e-5
    finally:
        engine.shutdown()


def test_seek_while_playing_is_declick_safe_for_discontinuous_waveform() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    samples = np.concatenate(
        (
            np.ones(_SAMPLE_RATE // 2, dtype=np.float32),
            -np.ones(_SAMPLE_RATE // 2, dtype=np.float32),
        )
    )
    engine.replace_tracks([engine.create_track("bed", samples, _SAMPLE_RATE)])
    engine.play()
    before = _callback(engine)

    engine.seek(_SAMPLE_RATE // 2)
    after = _callback(engine)

    assert _max_delta(before, after) <= _DELTA_LIMIT
    assert engine.last_discontinuity_reason == "seek"
    assert any(
        event.get("kind") == "callback-discontinuity" and event.get("reason") == "seek"
        for event in engine.recent_runtime_events
    )
    engine.shutdown()


def test_overlay_start_and_end_are_declick_safe() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks(
        [engine.create_track("bed", np.zeros(_SAMPLE_RATE, dtype=np.float32), _SAMPLE_RATE)]
    )
    engine.play()
    _ = _callback(engine)

    overlay = np.ones(_FRAMES * 2, dtype=np.float32)
    original_overlay = overlay.copy()
    assert engine.play_overlay(overlay, _SAMPLE_RATE)
    first = _callback(engine)
    second = _callback(engine)

    np.testing.assert_array_equal(overlay, original_overlay)
    assert _max_delta(first, second) <= _DELTA_LIMIT
    assert engine.last_ramp_reason in {"overlay-start", "overlay-end", "overlay-stop"}
    engine.shutdown()


def test_overlay_preview_fades_do_not_mutate_source_views() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    source = np.linspace(-1.0, 1.0, _FRAMES * 4, dtype=np.float32)
    original_source = source.copy()
    overlay_view = source[_FRAMES : _FRAMES * 3]

    assert engine.play_overlay(overlay_view, _SAMPLE_RATE)
    _ = _callback(engine)
    _ = _callback(engine)

    np.testing.assert_array_equal(source, original_source)
    staged_overlay = getattr(engine, "_overlay_buffer", None)
    assert staged_overlay is None or not np.shares_memory(staged_overlay, source)
    engine.shutdown()


def test_short_overlay_preview_fade_reaches_silence_at_clip_edges() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    overlay = np.ones(32, dtype=np.float32)

    assert engine.play_overlay(overlay, _SAMPLE_RATE)

    staged = getattr(engine, "_overlay_playback_buffer", None)
    assert staged is not None
    assert float(staged[0]) == 0.0
    assert float(staged[-1]) == 0.0
    assert float(np.max(staged)) > 0.9
    np.testing.assert_array_equal(overlay, np.ones(32, dtype=np.float32))
    engine.shutdown()


def test_rapid_overlay_replacement_hands_off_without_hard_discontinuity() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks(
        [engine.create_track("bed", np.zeros(_SAMPLE_RATE, dtype=np.float32), _SAMPLE_RATE)]
    )
    engine.play()
    _ = _callback(engine)
    first_overlay = np.ones(_FRAMES * 4, dtype=np.float32)
    second_overlay = -np.ones(_FRAMES * 4, dtype=np.float32)
    original_first = first_overlay.copy()
    original_second = second_overlay.copy()

    assert engine.play_overlay(first_overlay, _SAMPLE_RATE)
    first = _callback(engine)
    assert engine.play_overlay(second_overlay, _SAMPLE_RATE)
    replaced = _callback(engine)
    continued = _callback(engine)

    np.testing.assert_array_equal(first_overlay, original_first)
    np.testing.assert_array_equal(second_overlay, original_second)
    assert _max_delta(first, replaced, continued) <= _DELTA_LIMIT
    assert engine.last_discontinuity_reason == "overlay-start"
    event_kinds = {str(event.get("kind")) for event in engine.recent_runtime_events}
    assert "overlay-start" in event_kinds
    assert "overlay-replace" in event_kinds
    assert "overlay-release" in event_kinds
    engine.shutdown()


def test_stop_overlay_hands_off_active_nonzero_preview_without_hard_discontinuity() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    overlay = np.ones(_FRAMES * 4, dtype=np.float32)

    assert engine.play_overlay(overlay, _SAMPLE_RATE)
    active = _callback(engine)
    engine.stop_overlay()
    stopped = _callback(engine)
    after = _callback(engine)

    assert _max_delta(active, stopped, after) <= _DELTA_LIMIT
    assert engine.overlay_active is False
    assert engine.last_discontinuity_reason == "overlay-stop"
    engine.shutdown()


def test_natural_end_of_content_fades_to_silence_without_hard_discontinuity() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks(
        [engine.create_track("short", np.ones(_FRAMES * 2, dtype=np.float32), _SAMPLE_RATE)]
    )

    engine.play()
    first = _callback(engine)
    ending = _callback(engine)
    after = _callback(engine)

    assert engine.reached_end is True
    assert engine.transport.is_playing is False
    assert _max_delta(first, ending, after) <= _DELTA_LIMIT
    assert engine.last_ramp_reason == "end-of-content"
    engine.shutdown()


def test_mix_mute_unmute_and_route_changes_are_declick_safe_without_replacing_tracks() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=4, stream_factory=fake_stream_factory)
    track = _constant_track(engine, value=1.0)
    engine.replace_tracks([track])
    engine.play()
    before = _callback(engine)

    assert engine.apply_track_mix_updates({"bed": (True, 1.0, "outputs_1_2")}) is True
    muted = _callback(engine)
    assert engine.apply_track_mix_updates({"bed": (False, 1.0, "outputs_1_2")}) is True
    unmuted = _callback(engine)
    assert engine.apply_track_mix_updates({"bed": (False, 1.0, "outputs_3_4")}) is True
    routed = _callback(engine)

    assert engine.tracks[0] is track
    assert _max_delta(before, muted, unmuted, routed) <= _DELTA_LIMIT
    assert engine.last_discontinuity_reason == "mix-update"
    engine.shutdown()


def test_loop_wrap_crossfade_is_declick_safe() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    phase = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    loop = (0.25 * np.sin(phase)).astype(np.float32)
    engine.replace_tracks([engine.create_track("loop", loop, _SAMPLE_RATE)])
    engine.clock.set_loop(0, 256)
    engine.clock.loop_enabled = True
    engine.seek(192)
    engine.play()

    wrapped = _callback(engine)

    assert engine.clock.position < 256
    assert _max_delta(wrapped) <= _DELTA_LIMIT
    engine.shutdown()


def test_callback_declick_uses_precomputed_ramps_after_warmup(monkeypatch) -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks([_constant_track(engine)])
    engine.play()
    _ = _callback(engine)

    def _fail_linspace(*_args, **_kwargs):
        raise AssertionError("callback allocated a new ramp")

    monkeypatch.setattr("echozero.audio.engine.np.linspace", _fail_linspace)
    engine.pause()
    _ = _callback(engine)
    engine.play()
    _ = _callback(engine)

    assert engine.ramp_samples_remaining >= 0
    engine.shutdown()


def test_mono_stereo_and_multichannel_output_mapping_is_correct() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=4, stream_factory=fake_stream_factory)
    mono = engine.create_track(
        "mono", np.ones(1024, dtype=np.float32), _SAMPLE_RATE, output_bus="outputs_1_2"
    )
    stereo = engine.create_track(
        "stereo",
        np.column_stack(
            (np.full(1024, 0.5, dtype=np.float32), np.full(1024, -0.5, dtype=np.float32))
        ),
        _SAMPLE_RATE,
        output_bus="outputs_3_4",
    )
    engine.replace_tracks([mono, stereo])
    engine.play()

    out = _callback(engine, frames=4)

    ramp_denominator = _declick_ramp_samples(_SAMPLE_RATE) - 1
    np.testing.assert_allclose(
        out[:, 0],
        np.array(
            [0.0, 1 / ramp_denominator, 2 / ramp_denominator, 3 / ramp_denominator],
            dtype=np.float32,
        ),
        atol=1e-4,
    )
    np.testing.assert_allclose(out[:, 1], out[:, 0], atol=1e-4)
    np.testing.assert_allclose(
        out[:, 2],
        np.array(
            [0.0, 0.5 / ramp_denominator, 1.0 / ramp_denominator, 1.5 / ramp_denominator],
            dtype=np.float32,
        ),
        atol=1e-4,
    )
    np.testing.assert_allclose(out[:, 3], -out[:, 2], atol=1e-4)
    engine.shutdown()
