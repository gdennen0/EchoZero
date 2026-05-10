"""Audio declick and timecode seam proof cases.
Exists to pin transition-safe playback behavior without real audio hardware.
Connects synthetic callback output to runtime diagnostics and display authority contracts.
"""

from __future__ import annotations

import numpy as np

from echozero.audio.engine import AudioEngine
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
    engine.shutdown()


def test_overlay_start_and_end_are_declick_safe() -> None:
    engine = AudioEngine(sample_rate=_SAMPLE_RATE, channels=1, stream_factory=fake_stream_factory)
    engine.replace_tracks(
        [engine.create_track("bed", np.zeros(_SAMPLE_RATE, dtype=np.float32), _SAMPLE_RATE)]
    )
    engine.play()
    _ = _callback(engine)

    overlay = np.ones(_FRAMES * 2, dtype=np.float32)
    assert engine.play_overlay(overlay, _SAMPLE_RATE)
    first = _callback(engine)
    second = _callback(engine)

    assert _max_delta(first) <= _DELTA_LIMIT
    assert _max_delta(second) <= _DELTA_LIMIT
    assert engine.last_ramp_reason in {"overlay-start", "overlay-end", "overlay-stop"}
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
        "mono", np.ones(4, dtype=np.float32), _SAMPLE_RATE, output_bus="outputs_1_2"
    )
    stereo = engine.create_track(
        "stereo",
        np.column_stack((np.full(4, 0.5, dtype=np.float32), np.full(4, -0.5, dtype=np.float32))),
        _SAMPLE_RATE,
        output_bus="outputs_3_4",
    )
    engine.replace_tracks([mono, stereo])
    engine.play()

    out = _callback(engine, frames=4)

    np.testing.assert_allclose(
        out[:, 0], np.array([0.0, 1 / 63, 2 / 63, 3 / 63], dtype=np.float32), atol=1e-4
    )
    np.testing.assert_allclose(out[:, 1], out[:, 0], atol=1e-4)
    np.testing.assert_allclose(
        out[:, 2], np.array([0.0, 0.5 / 63, 1.0 / 63, 1.5 / 63], dtype=np.float32), atol=1e-4
    )
    np.testing.assert_allclose(out[:, 3], -out[:, 2], atol=1e-4)
    engine.shutdown()
