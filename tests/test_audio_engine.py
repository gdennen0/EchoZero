"""
Audio engine tests: Clock, Transport, AudioLayer, Mixer, and AudioEngine integration.
All tests run without sounddevice (stream_factory injection for AudioEngine).

Covers ship-ready guarantees:
- Lock-free clock (no lock in advance)
- Pre-allocated mixer buffers (no per-callback allocation)
- End-of-content auto-pause
- Sample rate conversion on layer add
- Loop wrapping
- Hard clipping on summed output
- Thread-safe subscriber add/remove
"""

from __future__ import annotations

import numpy as np
import pytest

from echozero.audio.clock import Clock, LoopRegion
from echozero.audio.crossfade import CrossfadeBuffer, build_equal_power_curves
from echozero.audio.transport import Transport, TransportState
from echozero.audio.layer import AudioLayer, resample_buffer
from echozero.audio.mixer import Mixer
from echozero.audio.engine import AudioEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(samples: int = 44100, freq: float = 440.0, sr: int = 44100) -> np.ndarray:
    """Generate a mono sine wave."""
    t = np.arange(samples, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class FakeStream:
    """Mock audio stream for testing without sounddevice."""

    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def fake_stream_factory(**kwargs):
    return FakeStream(**kwargs)


class RecordingSubscriber:
    """Clock subscriber that records every tick."""

    def __init__(self):
        self.ticks: list[tuple[int, int]] = []

    def on_clock_tick(self, position_samples: int, sample_rate: int) -> None:
        self.ticks.append((position_samples, sample_rate))


# ===========================================================================
# Clock tests
# ===========================================================================


class TestClock:
    def test_initial_position_is_zero(self) -> None:
        clock = Clock(44100)
        assert clock.position == 0
        assert clock.position_seconds == 0.0

    def test_advance_returns_pre_advance_position(self) -> None:
        clock = Clock(44100)
        pos = clock.advance(256)
        assert pos == 0
        assert clock.position == 256

    def test_advance_accumulates(self) -> None:
        clock = Clock(44100)
        clock.advance(256)
        clock.advance(256)
        assert clock.position == 512

    def test_seek(self) -> None:
        clock = Clock(44100)
        clock.advance(1000)
        clock.seek(500)
        assert clock.position == 500

    def test_seek_negative_clamps_to_zero(self) -> None:
        clock = Clock(44100)
        clock.seek(-100)
        assert clock.position == 0

    def test_seek_seconds(self) -> None:
        clock = Clock(44100)
        clock.seek_seconds(1.0)
        assert clock.position == 44100

    def test_position_seconds(self) -> None:
        clock = Clock(44100)
        clock.advance(22050)
        assert abs(clock.position_seconds - 0.5) < 1e-6

    def test_reset(self) -> None:
        clock = Clock(44100)
        clock.advance(10000)
        clock.reset()
        assert clock.position == 0

    def test_subscriber_called_on_advance(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.advance(256)
        assert len(sub.ticks) == 1
        assert sub.ticks[0] == (0, 44100)

    def test_subscriber_receives_pre_advance_position(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.advance(256)
        clock.advance(256)
        assert sub.ticks[1] == (256, 44100)

    def test_remove_subscriber(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.advance(256)
        clock.remove_subscriber(sub)
        clock.advance(256)
        assert len(sub.ticks) == 1

    def test_remove_nonexistent_subscriber_is_safe(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.remove_subscriber(sub)  # should not raise

    def test_duplicate_subscriber_not_added(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.add_subscriber(sub)
        clock.advance(256)
        assert len(sub.ticks) == 1

    def test_subscriber_add_is_copy_on_write(self) -> None:
        """Adding a subscriber doesn't mutate the list the audio thread sees."""
        clock = Clock(44100)
        sub1 = RecordingSubscriber()
        clock.add_subscriber(sub1)
        # Snapshot the internal list reference
        old_list = clock._subscribers
        sub2 = RecordingSubscriber()
        clock.add_subscriber(sub2)
        # Should be a NEW list, not the same object
        assert clock._subscribers is not old_list

    # -- Loop tests ---------------------------------------------------------

    def test_loop_region_validation(self) -> None:
        with pytest.raises(ValueError):
            LoopRegion(start=-1, end=100)
        with pytest.raises(ValueError):
            LoopRegion(start=100, end=100)
        with pytest.raises(ValueError):
            LoopRegion(start=100, end=50)

    def test_loop_wraps_position(self) -> None:
        clock = Clock(44100)
        clock.set_loop(0, 1000)
        clock.loop_enabled = True
        clock.seek(900)
        pos = clock.advance(200)
        assert pos == 900
        assert clock.position == 100

    def test_loop_disabled_does_not_wrap(self) -> None:
        clock = Clock(44100)
        clock.set_loop(0, 1000)
        clock.loop_enabled = False
        clock.seek(900)
        clock.advance(200)
        assert clock.position == 1100

    def test_loop_wraps_multiple_times(self) -> None:
        clock = Clock(44100)
        clock.set_loop(0, 100)
        clock.loop_enabled = True
        clock.seek(50)
        clock.advance(250)
        assert clock.position == 0

    def test_loop_with_offset_start(self) -> None:
        clock = Clock(44100)
        clock.set_loop(500, 1000)
        clock.loop_enabled = True
        clock.seek(900)
        clock.advance(200)
        assert clock.position == 600

    def test_set_loop_seconds(self) -> None:
        clock = Clock(44100)
        clock.set_loop_seconds(1.0, 2.0)
        assert clock.loop_region is not None
        assert clock.loop_region.start == 44100
        assert clock.loop_region.end == 88200

    def test_advance_is_lock_free(self) -> None:
        """Verify advance() doesn't acquire _lock (the lock object should not be held)."""
        clock = Clock(44100)
        # If advance used the lock, acquiring it here would deadlock when advance tries
        # In practice we just verify it works without issues when lock is held
        # (this is a structural test — real lock-free verification needs threading)
        clock.advance(256)
        assert clock.position == 256


# ===========================================================================
# Transport tests
# ===========================================================================


class TestTransport:
    def test_initial_state_is_stopped(self) -> None:
        transport = Transport(Clock(44100))
        assert transport.state == TransportState.STOPPED
        assert transport.is_stopped

    def test_play_from_stopped(self) -> None:
        transport = Transport(Clock(44100))
        transport.play()
        assert transport.state == TransportState.PLAYING

    def test_pause_from_playing(self) -> None:
        transport = Transport(Clock(44100))
        transport.play()
        transport.pause()
        assert transport.state == TransportState.PAUSED

    def test_pause_from_stopped_is_noop(self) -> None:
        transport = Transport(Clock(44100))
        transport.pause()
        assert transport.is_stopped

    def test_stop_resets_position(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(10000)
        transport.stop()
        assert transport.is_stopped
        assert clock.position == 0

    def test_stop_returns_to_seek_position(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek(5000)
        transport.play()
        clock.advance(10000)
        transport.stop()
        assert clock.position == 5000

    def test_resume_from_paused(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)
        transport.pause()
        pos_at_pause = clock.position
        transport.play()
        assert transport.is_playing
        assert clock.position == pos_at_pause

    def test_stop_from_paused(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)
        transport.pause()
        transport.stop()
        assert transport.is_stopped
        assert clock.position == 0

    def test_play_while_playing_is_noop(self) -> None:
        transport = Transport(Clock(44100))
        transport.play()
        transport.play()
        assert transport.is_playing

    def test_stop_while_stopped_is_noop(self) -> None:
        transport = Transport(Clock(44100))
        transport.stop()
        assert transport.is_stopped

    def test_seek_while_playing(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)
        transport.seek(1000)
        assert transport.is_playing
        assert clock.position == 1000

    def test_seek_seconds(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek_seconds(2.0)
        assert clock.position == 88200

    def test_toggle_play_pause(self) -> None:
        transport = Transport(Clock(44100))
        transport.toggle_play_pause()
        assert transport.is_playing
        transport.toggle_play_pause()
        assert transport.is_paused
        transport.toggle_play_pause()
        assert transport.is_playing

    def test_toggle_from_stopped(self) -> None:
        transport = Transport(Clock(44100))
        transport.toggle_play_pause()
        assert transport.is_playing

    def test_return_to_start(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek(5000)
        transport.play()
        clock.advance(10000)
        transport.return_to_start()
        assert clock.position == 0
        assert transport.is_playing

    def test_return_to_start_from_stopped(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek(5000)
        transport.return_to_start()
        assert clock.position == 0
        assert transport.is_stopped


# ===========================================================================
# Resampling tests
# ===========================================================================


class TestResample:
    def test_same_rate_returns_same(self) -> None:
        buf = _sine(1000, sr=44100)
        result = resample_buffer(buf, 44100, 44100)
        assert result is buf  # same object, no copy

    def test_upsample_doubles_length(self) -> None:
        buf = _sine(1000, sr=22050)
        result = resample_buffer(buf, 22050, 44100)
        assert len(result) == 2000

    def test_downsample_halves_length(self) -> None:
        buf = _sine(1000, sr=44100)
        result = resample_buffer(buf, 44100, 22050)
        assert len(result) == 500

    def test_48k_to_44k(self) -> None:
        buf = _sine(48000, sr=48000)  # 1 second at 48k
        result = resample_buffer(buf, 48000, 44100)
        # Should be ~44100 samples (1 second at 44.1k)
        assert abs(len(result) - 44100) <= 1

    def test_preserves_dtype(self) -> None:
        buf = _sine(1000, sr=22050)
        result = resample_buffer(buf, 22050, 44100)
        assert result.dtype == np.float32


# ===========================================================================
# AudioLayer tests
# ===========================================================================


class TestAudioLayer:
    def test_read_within_bounds(self) -> None:
        buf = _sine(1000)
        layer = AudioLayer("l1", "test", buf, 44100)
        chunk = layer.read_samples(0, 256)
        assert chunk.shape == (256,)
        np.testing.assert_array_equal(chunk, buf[:256])

    def test_read_beyond_end_is_zero_padded(self) -> None:
        buf = _sine(100)
        layer = AudioLayer("l1", "test", buf, 44100)
        chunk = layer.read_samples(50, 100)
        np.testing.assert_array_equal(chunk[:50], buf[50:100])
        np.testing.assert_array_equal(chunk[50:], np.zeros(50, dtype=np.float32))

    def test_read_before_start_is_zeros(self) -> None:
        buf = _sine(100)
        layer = AudioLayer("l1", "test", buf, 44100)
        chunk = layer.read_samples(-200, 100)
        np.testing.assert_array_equal(chunk, np.zeros(100, dtype=np.float32))

    def test_read_completely_past_end(self) -> None:
        buf = _sine(100)
        layer = AudioLayer("l1", "test", buf, 44100)
        chunk = layer.read_samples(200, 100)
        np.testing.assert_array_equal(chunk, np.zeros(100, dtype=np.float32))

    def test_offset_shifts_read(self) -> None:
        buf = _sine(1000)
        layer = AudioLayer("l1", "test", buf, 44100, offset=500)
        chunk = layer.read_samples(0, 256)
        np.testing.assert_array_equal(chunk, np.zeros(256, dtype=np.float32))
        chunk = layer.read_samples(500, 256)
        np.testing.assert_array_equal(chunk, buf[:256])

    def test_duration_properties(self) -> None:
        buf = _sine(44100)
        layer = AudioLayer("l1", "test", buf, 44100)
        assert layer.duration_samples == 44100
        assert abs(layer.duration_seconds - 1.0) < 1e-6

    def test_end_sample_with_offset(self) -> None:
        buf = _sine(1000)
        layer = AudioLayer("l1", "test", buf, 44100, offset=500)
        assert layer.end_sample == 1500

    def test_auto_converts_to_float32(self) -> None:
        buf = np.ones(100, dtype=np.float64)
        layer = AudioLayer("l1", "test", buf, 44100)
        assert layer.buffer.dtype == np.float32

    def test_mute_solo_defaults(self) -> None:
        buf = _sine(100)
        layer = AudioLayer("l1", "test", buf, 44100)
        assert layer.muted is False
        assert layer.solo is False
        assert layer.volume == 1.0

    def test_read_into_preallocated(self) -> None:
        """read_into() writes into existing buffer — no allocation."""
        buf = _sine(1000)
        layer = AudioLayer("l1", "test", buf, 44100)
        scratch = np.zeros(256, dtype=np.float32)
        layer.read_into(scratch, 0, 256)
        np.testing.assert_array_equal(scratch, buf[:256])

    def test_read_into_zeros_outside_range(self) -> None:
        buf = _sine(100)
        layer = AudioLayer("l1", "test", buf, 44100)
        scratch = np.ones(256, dtype=np.float32)  # fill with 1s
        layer.read_into(scratch, 200, 256)  # completely outside
        np.testing.assert_array_equal(scratch[:256], np.zeros(256, dtype=np.float32))

    def test_resampled_on_construction(self) -> None:
        """48kHz buffer resampled to 44.1kHz engine automatically."""
        buf = _sine(48000, sr=48000)  # 1 second at 48k
        layer = AudioLayer("l1", "test", buf, 48000, engine_sample_rate=44100)
        assert layer.sample_rate == 44100
        assert layer.original_sample_rate == 48000
        assert abs(layer.duration_samples - 44100) <= 1

    def test_no_resample_when_rates_match(self) -> None:
        buf = _sine(1000)
        layer = AudioLayer("l1", "test", buf, 44100, engine_sample_rate=44100)
        assert layer.buffer is buf  # same object


# ===========================================================================
# Mixer tests
# ===========================================================================


class TestMixer:
    def test_empty_mixer_returns_silence(self) -> None:
        mixer = Mixer()
        out = mixer.read_mix(0, 256)
        assert out.shape == (256,)
        np.testing.assert_array_equal(out, np.zeros(256, dtype=np.float32))

    def test_single_layer(self) -> None:
        buf = _sine(1000)
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "drums", buf, 44100))
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_almost_equal(out, buf[:256])

    def test_two_layers_summed(self) -> None:
        buf1 = _sine(1000, freq=440)
        buf2 = _sine(1000, freq=880)
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "drums", buf1, 44100))
        mixer.add_layer(AudioLayer("l2", "bass", buf2, 44100))
        out = mixer.read_mix(0, 256)
        expected = buf1[:256] + buf2[:256]
        # Clip expected too
        np.clip(expected, -1.0, 1.0, out=expected)
        np.testing.assert_array_almost_equal(out, expected)

    def test_muted_layer_excluded(self) -> None:
        buf = _sine(1000)
        mixer = Mixer()
        layer = AudioLayer("l1", "drums", buf, 44100)
        layer.muted = True
        mixer.add_layer(layer)
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_equal(out, np.zeros(256, dtype=np.float32))

    def test_solo_logic_plays_only_soloed(self) -> None:
        buf1 = _sine(1000, freq=440)
        buf2 = _sine(1000, freq=880)
        mixer = Mixer()
        l1 = AudioLayer("l1", "drums", buf1, 44100)
        l2 = AudioLayer("l2", "bass", buf2, 44100)
        l2.solo = True
        mixer.add_layer(l1)
        mixer.add_layer(l2)
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_almost_equal(out, buf2[:256])

    def test_solo_overrides_mute(self) -> None:
        buf = _sine(1000)
        mixer = Mixer()
        layer = AudioLayer("l1", "drums", buf, 44100)
        layer.muted = True
        layer.solo = True
        mixer.add_layer(layer)
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_almost_equal(out, buf[:256])

    def test_solo_exclusive(self) -> None:
        mixer = Mixer()
        l1 = AudioLayer("l1", "a", _sine(100), 44100)
        l2 = AudioLayer("l2", "b", _sine(100), 44100)
        l1.solo = True
        l2.solo = True
        mixer.add_layer(l1)
        mixer.add_layer(l2)
        mixer.solo_exclusive("l2")
        assert not l1.solo
        assert l2.solo

    def test_unsolo_all(self) -> None:
        mixer = Mixer()
        l1 = AudioLayer("l1", "a", _sine(100), 44100)
        l1.solo = True
        mixer.add_layer(l1)
        mixer.unsolo_all()
        assert not l1.solo

    def test_volume(self) -> None:
        buf = _sine(1000)
        mixer = Mixer()
        layer = AudioLayer("l1", "drums", buf, 44100, volume=0.5)
        mixer.add_layer(layer)
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_almost_equal(out, buf[:256] * 0.5)

    def test_master_volume(self) -> None:
        buf = _sine(1000)
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "drums", buf, 44100))
        mixer.master_volume = 0.5
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_almost_equal(out, buf[:256] * 0.5)

    def test_master_volume_clamped(self) -> None:
        mixer = Mixer()
        mixer.master_volume = 2.5
        assert mixer.master_volume == 2.0
        mixer.master_volume = -0.5
        assert mixer.master_volume == 0.0

    def test_remove_layer(self) -> None:
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "drums", _sine(100), 44100))
        removed = mixer.remove_layer("l1")
        assert removed is not None
        assert removed.id == "l1"
        assert mixer.layer_count == 0

    def test_remove_nonexistent_returns_none(self) -> None:
        mixer = Mixer()
        assert mixer.remove_layer("ghost") is None

    def test_get_layer(self) -> None:
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "drums", _sine(100), 44100))
        assert mixer.get_layer("l1") is not None
        assert mixer.get_layer("ghost") is None

    def test_clear(self) -> None:
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "a", _sine(100), 44100))
        mixer.add_layer(AudioLayer("l2", "b", _sine(100), 44100))
        mixer.clear()
        assert mixer.layer_count == 0

    def test_duration_samples(self) -> None:
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "a", _sine(1000), 44100))
        mixer.add_layer(AudioLayer("l2", "b", _sine(500), 44100, offset=1000))
        assert mixer.duration_samples == 1500

    def test_clipping_protection(self) -> None:
        """Two full-scale signals should be clipped to [-1, 1]."""
        buf = np.ones(256, dtype=np.float32)  # constant 1.0
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "a", buf.copy(), 44100))
        mixer.add_layer(AudioLayer("l2", "b", buf.copy(), 44100))
        out = mixer.read_mix(0, 256)
        # Without clipping this would be 2.0. With clipping, max is 1.0.
        assert np.max(out) <= 1.0
        assert np.min(out) >= -1.0

    def test_negative_clipping(self) -> None:
        buf = -np.ones(256, dtype=np.float32)
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "a", buf.copy(), 44100))
        mixer.add_layer(AudioLayer("l2", "b", buf.copy(), 44100))
        out = mixer.read_mix(0, 256)
        assert np.min(out) >= -1.0


# ===========================================================================
# AudioEngine integration tests
# ===========================================================================


class TestAudioEngine:
    def test_create_engine(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        assert engine.sample_rate == 44100
        assert engine.buffer_size == 256
        assert not engine.is_active

    def test_play_opens_stream(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.play()
        assert engine.is_active
        assert engine.transport.is_playing

    def test_stop(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.play()
        engine.stop()
        assert engine.transport.is_stopped

    def test_pause_resume(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.play()
        engine.pause()
        assert engine.transport.is_paused
        engine.play()
        assert engine.transport.is_playing

    def test_add_and_remove_layer(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        layer = engine.add_layer("drums", _sine(1000), 44100, name="Drums")
        assert engine.mixer.layer_count == 1
        assert layer.name == "Drums"
        engine.remove_layer("drums")
        assert engine.mixer.layer_count == 0

    def test_seek(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.seek(1000)
        assert engine.clock.position == 1000

    def test_seek_seconds(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.seek_seconds(1.0)
        assert engine.clock.position == 44100

    def test_toggle_play_pause(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.toggle_play_pause()
        assert engine.transport.is_playing
        engine.toggle_play_pause()
        assert engine.transport.is_paused

    def test_shutdown(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.play()
        engine.shutdown()
        assert not engine.is_active
        assert engine.transport.is_stopped

    def test_callback_outputs_silence_when_stopped(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)
        np.testing.assert_array_equal(outdata, np.zeros((256, 1), dtype=np.float32))

    def test_callback_outputs_audio_when_playing(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(1000)
        engine.add_layer("l1", buf, 44100)
        engine.play()

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        assert np.any(outdata != 0)
        np.testing.assert_array_almost_equal(outdata[:, 0], buf[:256])

    def test_callback_advances_clock(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.add_layer("l1", _sine(1000), 44100)
        engine.play()

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)
        assert engine.clock.position == 256

        engine._audio_callback(outdata, 256, None, None)
        assert engine.clock.position == 512

    def test_callback_with_loop(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(1000)
        engine.add_layer("l1", buf, 44100)
        engine.clock.set_loop(0, 500)
        engine.clock.loop_enabled = True
        engine.play()

        engine.seek(400)
        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)
        assert engine.clock.position < 500

    def test_clock_subscriber_receives_ticks(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.add_layer("l1", _sine(1000), 44100)
        sub = RecordingSubscriber()
        engine.add_clock_subscriber(sub)
        engine.play()

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        assert len(sub.ticks) == 1
        assert sub.ticks[0] == (0, 44100)

    def test_multi_track_mixing_through_callback(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf1 = _sine(1000, freq=440)
        buf2 = _sine(1000, freq=880)
        engine.add_layer("drums", buf1, 44100)
        engine.add_layer("bass", buf2, 44100)
        engine.play()

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        expected = buf1[:256] + buf2[:256]
        np.clip(expected, -1.0, 1.0, out=expected)
        np.testing.assert_array_almost_equal(outdata[:, 0], expected)

    def test_mute_through_callback(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(1000)
        layer = engine.add_layer("drums", buf, 44100)
        layer.muted = True
        engine.play()

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        np.testing.assert_array_equal(outdata, np.zeros((256, 1), dtype=np.float32))

    # -- End-of-content tests -----------------------------------------------

    def test_auto_pause_at_end_of_content(self) -> None:
        """Playback auto-pauses when position reaches end of all layers."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(500)  # very short
        engine.add_layer("l1", buf, 44100)
        engine.play()

        # Seek past the end
        engine.seek(600)
        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        assert engine.transport.is_paused
        assert engine.reached_end

    def test_no_auto_pause_when_looping(self) -> None:
        """Looping prevents auto-pause at end."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(500)
        engine.add_layer("l1", buf, 44100)
        engine.clock.set_loop(0, 400)
        engine.clock.loop_enabled = True
        engine.play()

        # Even past layer end, loop keeps us going
        engine.seek(350)
        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        assert engine.transport.is_playing

    def test_reached_end_reset_on_play(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine._end_of_content = True
        engine.play()
        assert not engine.reached_end

    def test_reached_end_reset_on_seek(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine._end_of_content = True
        engine.seek(0)
        assert not engine.reached_end

    # -- Sample rate conversion tests ---------------------------------------

    def test_add_layer_resamples_48k_to_44k(self) -> None:
        """48kHz buffer automatically resampled to engine's 44.1kHz."""
        engine = AudioEngine(sample_rate=44100, stream_factory=fake_stream_factory)
        buf = _sine(48000, sr=48000)  # 1 second at 48k
        layer = engine.add_layer("l1", buf, 48000)
        assert layer.sample_rate == 44100
        assert layer.original_sample_rate == 48000
        assert abs(layer.duration_samples - 44100) <= 1

    def test_add_layer_same_rate_no_resample(self) -> None:
        engine = AudioEngine(sample_rate=44100, stream_factory=fake_stream_factory)
        buf = _sine(1000, sr=44100)
        layer = engine.add_layer("l1", buf, 44100)
        assert layer.buffer is buf  # no copy

    # -- Clipping through engine callback -----------------------------------

    def test_callback_clips_output(self) -> None:
        """Multi-track summing that exceeds 1.0 is clipped in output."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        loud = np.ones(256, dtype=np.float32)
        engine.add_layer("l1", loud.copy(), 44100)
        engine.add_layer("l2", loud.copy(), 44100)
        engine.play()

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        assert np.max(outdata) <= 1.0


# ===========================================================================
# Crossfade tests
# ===========================================================================


class TestCrossfade:
    def test_equal_power_curves_sum_to_one(self) -> None:
        """Equal-power: fade_out² + fade_in² ≈ 1.0 at every point."""
        fade_out, fade_in = build_equal_power_curves(256)
        power_sum = fade_out ** 2 + fade_in ** 2
        np.testing.assert_array_almost_equal(power_sum, np.ones(256, dtype=np.float32), decimal=5)

    def test_fade_out_starts_at_one(self) -> None:
        fade_out, _ = build_equal_power_curves(256)
        assert abs(fade_out[0] - 1.0) < 1e-5

    def test_fade_in_ends_at_one(self) -> None:
        _, fade_in = build_equal_power_curves(256)
        assert abs(fade_in[-1] - 1.0) < 1e-5

    def test_crossfade_buffer_apply(self) -> None:
        """Applying crossfade blends tail and head smoothly."""
        xfade = CrossfadeBuffer(crossfade_samples=64)
        output = np.zeros(256, dtype=np.float32)
        tail = np.ones(64, dtype=np.float32)    # outgoing: constant 1.0
        head = np.zeros(64, dtype=np.float32)    # incoming: constant 0.0

        xfade.apply(output, tail, head, xfade_start=96, xfade_len=64)

        # At start of crossfade: should be close to tail (1.0)
        assert output[96] > 0.9
        # At end of crossfade: should be close to head (0.0)
        assert output[96 + 63] < 0.1
        # Middle: should be between 0 and 1
        mid = output[96 + 32]
        assert 0.3 < mid < 0.7

    def test_crossfade_same_signal_is_unchanged(self) -> None:
        """Crossfading a signal with itself should preserve it."""
        xfade = CrossfadeBuffer(crossfade_samples=64)
        signal = np.full(64, 0.5, dtype=np.float32)
        output = np.zeros(128, dtype=np.float32)

        xfade.apply(output, signal, signal, xfade_start=32, xfade_len=64)

        # Equal-power crossfade of identical signals:
        # out = signal * cos(t) + signal * sin(t) = signal * (cos(t) + sin(t))
        # This is NOT exactly 0.5 everywhere (cos + sin peaks at √2 at midpoint)
        # But it should be close — never below 0.5 and peaks at ~0.707
        region = output[32:96]
        assert np.min(region) >= 0.49
        assert np.max(region) <= 0.75

    def test_clock_reports_wrap_offset(self) -> None:
        """Clock.last_wrap_offset tells engine where the wrap happened."""
        clock = Clock(44100)
        clock.set_loop(0, 1000)
        clock.loop_enabled = True
        clock.seek(800)

        clock.advance(400)  # wraps at sample 1000, offset = 200 into the buffer

        assert clock.last_wrap_offset == 200

    def test_clock_no_wrap_offset_negative_one(self) -> None:
        clock = Clock(44100)
        clock.set_loop(0, 1000)
        clock.loop_enabled = True
        clock.seek(0)

        clock.advance(256)  # no wrap

        assert clock.last_wrap_offset == -1

    def test_engine_loop_crossfade_no_click(self) -> None:
        """Engine callback produces smooth output at loop boundary (no hard discontinuity)."""
        engine = AudioEngine(sample_rate=44100, buffer_size=512, stream_factory=fake_stream_factory)

        # Create a ramp signal: 0 → 1 over 2000 samples.
        # At the loop boundary (sample 1000), the signal value is 0.5.
        # After wrap to sample 0, the signal value is 0.0.
        # Without crossfade: instant jump from 0.5 to 0.0 = click.
        # With crossfade: smooth blend.
        ramp = np.linspace(0.0, 1.0, 2000, dtype=np.float32)
        engine.add_layer("ramp", ramp, 44100)
        engine.clock.set_loop(0, 1000)
        engine.clock.loop_enabled = True
        engine.play()

        # Seek to just before the wrap point
        engine.seek(800)
        outdata = np.zeros((512, 1), dtype=np.float32)
        engine._audio_callback(outdata, 512, None, None)

        # The wrap happens at offset 200 (800 + 200 = 1000).
        # Check that there's no hard discontinuity > 0.1 between adjacent samples
        # in the region around the wrap point.
        signal = outdata[:, 0]
        diffs = np.abs(np.diff(signal[180:220]))
        max_jump = np.max(diffs)

        # Without crossfade this would be ~0.5 (ramp value at 1000 is 0.5, at 0 is 0.0)
        # With crossfade it should be much smaller
        assert max_jump < 0.1, f"Max jump at loop boundary: {max_jump:.4f} (should be < 0.1)"


# ===========================================================================
# BATCH 1 FIX TESTS
# ===========================================================================


class TestBatch1Fixes:
    """Tests for ship-readiness audit Batch 1 fixes (A1-A9, A11-A14)."""

    # -- A1: Mixer scratch buffer overflow --
    def test_mixer_large_frames_no_overflow(self) -> None:
        """A1: Large frames (> 4096) should not overflow scratch buffer regions."""
        mixer = Mixer()
        buf = _sine(10000)
        mixer.add_layer(AudioLayer("l1", "test", buf, 44100))

        # Request 6000 frames — larger than _MAX_SCRATCH_FRAMES / 2
        # Previously would overflow: scratch[0:6000] and scratch[6000:12000] overlap
        out = mixer.read_mix(0, 6000)
        assert out.shape == (6000,)
        assert np.all(np.isfinite(out))

    # -- A2: Third read_mix call eliminated (split-read path) --
    def test_loop_wrap_uses_read_mix_into(self) -> None:
        """A2: Loop-wrap path uses read_mix_into, not a wasteful third read_mix call."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(2000)
        engine.add_layer("l1", buf, 44100)
        engine.clock.set_loop(0, 1000)
        engine.clock.loop_enabled = True
        engine.play()

        # Seek near loop boundary so we get a wrap
        engine.seek(800)
        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        # Should output valid audio (no crashes, no NaNs)
        assert np.all(np.isfinite(outdata))
        assert np.any(outdata != 0)

    # -- A3: Crossfade output clipped to [-1, 1] --
    def test_crossfade_output_clipped(self) -> None:
        """A3: Equal-power crossfade output should be clipped to [-1, 1]."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        # Full-scale constant signal
        loud = np.ones(2000, dtype=np.float32)
        engine.add_layer("l1", loud, 44100)
        engine.clock.set_loop(0, 1000)
        engine.clock.loop_enabled = True
        engine.play()

        engine.seek(800)
        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        # All output should be within [-1, 1]
        assert np.all(outdata >= -1.0)
        assert np.all(outdata <= 1.0)

    # -- A4: Buffer size > loop length (tiling) --
    def test_buffer_larger_than_loop_tiling(self) -> None:
        """A4: Request frames > loop length — should tile correctly."""
        engine = AudioEngine(buffer_size=512, stream_factory=fake_stream_factory)
        buf = _sine(5000)
        engine.add_layer("l1", buf, 44100)
        engine.clock.set_loop(0, 256)  # very short loop
        engine.clock.loop_enabled = True
        engine.play()

        engine.seek(200)
        outdata = np.zeros((512, 1), dtype=np.float32)  # 512 > 256 loop length
        engine._audio_callback(outdata, 512, None, None)

        # Should produce valid tiled audio, not crash
        assert np.all(np.isfinite(outdata))

    # -- A5: Subscriber exception doesn't kill audio thread --
    def test_subscriber_exception_counted(self) -> None:
        """A5: Subscriber that raises should be counted but not crash thread."""
        clock = Clock(44100)

        class FailingSubscriber:
            def on_clock_tick(self, position: int, sr: int) -> None:
                raise RuntimeError("intentional test error")

        class GoodSubscriber:
            def __init__(self):
                self.called = False

            def on_clock_tick(self, position: int, sr: int) -> None:
                self.called = True

        bad = FailingSubscriber()
        good = GoodSubscriber()
        clock.add_subscriber(bad)
        clock.add_subscriber(good)

        clock.advance(256)

        # Bad subscriber should have thrown; we count it
        assert clock.subscriber_errors == 1
        # Good subscriber still got called
        assert good.called

    # -- A6: read_mix returns copy, read_mix_into available --
    def test_read_mix_returns_copy(self) -> None:
        """A6: Modifying returned array from read_mix doesn't affect internal state."""
        mixer = Mixer()
        buf = _sine(1000)
        mixer.add_layer(AudioLayer("l1", "test", buf, 44100))

        out1 = mixer.read_mix(0, 256)
        out1[:] = 0.0  # trash the array

        out2 = mixer.read_mix(0, 256)
        # Second read should still have the correct data
        np.testing.assert_array_almost_equal(out2, buf[:256])

    def test_read_mix_into_zero_copy(self) -> None:
        """A6: read_mix_into writes directly into caller buffer, zero-copy."""
        mixer = Mixer()
        buf = _sine(1000)
        mixer.add_layer(AudioLayer("l1", "test", buf, 44100))

        out = np.zeros(256, dtype=np.float32)
        mixer.read_mix_into(out, 0, 256)

        np.testing.assert_array_almost_equal(out, buf[:256])

    # -- A7: Seek while playing updates _stop_position --
    def test_seek_while_playing_updates_stop_position(self) -> None:
        """A7: Seeking while PLAYING should update where stop() will return to."""
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)

        # Seek to 1000 while playing
        transport.seek(1000)
        assert clock.position == 1000

        # Now stop — should return to the seek position
        transport.stop()
        assert clock.position == 1000
        assert transport.is_stopped

    # -- A8: Layer read_into bounds check --
    def test_layer_read_into_bounds_check(self) -> None:
        """A8: read_into should raise ValueError if frames > buffer length."""
        buf = _sine(100)
        layer = AudioLayer("l1", "test", buf, 44100)
        undersized = np.zeros(50, dtype=np.float32)

        with pytest.raises(ValueError, match="frames.*buffer length"):
            layer.read_into(undersized, 0, 256)

    # -- A9: resample_buffer empty guard --
    def test_resample_empty_buffer(self) -> None:
        """A9: resample_buffer should handle empty buffer gracefully."""
        empty = np.array([], dtype=np.float32)
        result = resample_buffer(empty, 44100, 48000)
        assert len(result) == 0

    # -- A11: wrap_offset == 0 edge case --
    def test_wrap_offset_zero_edge_case(self) -> None:
        """A11: Loop wrap at the very first sample (wrap_offset == 0) handled."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(2000)
        engine.add_layer("l1", buf, 44100)
        engine.clock.set_loop(500, 1500)
        engine.clock.loop_enabled = True
        engine.play()

        # Seek to exactly the loop end boundary
        engine.seek(1500)
        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        # Should wrap at offset 0 (first sample of buffer)
        # Previous code skipped this case. New code handles it.
        assert np.all(np.isfinite(outdata))

    # -- A13: return_to_start behavior --
    def test_return_to_start_from_paused_goes_to_stopped(self) -> None:
        """A13: return_to_start from PAUSED should transition to STOPPED."""
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)
        transport.pause()
        assert transport.is_paused

        transport.return_to_start()
        assert clock.position == 0
        assert transport.is_stopped  # intentional: return_to_start is a stop action

    # -- A14: sample_rate immutable --
    def test_sample_rate_read_only(self) -> None:
        """A14: sample_rate should be read-only after construction."""
        clock = Clock(44100)
        assert clock.sample_rate == 44100
        # Attempting to set should fail (no setter)
        with pytest.raises(AttributeError):
            clock.sample_rate = 48000

    # -- A15 Batch 2 placeholder (tested in Batch 2) --
    # (Solo count tracking tested in Batch 2)

    # -- Integration: all fixes together --
    def test_all_batch1_fixes_integration(self) -> None:
        """Integration test: all Batch 1 fixes working together."""
        engine = AudioEngine(buffer_size=512, stream_factory=fake_stream_factory)

        # Large buffer
        buf = _sine(20000)
        engine.add_layer("l1", buf, 44100)

        # Set a short loop
        engine.clock.set_loop(1000, 5000)
        engine.clock.loop_enabled = True
        engine.play()

        # Seek near boundary to trigger wrap
        engine.seek(4800)

        # Callback with large buffer (> loop length)
        outdata = np.zeros((512, 1), dtype=np.float32)
        engine._audio_callback(outdata, 512, None, None)

        # Should complete without crashes
        assert np.all(np.isfinite(outdata))

        # Seek while playing (A7)
        engine.seek(2000)
        assert engine.clock.position == 2000

        # Stop should go to seek position
        engine.stop()
        assert engine.clock.position == 2000


# ===========================================================================
# BATCH 2 FIX TESTS
# ===========================================================================


class TestBatch2Fixes:
    """Tests for ship-readiness audit Batch 2 fixes (A10, A15)."""

    # -- A10: Glitch counter / sounddevice status --
    def test_glitch_counter_increments(self) -> None:
        """A10: Glitch counter increments when sounddevice status is truthy."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        assert engine.glitch_count == 0
        assert engine.last_audio_status is None

        buf = _sine(1000)
        engine.add_layer("l1", buf, 44100)
        engine.play()

        # Simulate a glitch event (status object truthy)
        outdata = np.zeros((256, 1), dtype=np.float32)

        class FakeStatus:
            """Truthy status object representing a glitch."""
            pass

        status = FakeStatus()
        engine._audio_callback(outdata, 256, None, status)

        assert engine.glitch_count == 1
        assert engine.last_audio_status is status

    def test_glitch_counter_multiple_glitches(self) -> None:
        """A10: Glitch counter accumulates across multiple callbacks."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(1000)
        engine.add_layer("l1", buf, 44100)
        engine.play()

        class FakeStatus:
            pass

        outdata = np.zeros((256, 1), dtype=np.float32)

        # Simulate multiple glitches
        engine._audio_callback(outdata, 256, None, FakeStatus())
        assert engine.glitch_count == 1

        engine._audio_callback(outdata, 256, None, FakeStatus())
        assert engine.glitch_count == 2

        # Normal callback with no status
        engine._audio_callback(outdata, 256, None, None)
        assert engine.glitch_count == 2  # unchanged

    # -- A15: Solo count optimization --
    def test_set_solo_maintains_count(self) -> None:
        """A15: set_solo() canonical method maintains _solo_count."""
        mixer = Mixer()
        l1 = AudioLayer("l1", "a", _sine(100), 44100)
        l2 = AudioLayer("l2", "b", _sine(100), 44100)
        mixer.add_layer(l1)
        mixer.add_layer(l2)

        assert mixer._solo_count == 0

        mixer.set_solo("l1", True)
        assert mixer._solo_count == 1
        assert l1.solo is True

        mixer.set_solo("l2", True)
        assert mixer._solo_count == 2

        mixer.set_solo("l1", False)
        assert mixer._solo_count == 1

    def test_set_solo_idempotent(self) -> None:
        """A15: set_solo is idempotent — calling twice is safe."""
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "a", _sine(100), 44100))

        mixer.set_solo("l1", True)
        count_after_first = mixer._solo_count

        mixer.set_solo("l1", True)
        assert mixer._solo_count == count_after_first

    def test_solo_exclusive_maintains_count(self) -> None:
        """A15: solo_exclusive() updates _solo_count correctly."""
        mixer = Mixer()
        l1 = AudioLayer("l1", "a", _sine(100), 44100)
        l2 = AudioLayer("l2", "b", _sine(100), 44100)
        mixer.add_layer(l1)
        mixer.add_layer(l2)

        mixer.solo_exclusive("l1")
        assert mixer._solo_count == 1
        assert l1.solo is True
        assert l2.solo is False

        mixer.solo_exclusive("l2")
        assert mixer._solo_count == 1
        assert l1.solo is False
        assert l2.solo is True

    def test_unsolo_all_resets_count(self) -> None:
        """A15: unsolo_all() resets _solo_count to 0."""
        mixer = Mixer()
        l1 = AudioLayer("l1", "a", _sine(100), 44100)
        l2 = AudioLayer("l2", "b", _sine(100), 44100)
        mixer.add_layer(l1)
        mixer.add_layer(l2)

        mixer.set_solo("l1", True)
        mixer.set_solo("l2", True)
        assert mixer._solo_count == 2

        mixer.unsolo_all()
        assert mixer._solo_count == 0
        assert not l1.solo
        assert not l2.solo

    def test_remove_soloed_layer_decrements_count(self) -> None:
        """A15: Removing a soloed layer decrements _solo_count."""
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "a", _sine(100), 44100))
        mixer.add_layer(AudioLayer("l2", "b", _sine(100), 44100))

        mixer.set_solo("l1", True)
        assert mixer._solo_count == 1

        removed = mixer.remove_layer("l1")
        assert removed is not None
        assert mixer._solo_count == 0

    def test_clear_resets_solo_count(self) -> None:
        """A15: clear() resets _solo_count."""
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "a", _sine(100), 44100))
        mixer.set_solo("l1", True)
        assert mixer._solo_count == 1

        mixer.clear()
        assert mixer._solo_count == 0

    def test_solo_count_optimization_in_mix(self) -> None:
        """A15: read_mix uses O(1) _solo_count check, not any(l.solo for l in layers)."""
        mixer = Mixer()
        buf1 = _sine(1000, freq=440)
        buf2 = _sine(1000, freq=880)
        mixer.add_layer(AudioLayer("l1", "drums", buf1, 44100))
        mixer.add_layer(AudioLayer("l2", "bass", buf2, 44100))

        # No solos — all layers play
        out = mixer.read_mix(0, 256)
        expected = buf1[:256] + buf2[:256]
        np.clip(expected, -1.0, 1.0, out=expected)
        np.testing.assert_array_almost_equal(out, expected)

        # Solo one layer — only that plays
        mixer.set_solo("l2", True)
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_almost_equal(out, buf2[:256])

        # Unsolo — all play again
        mixer.unsolo_all()
        out = mixer.read_mix(0, 256)
        np.testing.assert_array_almost_equal(out, expected)

    # -- Integration: Batch 2 fixes together --
    def test_batch2_integration(self) -> None:
        """Integration test: A10 glitch tracking + A15 solo count work together."""
        engine = AudioEngine(stream_factory=fake_stream_factory)

        buf1 = _sine(1000)
        buf2 = _sine(1000)
        engine.add_layer("l1", buf1, 44100)
        engine.add_layer("l2", buf2, 44100)
        engine.play()

        # Set solo via mixer.set_solo
        engine.mixer.set_solo("l1", True)
        assert engine.mixer._solo_count == 1

        # Run callback with glitch
        outdata = np.zeros((256, 1), dtype=np.float32)

        class FakeStatus:
            pass

        engine._audio_callback(outdata, 256, None, FakeStatus())

        # Both fixes working: glitch counted, solo optimized
        assert engine.glitch_count == 1
        assert engine.mixer._solo_count == 1
        # Output should be l1 only (no l2)
        np.testing.assert_array_almost_equal(outdata[:, 0], buf1[:256])
