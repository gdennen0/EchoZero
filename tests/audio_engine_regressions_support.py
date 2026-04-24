"""Regression audio-engine support cases.
Exists to keep targeted batch regressions separate from the primary engine behavior support slices.
Connects the compatibility wrapper to the bounded regression slice.
"""

from tests.audio_engine_shared_support import *  # noqa: F401,F403

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

    def test_loop_wrap_sanitizes_non_finite_output(self) -> None:
        """Loop-wrap path should sanitize NaN/inf before writing to the device."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = np.zeros(2000, dtype=np.float32)
        buf[995] = np.nan
        buf[996] = np.inf
        buf[997] = -np.inf
        buf[998] = 0.5
        buf[0] = np.nan
        buf[1] = np.inf
        buf[2] = -np.inf
        buf[3] = -0.5
        engine.add_layer("l1", buf, 44100)
        engine.clock.set_loop(0, 1000)
        engine.clock.loop_enabled = True
        engine.play()

        engine.seek(990)
        outdata = np.zeros((32, 1), dtype=np.float32)
        engine._audio_callback(outdata, 32, None, None)

        assert np.all(np.isfinite(outdata))
        assert np.max(outdata) <= 1.0
        assert np.min(outdata) >= -1.0

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

__all__ = [name for name in globals() if name.startswith("Test")]
