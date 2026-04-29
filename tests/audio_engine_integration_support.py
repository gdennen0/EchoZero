"""Integrated audio-engine support cases.
Exists to isolate engine and crossfade behavior from lower-level clock and layer support tests.
Connects the compatibility wrapper to the bounded integration slice.
"""

from tests.audio_engine_shared_support import *  # noqa: F401,F403

class TestAudioEngine:
    def test_create_engine(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        assert engine.sample_rate == 44100
        assert engine.buffer_size == 256
        assert not engine.is_active

    def test_resolve_output_defaults_keeps_legacy_defaults_for_injected_streams(self) -> None:
        sample_rate, channels = _resolve_output_defaults(fake_stream_factory)
        assert sample_rate == 44100
        assert channels == 1

    def test_resolve_output_defaults_prefers_default_output_device(self, monkeypatch) -> None:
        class _FakeSoundDevice:
            default = type("_Default", (), {"device": [0, 1]})()

            @staticmethod
            def query_devices(index):
                assert index == 1
                return {
                    "default_samplerate": 48000.0,
                    "max_output_channels": 2,
                }

        monkeypatch.setitem(__import__("sys").modules, "sounddevice", _FakeSoundDevice())

        sample_rate, channels = _resolve_output_defaults(None)

        assert sample_rate == 48000
        assert channels == 2

    def test_resolve_output_defaults_prefers_selected_output_device(self, monkeypatch) -> None:
        queried_indexes: list[int | str] = []

        class _FakeSoundDevice:
            default = type("_Default", (), {"device": [0, 1]})()

            @staticmethod
            def query_devices(index):
                queried_indexes.append(index)
                return {
                    "default_samplerate": 96000.0,
                    "max_output_channels": 2,
                }

        monkeypatch.setitem(__import__("sys").modules, "sounddevice", _FakeSoundDevice())

        sample_rate, channels = _resolve_output_defaults(None, output_device=7)

        assert queried_indexes == [7]
        assert sample_rate == 96000
        assert channels == 2

    def test_resolve_output_defaults_auto_prefers_device_default_when_supported(self, monkeypatch) -> None:
        checked_rates: list[int] = []

        class _FakeSoundDevice:
            default = type("_Default", (), {"device": [0, 1]})()

            @staticmethod
            def query_devices(index):
                assert index == 1
                return {
                    "default_samplerate": 48000.0,
                    "max_output_channels": 2,
                }

            @staticmethod
            def check_output_settings(*, device, channels, dtype, samplerate):
                assert device == 1
                assert channels == 2
                assert dtype == "float32"
                checked_rates.append(int(samplerate))
                if int(samplerate) not in {44100, 48000}:
                    raise ValueError("unsupported rate")

        monkeypatch.setitem(__import__("sys").modules, "sounddevice", _FakeSoundDevice())

        sample_rate, channels = _resolve_output_defaults(None)

        assert sample_rate == 48000
        assert channels == 2
        assert checked_rates[0] == 48000

    def test_resolve_output_defaults_auto_falls_back_when_default_is_unsupported(self, monkeypatch) -> None:
        checked_rates: list[int] = []

        class _FakeSoundDevice:
            default = type("_Default", (), {"device": [0, 1]})()

            @staticmethod
            def query_devices(index):
                assert index == 1
                return {
                    "default_samplerate": 12345.0,
                    "max_output_channels": 2,
                }

            @staticmethod
            def check_output_settings(*, device, channels, dtype, samplerate):
                assert device == 1
                assert channels == 2
                assert dtype == "float32"
                checked_rates.append(int(samplerate))
                if int(samplerate) != 44100:
                    raise ValueError("unsupported rate")

        monkeypatch.setitem(__import__("sys").modules, "sounddevice", _FakeSoundDevice())

        sample_rate, channels = _resolve_output_defaults(None)

        assert sample_rate == 44100
        assert channels == 2
        assert checked_rates[0] == 12345
        assert 44100 in checked_rates

    def test_resolve_stream_defaults_keeps_aggressive_injected_stream_behavior(self) -> None:
        blocksize, latency, prime_output = _resolve_stream_defaults(
            fake_stream_factory,
            buffer_size=256,
            blocksize=None,
            latency=None,
            prime_output_buffers_using_stream_callback=True,
        )

        assert blocksize == 0
        assert latency == "low"
        assert prime_output is True

    def test_resolve_stream_defaults_prefers_stable_real_device_behavior(self) -> None:
        blocksize, latency, prime_output = _resolve_stream_defaults(
            None,
            buffer_size=256,
            blocksize=None,
            latency=None,
            prime_output_buffers_using_stream_callback=True,
        )

        assert blocksize == 0
        assert latency == "high"
        assert prime_output is True

    def test_play_opens_stream(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.play()
        assert engine.is_active
        assert engine.transport.is_playing

    def test_play_passes_selected_output_device_to_stream(self) -> None:
        engine = AudioEngine(output_device="7", stream_factory=fake_stream_factory)

        engine.play()

        assert engine._stream.device == "7"

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

    def test_extract_output_latency_seconds_from_callback_time_info(self) -> None:
        latency = AudioEngine._extract_output_latency_seconds(
            {
                "currentTime": 10.0,
                "outputBufferDacTime": 10.125,
            }
        )

        assert latency == pytest.approx(0.125)

    def test_audible_time_seconds_extrapolates_between_callbacks(self, monkeypatch) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = _sine(4000)
        engine.add_layer("l1", buf, 44100)
        engine.play()

        monotonic_now = {"value": 100.0}
        monkeypatch.setattr("echozero.audio.engine.time.monotonic", lambda: monotonic_now["value"])

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(
            outdata,
            256,
            {"currentTime": 5.0, "outputBufferDacTime": 5.1},
            None,
        )

        snapshot_time = engine.audible_time_seconds
        monotonic_now["value"] = 100.03
        later_time = engine.audible_time_seconds

        assert later_time > snapshot_time
        assert later_time <= engine.clock.position_seconds

    def test_callback_resamples_source_and_duplicates_to_stereo_output(self) -> None:
        engine = AudioEngine(sample_rate=48000, channels=2, stream_factory=fake_stream_factory)
        buf = _sine(4410, sr=44100)
        layer = engine.add_layer("l1", buf, 44100)

        assert layer.original_sample_rate == 44100
        assert layer.sample_rate == 48000

        engine.play()
        outdata = np.zeros((256, 2), dtype=np.float32)
        engine._audio_callback(
            outdata,
            256,
            {"currentTime": 1.0, "outputBufferDacTime": 1.01},
            None,
        )

        assert np.all(np.isfinite(outdata))
        assert np.max(np.abs(outdata)) > 0.0
        np.testing.assert_allclose(outdata[:, 0], outdata[:, 1])

    def test_callback_preserves_stereo_source_channels(self) -> None:
        engine = AudioEngine(sample_rate=44100, channels=2, stream_factory=fake_stream_factory)
        stereo = np.column_stack(
            (
                np.ones(2048, dtype=np.float32),
                -np.ones(2048, dtype=np.float32),
            )
        )
        engine.add_layer("l1", stereo, 44100)

        engine.play()
        outdata = np.zeros((256, 2), dtype=np.float32)
        engine._audio_callback(
            outdata,
            256,
            {"currentTime": 1.0, "outputBufferDacTime": 1.01},
            None,
        )

        np.testing.assert_array_equal(outdata[:, 0], np.ones(256, dtype=np.float32))
        np.testing.assert_array_equal(outdata[:, 1], -np.ones(256, dtype=np.float32))

    def test_callback_handles_large_variable_frame_sizes_with_preallocated_scratch(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.add_layer("l1", _sine(40000), 44100)
        engine.play()

        outdata = np.zeros((10000, 1), dtype=np.float32)
        engine._audio_callback(outdata, 10000, None, None)

        assert np.all(np.isfinite(outdata))
        assert np.max(np.abs(outdata)) > 0.0

    def test_callback_zeros_and_counts_glitch_when_frame_count_exceeds_scratch_capacity(self) -> None:
        engine = AudioEngine(stream_factory=fake_stream_factory)
        engine.add_layer("l1", _sine(80000), 44100)
        engine.play()

        outdata = np.ones((40000, 1), dtype=np.float32)
        engine._audio_callback(outdata, 40000, None, None)

        np.testing.assert_array_equal(outdata, np.zeros((40000, 1), dtype=np.float32))
        assert engine.glitch_count == 1
        assert engine.last_audio_status == "callback_frames_exceeded_scratch:40000>32768"

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

    def test_callback_sanitizes_non_finite_output(self) -> None:
        """Non-finite layer samples should not reach the device buffer."""
        engine = AudioEngine(stream_factory=fake_stream_factory)
        buf = np.zeros(512, dtype=np.float32)
        buf[0] = np.nan
        buf[1] = np.inf
        buf[2] = -np.inf
        buf[3] = 0.25
        engine.add_layer("l1", buf, 44100)
        engine.play()

        outdata = np.zeros((256, 1), dtype=np.float32)
        engine._audio_callback(outdata, 256, None, None)

        assert np.all(np.isfinite(outdata))
        assert outdata[0, 0] == 0.0
        assert outdata[1, 0] == 1.0
        assert outdata[2, 0] == -1.0
        assert outdata[3, 0] == pytest.approx(0.25)


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



__all__ = [name for name in globals() if name.startswith("Test")]
