"""Layer and mixer audio-engine support cases.
Exists to keep resample, layer, and mixer coverage separate from transport and integration support tests.
Connects the compatibility wrapper to the bounded audio-layer slice.
"""

from tests.audio_engine_shared_support import *  # noqa: F401,F403

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

    def test_read_into_preserves_stereo_channels(self) -> None:
        left = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        right = np.linspace(1.0, -1.0, 1000, dtype=np.float32)
        buf = np.column_stack((left, right)).astype(np.float32)
        layer = AudioLayer("l1", "test", buf, 44100)
        scratch = np.zeros((256, 2), dtype=np.float32)

        layer.read_into(scratch, 0, 256)

        np.testing.assert_array_equal(scratch[:, 0], left[:256])
        np.testing.assert_array_equal(scratch[:, 1], right[:256])

    def test_read_into_downmixes_stereo_when_output_is_mono(self) -> None:
        buf = np.array(
            [
                [1.0, -1.0],
                [0.5, -0.5],
                [0.25, -0.25],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        layer = AudioLayer("l1", "test", buf, 44100)
        scratch = np.zeros(4, dtype=np.float32)

        layer.read_into(scratch, 0, 4)

        np.testing.assert_array_equal(scratch, np.zeros(4, dtype=np.float32))

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

    def test_stereo_layer_preserved_in_multichannel_mix(self) -> None:
        stereo = np.column_stack(
            (
                np.ones(256, dtype=np.float32),
                -np.ones(256, dtype=np.float32),
            )
        )
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "stereo", stereo, 44100))

        out = mixer.read_mix(0, 256, channels=2)

        assert out.shape == (256, 2)
        np.testing.assert_array_equal(out[:, 0], np.ones(256, dtype=np.float32))
        np.testing.assert_array_equal(out[:, 1], -np.ones(256, dtype=np.float32))

    def test_mono_layer_broadcasts_into_stereo_mix(self) -> None:
        buf = _sine(1000)
        mixer = Mixer()
        mixer.add_layer(AudioLayer("l1", "mono", buf, 44100))

        out = mixer.read_mix(0, 256, channels=2)

        assert out.shape == (256, 2)
        np.testing.assert_array_almost_equal(out[:, 0], buf[:256])
        np.testing.assert_array_almost_equal(out[:, 1], buf[:256])

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



__all__ = [name for name in globals() if name.startswith("Test")]
