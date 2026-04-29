"""
Audio file cache tests: verify decoded audio reuse across hot app paths.
Exists because import-time latency depends on avoiding duplicate file decodes.
Connects the shared audio cache to waveform registration and runtime playback loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from echozero.application.playback.runtime import _load_runtime_audio
from echozero.audio.file_cache import clear_audio_file_cache, load_audio_file
from echozero.ui.qt.timeline.waveform_cache import (
    clear_waveform_cache,
    register_waveform_from_audio_file,
)


def _write_audio_marker(path: Path) -> Path:
    """Create a real file path for cache-key stat tracking during tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + b"\x00" * 32)
    return path


class TestAudioFileCache:
    def test_load_audio_file_reuses_cached_buffer_for_unchanged_path(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        clear_audio_file_cache()
        audio_path = _write_audio_marker(tmp_path / "song.wav")
        calls: list[Path] = []

        def _fake_read(path: Path) -> tuple[np.ndarray, int]:
            calls.append(path)
            return np.array([0.25, -0.25], dtype=np.float32), 44100

        monkeypatch.setattr("echozero.audio.file_cache._read_audio_file", _fake_read)

        first_samples, first_sample_rate = load_audio_file(audio_path)
        second_samples, second_sample_rate = load_audio_file(audio_path)

        assert len(calls) == 1
        assert first_sample_rate == 44100
        assert second_sample_rate == 44100
        assert first_samples is second_samples
        clear_audio_file_cache()

    def test_waveform_and_runtime_audio_share_decoded_file_cache(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        clear_audio_file_cache()
        clear_waveform_cache()
        audio_path = _write_audio_marker(tmp_path / "shared.wav")
        calls: list[Path] = []
        stereo = np.column_stack(
            (
                np.linspace(-1.0, 1.0, 8, dtype=np.float32),
                np.linspace(1.0, -1.0, 8, dtype=np.float32),
            )
        )

        def _fake_read(path: Path) -> tuple[np.ndarray, int]:
            calls.append(path)
            return stereo, 48000

        monkeypatch.setattr("echozero.audio.file_cache._read_audio_file", _fake_read)

        waveform = register_waveform_from_audio_file("song", audio_path, window_size=2)
        samples, sample_rate = _load_runtime_audio(audio_path)

        assert len(calls) == 1
        assert waveform.sample_rate == 48000
        assert waveform.peaks.shape == (4, 2)
        assert sample_rate == 48000
        assert samples.shape == (8, 2)
        clear_waveform_cache()
        clear_audio_file_cache()
