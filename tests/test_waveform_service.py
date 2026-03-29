"""
WaveformService tests: Verify auto-generation of waveform peaks for song versions.
Exists because waveform generation must run through the real engine (FP1) on every
song import — this service is the bridge between persistence and engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from echozero.domain.types import WaveformData
from echozero.persistence.entities import SongVersion
from echozero.processors.load_audio import AudioFileInfo
from echozero.services.waveform import generate_waveform_for_version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_version(audio_file: str = "audio/test.wav") -> SongVersion:
    return SongVersion(
        id="v1",
        song_id="s1",
        label="Original",
        audio_file=audio_file,
        duration_seconds=3.0,
        original_sample_rate=44100,
        audio_hash="abc123",
        created_at=datetime.now(timezone.utc),
    )


def _mock_audio_info(path: str) -> AudioFileInfo:
    return AudioFileInfo(sample_rate=44100, duration=3.0, channels=2)


def _mock_load_samples(path: str, sr: int) -> np.ndarray:
    """Return a 1-second sine wave."""
    t = np.arange(sr, dtype=np.float32) / sr
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateWaveformForVersion:
    """Verify the waveform service produces WaveformData through the engine."""

    def test_returns_waveform_data(self, tmp_path: Path) -> None:
        # Create a fake audio file
        audio_dir = tmp_path
        (audio_dir / "audio").mkdir()
        (audio_dir / "audio" / "test.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        version = _mock_version()
        result = generate_waveform_for_version(
            version,
            audio_dir=audio_dir,
            load_samples_fn=_mock_load_samples,
            audio_info_fn=_mock_audio_info,
        )

        assert result is not None
        assert isinstance(result, WaveformData)

    def test_has_four_lod_levels(self, tmp_path: Path) -> None:
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "test.wav").write_bytes(b"fake")

        result = generate_waveform_for_version(
            _mock_version(),
            audio_dir=tmp_path,
            load_samples_fn=_mock_load_samples,
            audio_info_fn=_mock_audio_info,
        )

        assert result is not None
        assert len(result.lods) == 4

    def test_lod_peaks_have_correct_shape(self, tmp_path: Path) -> None:
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "test.wav").write_bytes(b"fake")

        result = generate_waveform_for_version(
            _mock_version(),
            audio_dir=tmp_path,
            load_samples_fn=_mock_load_samples,
            audio_info_fn=_mock_audio_info,
        )

        assert result is not None
        for lod in result.lods:
            assert lod.peaks.ndim == 2
            assert lod.peaks.shape[1] == 2  # min/max pairs

    def test_preserves_sample_rate(self, tmp_path: Path) -> None:
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "test.wav").write_bytes(b"fake")

        result = generate_waveform_for_version(
            _mock_version(),
            audio_dir=tmp_path,
            load_samples_fn=_mock_load_samples,
            audio_info_fn=_mock_audio_info,
        )

        assert result is not None
        assert result.sample_rate == 44100

    def test_missing_audio_returns_none(self, tmp_path: Path) -> None:
        # No audio file created
        version = _mock_version("audio/nonexistent.wav")
        result = generate_waveform_for_version(
            version,
            audio_dir=tmp_path,
            load_samples_fn=_mock_load_samples,
            audio_info_fn=_mock_audio_info,
        )

        assert result is None

    def test_load_failure_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "test.wav").write_bytes(b"fake")

        def exploding_load(path, sr):
            raise RuntimeError("Corrupted")

        result = generate_waveform_for_version(
            _mock_version(),
            audio_dir=tmp_path,
            load_samples_fn=exploding_load,
            audio_info_fn=_mock_audio_info,
        )

        assert result is None
