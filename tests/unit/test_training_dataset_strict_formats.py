"""
Canonical audio standardization tests for AudioClassificationDataset.
"""
from pathlib import Path

import numpy as np
import pytest

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
try:
    import librosa  # noqa: F401
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from src.application.blocks.training.datasets import AudioClassificationDataset


@pytest.mark.skipif((not HAS_SF) or (not HAS_LIBROSA), reason="soundfile/librosa not installed")
def test_standardize_samples_to_canonical_converts_wav_to_cached_wav(tmp_path: Path, monkeypatch):
    src_wav = tmp_path / "input.wav"
    sr = 22050
    audio = np.zeros(sr // 10, dtype=np.float32)
    sf.write(str(src_wav), audio, sr, subtype="PCM_16")

    ds = AudioClassificationDataset.__new__(AudioClassificationDataset)
    ds.sample_rate = sr
    ds.samples = [(src_wav, 1)]
    ds._formats_used = {"wav"}
    ds.config = {"exclude_bad_files": True}
    monkeypatch.setattr(
        "src.application.blocks.training.datasets.get_user_cache_dir",
        lambda: tmp_path / "cache",
    )

    ds._standardize_samples_to_canonical()

    assert len(ds.samples) == 1
    standardized_path, label = ds.samples[0]
    assert label == 1
    assert standardized_path.suffix.lower() == ".wav"
    assert standardized_path.exists()
    assert ds._formats_used == {"wav"}


@pytest.mark.skipif((not HAS_SF) or (not HAS_LIBROSA), reason="soundfile/librosa not installed")
def test_standardize_excludes_bad_files_when_enabled(tmp_path: Path, monkeypatch):
    sr = 22050
    good_wav = tmp_path / "good.wav"
    bad_mp3 = tmp_path / "bad.mp3"
    sf.write(str(good_wav), np.zeros(sr // 10, dtype=np.float32), sr, subtype="PCM_16")
    bad_mp3.write_bytes(b"")

    ds = AudioClassificationDataset.__new__(AudioClassificationDataset)
    ds.sample_rate = sr
    ds.samples = [(good_wav, 1), (bad_mp3, 0)]
    ds._formats_used = {"wav", "mp3"}
    ds.config = {"exclude_bad_files": True}
    monkeypatch.setattr(
        "src.application.blocks.training.datasets.get_user_cache_dir",
        lambda: tmp_path / "cache",
    )

    ds._standardize_samples_to_canonical()

    assert len(ds.samples) == 1
    standardized_path, label = ds.samples[0]
    assert label == 1
    assert standardized_path.suffix.lower() == ".wav"
    assert ds._excluded_bad_file_count == 1


@pytest.mark.skipif((not HAS_SF) or (not HAS_LIBROSA), reason="soundfile/librosa not installed")
def test_standardize_raises_for_bad_files_when_exclusion_disabled(tmp_path: Path, monkeypatch):
    sr = 22050
    bad_mp3 = tmp_path / "bad.mp3"
    bad_mp3.write_bytes(b"")

    ds = AudioClassificationDataset.__new__(AudioClassificationDataset)
    ds.sample_rate = sr
    ds.samples = [(bad_mp3, 0)]
    ds._formats_used = {"mp3"}
    ds.config = {"exclude_bad_files": False}
    monkeypatch.setattr(
        "src.application.blocks.training.datasets.get_user_cache_dir",
        lambda: tmp_path / "cache",
    )

    with pytest.raises(ValueError, match="Failed to standardize input audio files"):
        ds._standardize_samples_to_canonical()
