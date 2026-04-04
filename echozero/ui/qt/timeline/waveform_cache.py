"""Lightweight waveform peak cache for timeline rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CachedWaveform:
    sample_rate: int
    window_size: int
    peaks: np.ndarray  # shape (N,2), float32 min/max

    @property
    def seconds_per_peak(self) -> float:
        return float(self.window_size) / float(self.sample_rate)


_CACHE: dict[str, CachedWaveform] = {}


def get_cached_waveform(key: str | None) -> CachedWaveform | None:
    if not key:
        return None
    return _CACHE.get(key)


def register_waveform_from_audio_file(
    key: str,
    audio_file: str | Path,
    *,
    window_size: int = 256,
) -> CachedWaveform:
    path = Path(audio_file)
    sr, samples = _load_audio(path)
    mono = _to_mono_float32(samples)
    peaks = _compute_min_max_peaks(mono, window_size=window_size)
    cached = CachedWaveform(sample_rate=sr, window_size=window_size, peaks=peaks)
    _CACHE[key] = cached
    return cached


def _load_audio(path: Path) -> tuple[int, np.ndarray]:
    # Fast path for WAV
    if path.suffix.lower() == ".wav":
        from scipy.io import wavfile

        sr, data = wavfile.read(path)
        return int(sr), np.asarray(data)

    # Fallback for other formats
    import librosa

    data, sr = librosa.load(str(path), sr=None, mono=False)
    if data.ndim == 1:
        return int(sr), np.asarray(data, dtype=np.float32)
    # librosa multi-channel returns (channels, samples)
    return int(sr), np.asarray(np.moveaxis(data, 0, 1), dtype=np.float32)


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples)
    if x.ndim > 1:
        x = x.mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if np.issubdtype(x.dtype, np.unsignedinteger):
            x = x.astype(np.float32) - (info.max / 2.0)
            denom = max(1.0, info.max / 2.0)
            x = x / denom
        else:
            denom = float(max(abs(info.min), abs(info.max)))
            x = x.astype(np.float32) / max(1.0, denom)
    else:
        x = x.astype(np.float32)

    peak = float(np.max(np.abs(x))) if x.size else 1.0
    if peak > 0:
        x = x / peak
    return x.astype(np.float32, copy=False)


def _compute_min_max_peaks(samples: np.ndarray, *, window_size: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    n = samples.size
    whole = (n // window_size) * window_size
    chunks = []

    if whole > 0:
        reshaped = samples[:whole].reshape(-1, window_size)
        mins = reshaped.min(axis=1)
        maxs = reshaped.max(axis=1)
        chunks.append(np.column_stack((mins, maxs)).astype(np.float32))

    if whole < n:
        tail = samples[whole:]
        chunks.append(np.array([[float(tail.min()), float(tail.max())]], dtype=np.float32))

    return np.vstack(chunks).astype(np.float32)
