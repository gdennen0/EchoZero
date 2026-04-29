"""Lightweight waveform peak cache for timeline rendering."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echozero.audio.file_cache import load_audio_file


@dataclass(slots=True)
class CachedWaveform:
    sample_rate: int
    window_size: int
    peaks: np.ndarray  # shape (N,2), float32 min/max

    @property
    def seconds_per_peak(self) -> float:
        return float(self.window_size) / float(self.sample_rate)


_CACHE: OrderedDict[str, CachedWaveform] = OrderedDict()
_CACHE_BYTES = 0
_CACHE_MAX_BYTES = int(float(os.environ.get("ECHOZERO_WAVEFORM_CACHE_MB", "256")) * 1024 * 1024)


def get_cached_waveform(key: str | None) -> CachedWaveform | None:
    if not key:
        return None
    cached = _CACHE.get(key)
    if cached is None:
        return None
    _CACHE.move_to_end(key)
    return cached


def register_waveform_from_audio_file(
    key: str,
    audio_file: str | Path,
    *,
    window_size: int = 256,
) -> CachedWaveform:
    path = Path(audio_file)
    samples, sample_rate = load_audio_file(path)
    mono = _to_mono_float32(samples)
    peaks = _compute_min_max_peaks(mono, window_size=window_size)
    cached = CachedWaveform(
        sample_rate=int(sample_rate),
        window_size=window_size,
        peaks=peaks,
    )
    _put_cached_waveform(key, cached)
    return cached


def clear_waveform_cache() -> None:
    global _CACHE_BYTES
    _CACHE.clear()
    _CACHE_BYTES = 0


def set_waveform_cache_limit_bytes(limit_bytes: int) -> None:
    global _CACHE_MAX_BYTES
    _CACHE_MAX_BYTES = max(1024, int(limit_bytes))
    _evict_if_needed()


def waveform_cache_stats() -> dict[str, int]:
    return {
        "entries": len(_CACHE),
        "bytes": int(_CACHE_BYTES),
        "max_bytes": int(_CACHE_MAX_BYTES),
    }


def _estimate_bytes(cached: CachedWaveform) -> int:
    return int(cached.peaks.nbytes + 96)


def _put_cached_waveform(key: str, cached: CachedWaveform) -> None:
    global _CACHE_BYTES
    existing = _CACHE.pop(key, None)
    if existing is not None:
        _CACHE_BYTES -= _estimate_bytes(existing)

    _CACHE[key] = cached
    _CACHE_BYTES += _estimate_bytes(cached)
    _CACHE.move_to_end(key)
    _evict_if_needed()


def _evict_if_needed() -> None:
    global _CACHE_BYTES
    while _CACHE and _CACHE_BYTES > _CACHE_MAX_BYTES:
        _, evicted = _CACHE.popitem(last=False)
        _CACHE_BYTES -= _estimate_bytes(evicted)


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
