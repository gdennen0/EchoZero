"""
Audio file cache: Shared LRU cache for decoded audio file buffers.
Exists because waveform registration and runtime playback often touch the same file back-to-back.
Connects UI waveform loading and playback runtime to one bounded decode path.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np


@dataclass(slots=True)
class CachedAudioFile:
    """One decoded audio file cached for reuse across app surfaces."""

    sample_rate: int
    samples: np.ndarray

    @property
    def size_bytes(self) -> int:
        return int(self.samples.nbytes + 96)


_CACHE: OrderedDict[str, CachedAudioFile] = OrderedDict()
_CACHE_BYTES = 0
_CACHE_MAX_BYTES = int(
    float(os.environ.get("ECHOZERO_AUDIO_FILE_CACHE_MB", "256")) * 1024 * 1024
)
_CACHE_LOCK = RLock()


def load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Load one audio file through a shared decoded-buffer cache."""

    resolved_path = Path(path).expanduser().resolve()
    stat = resolved_path.stat()
    cache_key = (
        f"{resolved_path}"
        f"|mtime:{int(stat.st_mtime_ns)}"
        f"|size:{int(stat.st_size)}"
    )

    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _CACHE.move_to_end(cache_key)
            return cached.samples, cached.sample_rate

    samples, sample_rate = _read_audio_file(resolved_path)
    cached = CachedAudioFile(
        sample_rate=int(sample_rate),
        samples=np.asarray(samples, dtype=np.float32),
    )

    with _CACHE_LOCK:
        _put_cached_audio(cache_key, cached)
    return cached.samples, cached.sample_rate


def clear_audio_file_cache() -> None:
    """Drop all cached decoded audio buffers."""

    global _CACHE_BYTES
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0


def _put_cached_audio(cache_key: str, cached: CachedAudioFile) -> None:
    global _CACHE_BYTES

    existing = _CACHE.pop(cache_key, None)
    if existing is not None:
        _CACHE_BYTES -= existing.size_bytes

    _CACHE[cache_key] = cached
    _CACHE_BYTES += cached.size_bytes
    _CACHE.move_to_end(cache_key)
    _evict_if_needed()


def _evict_if_needed() -> None:
    global _CACHE_BYTES

    while _CACHE and _CACHE_BYTES > _CACHE_MAX_BYTES:
        _, evicted = _CACHE.popitem(last=False)
        _CACHE_BYTES -= evicted.size_bytes


def _read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        samples, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    except Exception:
        import librosa

        samples, sample_rate = librosa.load(str(path), sr=None, mono=False)
        if np.asarray(samples).ndim == 1:
            return np.asarray(samples, dtype=np.float32), int(sample_rate)
        return np.asarray(np.moveaxis(samples, 0, 1), dtype=np.float32), int(sample_rate)
