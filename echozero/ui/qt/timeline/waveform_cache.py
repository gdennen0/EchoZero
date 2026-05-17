"""Waveform cache and background loaders for timeline rendering.
Exists to keep waveform extraction off the Qt paint path and reuse decoded peaks across views.
Connects timeline lanes and inspector previews to cached min/max waveform envelopes.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, RLock

import numpy as np
from PyQt6.QtCore import QMetaObject, QObject, Qt

from echozero.audio.file_cache import load_audio_file


@dataclass(slots=True)
class CachedWaveform:
    sample_rate: int
    window_size: int
    peaks: np.ndarray  # shape (N,2), float32 min/max
    duration_seconds: float | None = None

    @property
    def seconds_per_peak(self) -> float:
        return float(self.window_size) / float(self.sample_rate)

    @property
    def resolved_duration_seconds(self) -> float:
        if self.duration_seconds is not None:
            return max(0.0, float(self.duration_seconds))
        return max(0.0, float(self.peaks.shape[0]) * self.seconds_per_peak)


_CACHE: OrderedDict[str, CachedWaveform] = OrderedDict()
_CACHE_BYTES = 0
_CACHE_MAX_BYTES = int(float(os.environ.get("ECHOZERO_WAVEFORM_CACHE_MB", "256")) * 1024 * 1024)
_CACHE_LOCK = RLock()
_LOAD_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("ECHOZERO_WAVEFORM_LOAD_WORKERS", "2"))),
    thread_name_prefix="ez-waveform",
)
_LOAD_LOCK = Lock()
_PENDING_LOADS: set[tuple[str, str, int]] = set()
_FAILED_LOAD_RETRY_SECONDS = float(os.environ.get("ECHOZERO_WAVEFORM_FAILED_RETRY_SECONDS", "30"))
_FAILED_LOADS: dict[tuple[str, str, int], float] = {}


def get_cached_waveform(key: str | None) -> CachedWaveform | None:
    if not key:
        return None
    with _CACHE_LOCK:
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
        duration_seconds=(float(mono.size) / float(sample_rate)) if sample_rate > 0 else 0.0,
    )
    _put_cached_waveform(key, cached)
    return cached


def request_waveform_from_audio_file(
    key: str | None,
    audio_file: str | Path | None,
    *,
    receiver: QObject | None = None,
    window_size: int = 256,
) -> CachedWaveform | None:
    """Return a cached waveform or schedule a background load for a later repaint."""

    normalized_key = str(key or "").strip()
    normalized_path = str(audio_file or "").strip()
    if not normalized_key or not normalized_path:
        return None
    cached = get_cached_waveform(normalized_key)
    if cached is not None:
        return cached

    load_key = (normalized_key, normalized_path, int(window_size))
    with _LOAD_LOCK:
        retry_after = float(_FAILED_LOADS.get(load_key, 0.0))
        if retry_after > time.monotonic():
            return None
        if retry_after:
            _FAILED_LOADS.pop(load_key, None)
        if load_key in _PENDING_LOADS:
            return None
        _PENDING_LOADS.add(load_key)

    def _load() -> None:
        loaded = False
        try:
            register_waveform_from_audio_file(
                normalized_key,
                normalized_path,
                window_size=window_size,
            )
            loaded = True
        except Exception:
            with _LOAD_LOCK:
                _FAILED_LOADS[load_key] = time.monotonic() + _FAILED_LOAD_RETRY_SECONDS
        finally:
            with _LOAD_LOCK:
                if loaded:
                    _FAILED_LOADS.pop(load_key, None)
                _PENDING_LOADS.discard(load_key)
                should_notify = loaded or load_key in _FAILED_LOADS
            if receiver is not None and should_notify:
                _queue_receiver_update(receiver)

    _LOAD_EXECUTOR.submit(_load)
    return None


def clear_waveform_cache() -> None:
    global _CACHE_BYTES
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0
    with _LOAD_LOCK:
        _FAILED_LOADS.clear()


def set_waveform_cache_limit_bytes(limit_bytes: int) -> None:
    global _CACHE_MAX_BYTES
    with _CACHE_LOCK:
        _CACHE_MAX_BYTES = max(1024, int(limit_bytes))
        _evict_if_needed()


def waveform_cache_stats() -> dict[str, int]:
    with _CACHE_LOCK:
        return {
            "entries": len(_CACHE),
            "bytes": int(_CACHE_BYTES),
            "max_bytes": int(_CACHE_MAX_BYTES),
        }


def _estimate_bytes(cached: CachedWaveform) -> int:
    return int(cached.peaks.nbytes + 96)


def _put_cached_waveform(key: str, cached: CachedWaveform) -> None:
    global _CACHE_BYTES
    with _CACHE_LOCK:
        existing = _CACHE.pop(key, None)
        if existing is not None:
            _CACHE_BYTES -= _estimate_bytes(existing)

        _CACHE[key] = cached
        _CACHE_BYTES += _estimate_bytes(cached)
        _CACHE.move_to_end(key)
        _evict_if_needed()


def _evict_if_needed() -> None:
    global _CACHE_BYTES
    with _CACHE_LOCK:
        while _CACHE and _CACHE_BYTES > _CACHE_MAX_BYTES:
            _, evicted = _CACHE.popitem(last=False)
            _CACHE_BYTES -= _estimate_bytes(evicted)


def _queue_receiver_update(receiver: QObject) -> None:
    try:
        QMetaObject.invokeMethod(
            receiver,
            "update",
            Qt.ConnectionType.QueuedConnection,
        )
    except Exception:
        return


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
