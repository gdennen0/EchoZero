"""Audio event-shape similarity helpers."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def align_shape_to_reference(reference: tuple[float, ...], candidate: tuple[float, ...]) -> tuple[float, ...]:
    ref = _normalize(np.asarray(reference, dtype=np.float32))
    cand = _normalize(np.asarray(candidate, dtype=np.float32))
    if ref.size == 0 or cand.size == 0:
        return tuple(float(x) for x in cand)
    if cand.size != ref.size:
        cand = np.interp(
            np.linspace(0.0, 1.0, ref.size),
            np.linspace(0.0, 1.0, cand.size),
            cand,
        ).astype(np.float32)
    best = cand
    best_score = -1.0
    max_shift = max(1, ref.size // 3)
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.zeros_like(cand)
        if shift < 0:
            shifted[:shift] = cand[-shift:]
        elif shift > 0:
            shifted[shift:] = cand[:-shift]
        else:
            shifted = cand.copy()
        score = float(np.dot(ref, _normalize(shifted)))
        if score > best_score:
            best_score = score
            best = shifted
    return tuple(float(x) for x in best)


def compare_shape_similarity(reference: tuple[float, ...], candidate: tuple[float, ...]) -> float:
    ref = _normalize(np.asarray(reference, dtype=np.float32))
    aligned = _normalize(np.asarray(align_shape_to_reference(reference, candidate), dtype=np.float32))
    if ref.size == 0 or aligned.size == 0:
        return 0.0
    cosine = float(np.dot(ref, aligned))
    rmse_penalty = float(np.sqrt(np.mean(np.square(ref - aligned)))) * 3.5
    return max(0.0, min(1.0, cosine - rmse_penalty))


def read_mono_audio_slice(path: str | Path, *, start_seconds: float, end_seconds: float) -> tuple[np.ndarray, int] | None:
    try:
        with wave.open(str(path), "rb") as handle:
            sr = int(handle.getframerate())
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            start = max(0, int(round(float(start_seconds) * sr)))
            end = max(start, int(round(float(end_seconds) * sr)))
            handle.setpos(min(start, handle.getnframes()))
            raw = handle.readframes(max(0, min(end, handle.getnframes()) - start))
        if not raw:
            return None
        data = _decode_pcm(raw, sample_width=width)
        if data is None:
            return None
        if channels > 1:
            data = data.reshape((-1, channels)).mean(axis=1)
        return data, sr
    except Exception:
        return None


def _decode_pcm(raw: bytes, *, sample_width: int) -> np.ndarray | None:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        bytes_array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = bytes_array[:, 0] | (bytes_array[:, 1] << 8) | (bytes_array[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    return None


def audio_shape_preview(audio: np.ndarray, *, sample_count: int = 64) -> tuple[float, ...]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ()
    env = np.abs(arr)
    if env.size != sample_count:
        env = np.interp(
            np.linspace(0.0, 1.0, sample_count),
            np.linspace(0.0, 1.0, env.size),
            env,
        ).astype(np.float32)
    return tuple(float(x) for x in _normalize(env))


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    arr = arr - float(arr.min())
    peak = float(arr.max())
    if peak > 1e-9:
        arr = arr / peak
    norm = float(np.linalg.norm(arr))
    if norm > 1e-9:
        arr = arr / norm
    return arr.astype(np.float32, copy=False)


__all__ = ["align_shape_to_reference", "compare_shape_similarity", "read_mono_audio_slice", "audio_shape_preview"]
