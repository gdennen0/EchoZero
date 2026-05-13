"""
Event similarity shape extraction for timeline events.
Exists because find-similar now compares one normalized event graph instead of mixed scoring modes.
Connects event audio slices to simplified XY envelopes and percentage-based shape comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

_FEATURE_SAMPLE_RATE = 22050
_MIN_EVENT_DURATION_SECONDS = 0.08


@dataclass(slots=True)
class ShapeNormalizationSettings:
    """Controls how one event clip is normalized into a simplified graph."""

    sample_count: int = 64
    smoothing_ms: float = 12.0
    padding_ms: float = 20.0

    def __post_init__(self) -> None:
        self.sample_count = max(16, min(256, int(self.sample_count)))
        self.smoothing_ms = max(0.0, min(250.0, float(self.smoothing_ms)))
        self.padding_ms = max(0.0, min(250.0, float(self.padding_ms)))


@dataclass(slots=True)
class EventShapeBundle:
    """Normalized graph data for one event clip."""

    normalized_samples: tuple[float, ...]


def load_event_shape_bundle(
    *,
    audio_path: str,
    start_seconds: float,
    end_seconds: float,
    settings: ShapeNormalizationSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> EventShapeBundle | None:
    """Load one event slice and convert it into a normalized graph shape."""

    audio, sample_rate = _load_audio(audio_path, audio_cache)
    if audio.size == 0:
        return None
    segment = _slice_audio(
        audio=audio,
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        padding_seconds=settings.padding_ms / 1000.0,
    )
    if segment is None:
        return None
    samples = _build_normalized_shape(segment, sample_rate, settings)
    return EventShapeBundle(normalized_samples=tuple(float(value) for value in samples))


def compare_shape_similarity(
    anchor_samples: tuple[float, ...],
    candidate_samples: tuple[float, ...],
) -> float:
    """Return a 0..1 similarity score for two normalized graph shapes."""

    anchor, candidate = _coerce_comparable_curves(anchor_samples, candidate_samples)
    if anchor is None or candidate is None:
        return 0.0
    aligned_candidate = align_shape_to_reference(
        tuple(float(value) for value in anchor),
        tuple(float(value) for value in candidate),
    )
    candidate_array = np.asarray(aligned_candidate, dtype=np.float32)
    rmse = float(np.sqrt(np.mean(np.square(anchor - candidate_array))))
    return max(0.0, min(1.0, 1.0 - rmse))


def align_shape_to_reference(
    anchor_samples: tuple[float, ...],
    candidate_samples: tuple[float, ...],
) -> tuple[float, ...]:
    """Shift one candidate curve against the anchor to find the best visual alignment."""

    anchor, candidate = _coerce_comparable_curves(anchor_samples, candidate_samples)
    if anchor is None or candidate is None:
        return candidate_samples
    max_shift = max(1, anchor.size // 5)
    best_shift = 0
    best_curve = candidate
    best_error = _curve_rmse(anchor, candidate)
    for shift in _candidate_shifts(anchor, candidate, max_shift=max_shift):
        shifted = _shift_curve(candidate, shift)
        error = _curve_rmse(anchor, shifted)
        if error < best_error:
            best_error = error
            best_shift = shift
            best_curve = shifted
    if best_shift == 0:
        return tuple(float(value) for value in best_curve)
    return tuple(float(value) for value in best_curve)


def _coerce_comparable_curves(
    anchor_samples: tuple[float, ...],
    candidate_samples: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not anchor_samples or not candidate_samples:
        return None, None
    anchor = np.asarray(anchor_samples, dtype=np.float32)
    candidate = np.asarray(candidate_samples, dtype=np.float32)
    if anchor.shape != candidate.shape:
        target_size = min(anchor.size, candidate.size)
        if target_size <= 0:
            return None, None
        anchor = _resample_curve(anchor, target_size)
        candidate = _resample_curve(candidate, target_size)
    return anchor, candidate


def _load_audio(
    audio_path: str,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    cached = audio_cache.get(audio_path)
    if cached is not None:
        return cached
    resolved = str(Path(audio_path).expanduser())
    audio, sample_rate = librosa.load(resolved, sr=_FEATURE_SAMPLE_RATE, mono=True)
    cached = (audio.astype(np.float32, copy=False), int(sample_rate))
    audio_cache[audio_path] = cached
    return cached


def _slice_audio(
    *,
    audio: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    padding_seconds: float,
) -> np.ndarray | None:
    padded_start = max(0.0, float(start_seconds) - padding_seconds)
    padded_end = max(
        padded_start + _MIN_EVENT_DURATION_SECONDS,
        float(end_seconds) + padding_seconds,
    )
    start_sample = int(round(padded_start * sample_rate))
    end_sample = min(len(audio), int(round(padded_end * sample_rate)))
    if end_sample <= start_sample:
        return None
    segment = audio[start_sample:end_sample].astype(np.float32, copy=False)
    return segment if segment.size > 0 else None


def _build_normalized_shape(
    segment: np.ndarray,
    sample_rate: int,
    settings: ShapeNormalizationSettings,
) -> np.ndarray:
    envelope = np.abs(segment).astype(np.float32, copy=False)
    smoothed = _smooth_envelope(
        envelope,
        sample_rate=sample_rate,
        smoothing_ms=settings.smoothing_ms,
    )
    resampled = _resample_curve(smoothed, settings.sample_count)
    return _normalize_curve(resampled)


def _smooth_envelope(
    envelope: np.ndarray,
    *,
    sample_rate: int,
    smoothing_ms: float,
) -> np.ndarray:
    window_size = int(round((smoothing_ms / 1000.0) * sample_rate))
    if window_size <= 1:
        return envelope
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(envelope, kernel, mode="same").astype(np.float32, copy=False)


def _resample_curve(samples: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size <= 1:
        return np.zeros(target_size, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, samples.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_size, dtype=np.float32)
    return np.interp(target_x, source_x, samples).astype(np.float32, copy=False)


def _normalize_curve(samples: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(samples.astype(np.float32, copy=False), copy=False)
    if finite.size == 0:
        return finite
    minimum = float(np.min(finite))
    shifted = finite - minimum
    peak = float(np.max(shifted))
    if peak <= 1e-8:
        return np.zeros_like(shifted, dtype=np.float32)
    return shifted / peak


def _candidate_shifts(
    anchor: np.ndarray,
    candidate: np.ndarray,
    *,
    max_shift: int,
) -> list[int]:
    shifts = {0}
    anchor_peak = int(np.argmax(anchor)) if anchor.size > 0 else 0
    candidate_peak = int(np.argmax(candidate)) if candidate.size > 0 else 0
    shifts.add(int(np.clip(anchor_peak - candidate_peak, -max_shift, max_shift)))

    anchor_center = _energy_center(anchor)
    candidate_center = _energy_center(candidate)
    shifts.add(
        int(
            np.clip(
                int(round(anchor_center - candidate_center)),
                -max_shift,
                max_shift,
            )
        )
    )

    for shift in range(-max_shift, max_shift + 1):
        shifts.add(shift)
    return sorted(shifts)


def _energy_center(samples: np.ndarray) -> float:
    if samples.size <= 0:
        return 0.0
    weights = np.maximum(samples, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-8:
        return float(samples.size - 1) / 2.0
    indices = np.arange(samples.size, dtype=np.float32)
    return float(np.sum(indices * weights) / total)


def _curve_rmse(reference: np.ndarray, comparison: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(reference - comparison))))


def _shift_curve(samples: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0 or samples.size == 0:
        return samples
    shifted = np.zeros_like(samples)
    if shift > 0:
        shifted[shift:] = samples[:-shift]
        return shifted
    shifted[:shift] = samples[-shift:]
    return shifted
