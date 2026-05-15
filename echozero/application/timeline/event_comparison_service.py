"""Event comparison service for selecting similar timeline events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_similarity_audio import (
    audio_shape_preview,
    compare_shape_similarity,
    read_mono_audio_slice,
)
from echozero.application.timeline.models import Event, EventRef, Layer, Take


@dataclass(frozen=True, slots=True)
class TimbreFingerprintSettings:
    sample_count: int = 64
    padding_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class EventComparisonCandidateRecord:
    layer_id: LayerId
    take_id: TakeId
    event: Event
    layer: Layer
    take: Take


@dataclass(frozen=True, slots=True)
class EventComparisonRequest:
    anchor_event_id: EventId
    comparison_mode: str = "shape_envelope"
    similarity_threshold: float = 0.85
    comparison_settings: object | None = None


class EventComparisonService:
    """Select events whose audio comparison fingerprint matches an anchor."""

    def select_matching_event_refs(
        self,
        *,
        anchor_layer: Layer,
        anchor_take: Take,
        candidate_records: list[EventComparisonCandidateRecord],
        request: EventComparisonRequest,
    ) -> tuple[EventRef, ...]:
        anchor = next((event for event in anchor_take.events if event.id == request.anchor_event_id), None)
        if anchor is None:
            return ()
        settings = request.comparison_settings
        if not isinstance(settings, TimbreFingerprintSettings):
            settings = TimbreFingerprintSettings()
        mode = (request.comparison_mode or "shape_envelope").strip().lower()
        cache: dict[str, tuple[np.ndarray, int]] = {}
        anchor_preview = _event_preview(anchor_take, anchor, settings=settings, mode=mode, cache=cache)
        if anchor_preview is None:
            return (EventRef(anchor_layer.id, anchor_take.id, anchor.id),)

        matches: list[EventRef] = []
        threshold = max(0.0, min(1.0, float(request.similarity_threshold)))
        for record in candidate_records:
            preview = _event_preview(record.take, record.event, settings=settings, mode=mode, cache=cache)
            score = 1.0 if record.event.id == anchor.id and record.take.id == anchor_take.id else 0.0
            if preview is not None:
                score = (
                    compare_timbre_fingerprint_similarity(anchor_preview, preview)
                    if mode == "timbre_fingerprint"
                    else compare_shape_similarity(anchor_preview, preview)
                )
            if score >= threshold:
                matches.append(EventRef(record.layer_id, record.take_id, record.event.id))
        if not matches:
            matches.append(EventRef(anchor_layer.id, anchor_take.id, anchor.id))
        return tuple(matches)


def build_timbre_fingerprint_preview(
    *,
    audio_path: str | None,
    start_seconds: float,
    end_seconds: float,
    settings: TimbreFingerprintSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[float, ...] | None:
    audio = _read_cached_slice(
        audio_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        settings=settings,
        audio_cache=audio_cache,
    )
    if audio is None:
        return None
    samples, sample_rate = audio
    return _timbre_fingerprint(samples, sample_rate=sample_rate, sample_count=max(8, int(settings.sample_count)))


def compare_timbre_fingerprint_similarity(
    reference: tuple[float, ...], candidate: tuple[float, ...]
) -> float:
    ref = _unit_vector(reference)
    cand = _unit_vector(candidate)
    if ref.size == 0 or cand.size == 0:
        return 0.0
    if cand.size != ref.size:
        cand = np.interp(
            np.linspace(0.0, 1.0, ref.size),
            np.linspace(0.0, 1.0, cand.size),
            cand,
        ).astype(np.float32)
        cand = _unit_array(cand)
    return max(0.0, min(1.0, float(np.dot(ref, cand))))


def _event_preview(
    take: Take,
    event: Event,
    *,
    settings: TimbreFingerprintSettings,
    mode: str,
    cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[float, ...] | None:
    audio = _read_cached_slice(
        _take_audio_path(take),
        start_seconds=float(event.start),
        end_seconds=float(event.end),
        settings=settings,
        audio_cache=cache,
    )
    if audio is None:
        return None
    samples, sample_rate = audio
    if mode == "timbre_fingerprint":
        return _timbre_fingerprint(samples, sample_rate=sample_rate, sample_count=max(8, int(settings.sample_count)))
    return audio_shape_preview(samples, sample_count=max(8, int(settings.sample_count)))


def _read_cached_slice(
    audio_path: str | None,
    *,
    start_seconds: float,
    end_seconds: float,
    settings: TimbreFingerprintSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[np.ndarray, int] | None:
    if not audio_path:
        return None
    padding = max(0.0, float(settings.padding_ms) / 1000.0)
    cache_key = f"{audio_path}|{max(0.0, float(start_seconds) - padding):.6f}|{max(float(start_seconds), float(end_seconds) + padding):.6f}"
    if audio_cache is not None and cache_key in audio_cache:
        return audio_cache[cache_key]
    sliced = read_mono_audio_slice(
        audio_path,
        start_seconds=max(0.0, float(start_seconds) - padding),
        end_seconds=max(float(start_seconds), float(end_seconds) + padding),
    )
    if sliced is not None and audio_cache is not None:
        audio_cache[cache_key] = sliced
    return sliced


def _timbre_fingerprint(
    audio: np.ndarray,
    *,
    sample_rate: int,
    sample_count: int,
) -> tuple[float, ...]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ()
    envelope = np.asarray(audio_shape_preview(arr, sample_count=sample_count), dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(arr * np.hanning(arr.size))).astype(np.float32)
    if spectrum.size <= 1 or float(np.sum(spectrum)) <= 1e-9:
        spectral = np.zeros(sample_count, dtype=np.float32)
    else:
        freqs = np.fft.rfftfreq(arr.size, d=1.0 / max(1, int(sample_rate))).astype(np.float32)
        # Compress frequency energy into fixed bands. Log-ish interpolation keeps one-shot
        # low/high timbres distinct while staying cheap and dependency-free.
        band_x = np.linspace(0.0, 1.0, sample_count)
        source_x = np.linspace(0.0, 1.0, spectrum.size)
        spectral = np.interp(band_x, source_x, spectrum).astype(np.float32)
        centroid = float(np.sum(freqs * spectrum) / max(1e-9, float(np.sum(spectrum))))
        spectral = _unit_array(spectral) * 0.75
        centroid_feature = np.full(sample_count // 4 or 1, centroid / max(1.0, sample_rate / 2.0), dtype=np.float32)
        return tuple(float(x) for x in _unit_array(np.concatenate((envelope, spectral, centroid_feature))))
    return tuple(float(x) for x in _unit_array(np.concatenate((envelope, spectral))))


def _unit_vector(value: tuple[float, ...]) -> np.ndarray:
    return _unit_array(np.asarray(value, dtype=np.float32).reshape(-1))


def _unit_array(arr: np.ndarray) -> np.ndarray:
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


def _take_audio_path(take: Take) -> str | None:
    source_ref = getattr(take, "source_content_ref", None)
    locator = getattr(source_ref, "locator", None)
    if locator and Path(str(locator)).exists():
        return str(locator)
    return None


__all__ = [
    "EventComparisonCandidateRecord",
    "EventComparisonRequest",
    "EventComparisonService",
    "TimbreFingerprintSettings",
    "build_timbre_fingerprint_preview",
    "compare_timbre_fingerprint_similarity",
]
