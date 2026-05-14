"""Event comparison service for selecting similar timeline events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """Select events whose audio fingerprint matches the anchor above a threshold."""

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
        cache: dict[str, tuple[np.ndarray, int]] = {}
        anchor_preview = build_timbre_fingerprint_preview(
            audio_path=_take_audio_path(anchor_take),
            start_seconds=float(anchor.start),
            end_seconds=float(anchor.end),
            settings=settings,
            audio_cache=cache,
        )
        if anchor_preview is None:
            return (EventRef(anchor_layer.id, anchor_take.id, anchor.id),)
        matches: list[EventRef] = []
        threshold = max(0.0, min(1.0, float(request.similarity_threshold)))
        for record in candidate_records:
            preview = build_timbre_fingerprint_preview(
                audio_path=_take_audio_path(record.take),
                start_seconds=float(record.event.start),
                end_seconds=float(record.event.end),
                settings=settings,
                audio_cache=cache,
            )
            score = 1.0 if record.event.id == anchor.id and record.take.id == anchor_take.id else 0.0
            if preview is not None:
                score = compare_timbre_fingerprint_similarity(anchor_preview, preview)
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
    if not audio_path:
        return None
    padding = max(0.0, float(settings.padding_ms) / 1000.0)
    sliced = read_mono_audio_slice(
        audio_path,
        start_seconds=max(0.0, float(start_seconds) - padding),
        end_seconds=max(float(start_seconds), float(end_seconds) + padding),
    )
    if sliced is None:
        return None
    audio, _sr = sliced
    return audio_shape_preview(audio, sample_count=max(8, int(settings.sample_count)))


def compare_timbre_fingerprint_similarity(
    reference: tuple[float, ...], candidate: tuple[float, ...]
) -> float:
    return compare_shape_similarity(reference, candidate)


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
