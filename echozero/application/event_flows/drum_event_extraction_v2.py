"""
Drum event extraction v2: explicit per-label lane contracts.
Exists because selected drum labels need typed source, candidate, classifier, and audition lanes.
Connects pipeline outputs to persistence/playback drafts without relying on shared fallback state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from echozero.application.event_flows.drum_events import normalize_drum_event_labels


@dataclass(frozen=True, slots=True)
class DrumEventExtractionRequest:
    """Selected labels and per-label source audio for one drum event extraction run."""

    labels: tuple[str, ...] = ("kick", "snare")
    source_audio_by_label: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", normalize_drum_event_labels(self.labels))
        object.__setattr__(
            self,
            "source_audio_by_label",
            MappingProxyType(_normalize_source_mapping(self.source_audio_by_label)),
        )


@dataclass(frozen=True, slots=True)
class DrumEventLabelLane:
    """One explicit drum-label lane from source audio through promoted events."""

    label: str
    source_audio_ref: str | None = None
    candidate_events: tuple[object, ...] = ()
    model_output_events: tuple[object, ...] = ()
    promoted_events: tuple[object, ...] = ()
    audition_source_ref: str | None = None

    def __post_init__(self) -> None:
        labels = normalize_drum_event_labels((self.label,))
        normalized_label = labels[0] if labels else ""
        object.__setattr__(self, "label", normalized_label)
        source_audio_ref = _normalize_optional_ref(self.source_audio_ref)
        audition_source_ref = _normalize_optional_ref(self.audition_source_ref) or source_audio_ref
        object.__setattr__(self, "source_audio_ref", source_audio_ref)
        object.__setattr__(self, "audition_source_ref", audition_source_ref)
        object.__setattr__(self, "candidate_events", tuple(self.candidate_events))
        object.__setattr__(self, "model_output_events", tuple(self.model_output_events))
        object.__setattr__(self, "promoted_events", tuple(self.promoted_events))


@dataclass(frozen=True, slots=True)
class DrumEventExtractionResult:
    """Typed result for all requested drum-label lanes."""

    lanes: tuple[DrumEventLabelLane, ...]

    def lane_for(self, label: object) -> DrumEventLabelLane | None:
        """Return the lane for one normalized label, if that label was requested."""

        labels = normalize_drum_event_labels((label,))
        if not labels:
            return None
        normalized_label = labels[0]
        return next((lane for lane in self.lanes if lane.label == normalized_label), None)


@dataclass(frozen=True, slots=True)
class DrumEventLayerTakeDraft:
    """Persistence/playback draft for one generated event layer or take."""

    label: str
    layer_name: str
    take_name: str
    events: tuple[object, ...]
    source_audio_ref: str | None
    playback_source_ref: str | None


def build_drum_event_extraction_result(
    request: DrumEventExtractionRequest,
    *,
    candidate_events_by_label: Mapping[str, Sequence[object]] | None = None,
    model_output_events_by_label: Mapping[str, Sequence[object]] | None = None,
    promoted_events_by_label: Mapping[str, Sequence[object]] | None = None,
    source_audio_by_label: Mapping[str, str] | None = None,
    audition_source_by_label: Mapping[str, str] | None = None,
) -> DrumEventExtractionResult:
    """Build explicit per-label lanes without borrowing candidates or sources across labels."""

    candidate_events = _normalize_event_mapping(candidate_events_by_label)
    model_output_events = _normalize_event_mapping(model_output_events_by_label)
    explicit_promoted_events = _normalize_event_mapping(promoted_events_by_label)
    request_sources = dict(request.source_audio_by_label)
    override_sources = _normalize_source_mapping(source_audio_by_label or {})
    audition_sources = _normalize_source_mapping(audition_source_by_label or {})

    lanes: list[DrumEventLabelLane] = []
    for label in request.labels:
        model_events = model_output_events.get(label, ())
        promoted_events = (
            explicit_promoted_events[label]
            if label in explicit_promoted_events
            else _promoted_events_from_model_output(model_events)
        )
        lanes.append(
            DrumEventLabelLane(
                label=label,
                source_audio_ref=override_sources.get(label) or request_sources.get(label),
                candidate_events=candidate_events.get(label, ()),
                model_output_events=model_events,
                promoted_events=promoted_events,
                audition_source_ref=audition_sources.get(label),
            )
        )
    return DrumEventExtractionResult(lanes=tuple(lanes))


def build_drum_event_layer_take_drafts(
    result: DrumEventExtractionResult,
    *,
    take_name: str = "Extracted Events",
) -> tuple[DrumEventLayerTakeDraft, ...]:
    """Build per-label persistence/playback drafts using each lane's audition source."""

    return tuple(
        DrumEventLayerTakeDraft(
            label=lane.label,
            layer_name=f"{lane.label.title()} Events",
            take_name=take_name,
            events=lane.promoted_events,
            source_audio_ref=lane.source_audio_ref,
            playback_source_ref=lane.audition_source_ref,
        )
        for lane in result.lanes
    )


def _normalize_source_mapping(values: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_label, raw_ref in values.items():
        labels = normalize_drum_event_labels((raw_label,))
        if not labels:
            continue
        source_ref = _normalize_optional_ref(raw_ref)
        if source_ref:
            normalized[labels[0]] = source_ref
    return normalized


def _normalize_event_mapping(
    values: Mapping[str, Sequence[object]] | None,
) -> dict[str, tuple[object, ...]]:
    normalized: dict[str, tuple[object, ...]] = {}
    for raw_label, events in (values or {}).items():
        labels = normalize_drum_event_labels((raw_label,))
        if labels:
            normalized[labels[0]] = tuple(events)
    return normalized


def _promoted_events_from_model_output(events: Sequence[object]) -> tuple[object, ...]:
    return tuple(event for event in events if _is_promoted_event(event))


def _is_promoted_event(event: object) -> bool:
    promotion_state = _promotion_state(event)
    if promotion_state == "demoted":
        return False
    return True


def _promotion_state(event: object) -> str:
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, Mapping):
        return "promoted"
    detection = metadata.get("detection")
    if isinstance(detection, Mapping):
        state = str(detection.get("promotion_state", "")).strip().lower()
        if state:
            return state
        if detection.get("threshold_passed") is False:
            return "demoted"
    review = metadata.get("review")
    if isinstance(review, Mapping):
        state = str(review.get("promotion_state", "")).strip().lower()
        if state:
            return state
    state = str(metadata.get("promotion_state", "")).strip().lower()
    return state or "promoted"


def _normalize_optional_ref(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
