"""
Event sequence similarity for timeline event layers.
Exists so selected rhythmic hit patterns can be found elsewhere without audio analysis.
Connects selected event refs to scored timing/class windows for the timeline app contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from echozero.application.timeline.models import Event, EventRef


@dataclass(frozen=True, slots=True)
class EventSequenceRecord:
    """One event with its stable timeline reference."""

    event_ref: EventRef
    event: Event


@dataclass(frozen=True, slots=True)
class EventSequenceMatch:
    """One candidate sequence window with scoring diagnostics."""

    event_refs: tuple[EventRef, ...]
    score: float
    timing_score: float
    label_score: float
    length_score: float
    missing_hit_count: int
    status: str


@dataclass(frozen=True, slots=True)
class EventSequenceMatchRequest:
    """Tuning values for selected-event timing sequence search."""

    min_events: int = 3
    strictness: str = "balanced"
    minimum_score: float | None = None
    timing_weight: float = 0.62
    label_weight: float = 0.30
    length_weight: float = 0.08
    use_label_similarity: bool = True
    allow_missing_events: int = 1
    interval_tolerance_seconds: float | None = None
    interval_tolerance_ratio: float | None = None

    def __post_init__(self) -> None:
        strictness = str(self.strictness or "").strip().lower()
        if strictness not in {"loose", "balanced", "strict"}:
            strictness = "balanced"
        object.__setattr__(self, "strictness", strictness)
        profile = _strictness_profile(strictness)
        if self.minimum_score is None:
            object.__setattr__(self, "minimum_score", profile["minimum_score"])
        object.__setattr__(
            self,
            "interval_tolerance_seconds",
            (
                float(self.interval_tolerance_seconds)
                if self.interval_tolerance_seconds is not None
                else profile["tolerance_seconds"]
            ),
        )
        object.__setattr__(
            self,
            "interval_tolerance_ratio",
            (
                float(self.interval_tolerance_ratio)
                if self.interval_tolerance_ratio is not None
                else profile["tolerance_ratio"]
            ),
        )


class EventSequenceSimilarityService:
    """Find event windows with inter-hit timing similar to a selected event sequence."""

    def find_matches(
        self,
        *,
        anchor_records: list[EventSequenceRecord],
        candidate_groups: list[list[EventSequenceRecord]],
        request: EventSequenceMatchRequest | None = None,
    ) -> tuple[EventSequenceMatch, ...]:
        """Return scored sequence windows sorted by confidence."""

        resolved_request = request or EventSequenceMatchRequest()
        anchors = _ordered_records(anchor_records)
        if len(anchors) < max(2, int(resolved_request.min_events)):
            return ()

        anchor_offsets = _relative_offsets(anchors)
        matches: list[EventSequenceMatch] = [
            EventSequenceMatch(
                event_refs=tuple(record.event_ref for record in anchors),
                score=1.0,
                timing_score=1.0,
                label_score=1.0,
                length_score=1.0,
                missing_hit_count=0,
                status="anchor",
            )
        ]
        for candidate_group in candidate_groups:
            matches.extend(
                self._match_group(
                    _ordered_records(candidate_group),
                    anchor_records=anchors,
                    anchor_offsets=anchor_offsets,
                    request=resolved_request,
                )
            )

        threshold = max(0.0, min(1.0, float(resolved_request.minimum_score or 0.0)))
        deduped = _dedupe_matches(match for match in matches if match.score >= threshold)
        return tuple(
            sorted(
                deduped,
                key=lambda match: (
                    match.status != "anchor",
                    -match.score,
                    match.missing_hit_count,
                    -len(match.event_refs),
                    str(match.event_refs[0].layer_id) if match.event_refs else "",
                    str(match.event_refs[0].take_id) if match.event_refs else "",
                    str(match.event_refs[0].event_id) if match.event_refs else "",
                ),
            )
        )

    def select_matching_event_refs(
        self,
        *,
        anchor_records: list[EventSequenceRecord],
        candidate_groups: list[list[EventSequenceRecord]],
        request: EventSequenceMatchRequest | None = None,
    ) -> tuple[EventRef, ...]:
        """Return deduped event refs from matching timing windows."""

        selected_refs: list[EventRef] = []
        for match in self.find_matches(
            anchor_records=anchor_records,
            candidate_groups=candidate_groups,
            request=request,
        ):
            selected_refs.extend(match.event_refs)
        return _dedupe_refs(selected_refs)

    def _match_group(
        self,
        records: list[EventSequenceRecord],
        *,
        anchor_records: list[EventSequenceRecord],
        anchor_offsets: tuple[float, ...],
        request: EventSequenceMatchRequest,
    ) -> list[EventSequenceMatch]:
        if len(records) < max(1, len(anchor_offsets) - request.allow_missing_events):
            return []

        matches: list[EventSequenceMatch] = []
        match_lengths = _candidate_match_lengths(
            len(anchor_offsets),
            request.allow_missing_events,
            min_events=request.min_events,
        )
        for match_length in match_lengths:
            for start_index in range(0, len(records) - match_length + 1):
                window = records[start_index : start_index + match_length]
                match = _score_window(anchor_records, anchor_offsets, window, request=request)
                if match is not None:
                    matches.append(match)
        return matches


def _ordered_records(records: list[EventSequenceRecord]) -> list[EventSequenceRecord]:
    return sorted(records, key=lambda record: (float(record.event.start), str(record.event.id)))


def _relative_offsets(records: list[EventSequenceRecord]) -> tuple[float, ...]:
    if not records:
        return ()
    first_start = float(records[0].event.start)
    return tuple(float(record.event.start) - first_start for record in records)


def _candidate_match_lengths(
    anchor_count: int,
    allowed_missing: int,
    *,
    min_events: int = 3,
) -> tuple[int, ...]:
    min_length = max(int(min_events), anchor_count - max(0, int(allowed_missing)))
    return tuple(range(anchor_count, min_length - 1, -1))


def _score_window(
    anchor_records: list[EventSequenceRecord],
    anchor_offsets: tuple[float, ...],
    records: list[EventSequenceRecord],
    *,
    request: EventSequenceMatchRequest,
) -> EventSequenceMatch | None:
    candidate_offsets = _relative_offsets(records)
    if len(candidate_offsets) == len(anchor_offsets):
        timing_score = _offsets_score(anchor_offsets, candidate_offsets, request=request)
        if timing_score <= 0.0:
            return None
        label_score = _label_score(anchor_records, records, request=request)
        return _build_match(
            records,
            timing_score=timing_score,
            label_score=label_score,
            length_score=1.0,
            missing_hit_count=0,
            request=request,
        )

    missing_count = len(anchor_offsets) - len(candidate_offsets)
    if missing_count < 0 or missing_count > request.allow_missing_events:
        return None
    if len(anchor_offsets) < 4:
        return None
    best_match: EventSequenceMatch | None = None
    for anchor_indexes, anchor_subset in _offset_subsets(anchor_offsets, len(candidate_offsets)):
        timing_score = _offsets_score(anchor_subset, candidate_offsets, request=request)
        if timing_score <= 0.0:
            continue
        subset_records = [anchor_records[index] for index in anchor_indexes]
        label_score = _label_score(subset_records, records, request=request)
        match = _build_match(
            records,
            timing_score=timing_score,
            label_score=label_score,
            length_score=max(0.0, 1.0 - (0.18 * missing_count)),
            missing_hit_count=missing_count,
            request=request,
        )
        if best_match is None or match.score > best_match.score:
            best_match = match
    return best_match


def _offset_subsets(
    offsets: tuple[float, ...],
    target_count: int,
) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
    if target_count >= len(offsets):
        return ((tuple(range(len(offsets))), offsets),)
    subsets: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
    for indexes in combinations(range(len(offsets)), target_count):
        subset = tuple(offsets[index] for index in indexes)
        base = subset[0]
        subsets.append((indexes, tuple(offset - base for offset in subset)))
    return tuple(subsets)


def _offsets_score(
    anchor_offsets: tuple[float, ...],
    candidate_offsets: tuple[float, ...],
    *,
    request: EventSequenceMatchRequest,
) -> float:
    if len(anchor_offsets) != len(candidate_offsets):
        return 0.0
    anchor_intervals = _intervals(anchor_offsets)
    candidate_intervals = _intervals(candidate_offsets)
    if not anchor_intervals:
        return 0.0
    interval_scores: list[float] = []
    for anchor_interval, candidate_interval in zip(anchor_intervals, candidate_intervals):
        tolerance = max(
            float(request.interval_tolerance_seconds),
            abs(float(anchor_interval)) * float(request.interval_tolerance_ratio),
        )
        error = abs(float(anchor_interval) - float(candidate_interval))
        if error > tolerance:
            return 0.0
        interval_scores.append(max(0.0, 1.0 - (error / max(tolerance, 1e-9))))
    return sum(interval_scores) / len(interval_scores)


def _build_match(
    records: list[EventSequenceRecord],
    *,
    timing_score: float,
    label_score: float,
    length_score: float,
    missing_hit_count: int,
    request: EventSequenceMatchRequest,
) -> EventSequenceMatch:
    timing_weight = max(0.0, float(request.timing_weight))
    label_weight = max(0.0, float(request.label_weight if request.use_label_similarity else 0.0))
    length_weight = max(0.0, float(request.length_weight))
    total_weight = max(1e-9, timing_weight + label_weight + length_weight)
    score = (
        timing_score * timing_weight
        + label_score * label_weight
        + length_score * length_weight
    ) / total_weight
    return EventSequenceMatch(
        event_refs=tuple(record.event_ref for record in records),
        score=max(0.0, min(1.0, score)),
        timing_score=max(0.0, min(1.0, timing_score)),
        label_score=max(0.0, min(1.0, label_score)),
        length_score=max(0.0, min(1.0, length_score)),
        missing_hit_count=max(0, int(missing_hit_count)),
        status="missing-hit" if missing_hit_count else "match",
    )


def _label_score(
    anchor_records: list[EventSequenceRecord],
    candidate_records: list[EventSequenceRecord],
    *,
    request: EventSequenceMatchRequest,
) -> float:
    if not request.use_label_similarity:
        return 1.0
    scores = [
        _event_label_similarity(anchor.event, candidate.event)
        for anchor, candidate in zip(anchor_records, candidate_records)
    ]
    return sum(scores) / len(scores) if scores else 1.0


def _event_label_similarity(anchor: Event, candidate: Event) -> float:
    anchor_tokens = _event_label_tokens(anchor)
    candidate_tokens = _event_label_tokens(candidate)
    if not anchor_tokens or not candidate_tokens:
        return 1.0
    if anchor_tokens.intersection(candidate_tokens):
        return 1.0
    return 0.0


def _event_label_tokens(event: Event) -> set[str]:
    event_id_token = str(event.id).strip().lower()
    label_token = str(event.label or "").strip().lower()
    tokens = set()
    if label_token and label_token != event_id_token:
        tokens.add(label_token)
    for key in ("class", "label", "instrument", "kind"):
        value = event.classifications.get(key)
        if value is not None:
            tokens.add(str(value).strip().lower())
    return {token for token in tokens if token and token != "event"}


def _intervals(offsets: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        float(offsets[index]) - float(offsets[index - 1])
        for index in range(1, len(offsets))
    )


def _dedupe_refs(event_refs: list[EventRef]) -> tuple[EventRef, ...]:
    deduped: list[EventRef] = []
    seen: set[tuple[str, str, str]] = set()
    for event_ref in event_refs:
        key = (str(event_ref.layer_id), str(event_ref.take_id), str(event_ref.event_id))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event_ref)
    return tuple(deduped)


def _dedupe_matches(matches: Iterable[EventSequenceMatch]) -> tuple[EventSequenceMatch, ...]:
    deduped: list[EventSequenceMatch] = []
    best_by_key: dict[tuple[tuple[str, str, str], ...], EventSequenceMatch] = {}
    for match in matches:
        if not isinstance(match, EventSequenceMatch):
            continue
        key = tuple(
            (str(event_ref.layer_id), str(event_ref.take_id), str(event_ref.event_id))
            for event_ref in match.event_refs
        )
        previous = best_by_key.get(key)
        if previous is None or match.score > previous.score:
            best_by_key[key] = match
    for match in best_by_key.values():
        deduped.append(match)
    return tuple(deduped)


def _strictness_profile(strictness: str) -> dict[str, float]:
    return {
        "loose": {
            "minimum_score": 0.62,
            "tolerance_seconds": 0.09,
            "tolerance_ratio": 0.22,
        },
        "balanced": {
            "minimum_score": 0.74,
            "tolerance_seconds": 0.05,
            "tolerance_ratio": 0.12,
        },
        "strict": {
            "minimum_score": 0.86,
            "tolerance_seconds": 0.025,
            "tolerance_ratio": 0.07,
        },
    }[strictness]


__all__ = [
    "EventSequenceMatchRequest",
    "EventSequenceMatch",
    "EventSequenceRecord",
    "EventSequenceSimilarityService",
]
