"""
Find Similar review service for interactive event matching.
Exists so the review/training flow is application state, not Qt widget state.
Connects timeline presentation events to lightweight local ranking and saved review artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import numpy as np

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_comparison_service import (
    TimbreFingerprintSettings,
    build_timbre_fingerprint_preview,
    compare_timbre_fingerprint_similarity,
)
from echozero.application.timeline.event_similarity_audio import audio_shape_preview
from echozero.application.timeline.event_similarity_mini_model import (
    ensure_find_similar_models_dir,
    load_timbre_mini_model,
)
from echozero.application.timeline.models import EventRef

REVIEW_MODEL_SCHEMA = "echozero.find-similar-review-model.v1"
TOP_CANDIDATE_LIMIT = 50
REVIEW_CANDIDATE_LIMIT = 15


class ReviewLabel(Enum):
    """Human review state for one candidate event."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class FindSimilarReviewPhase(Enum):
    """Release flow phase for a Find Similar review session."""

    CHOOSE_EXAMPLES = "choose_examples"
    MODEL_RESULTS = "model_results"


@dataclass(frozen=True, slots=True)
class FindSimilarMatchProfile:
    """Live match profile produced by the current review labels."""

    positive_centroid: tuple[float, ...] = ()
    negative_centroid: tuple[float, ...] = ()
    positive_count: int = 0
    negative_count: int = 0
    labeled_count: int = 0
    confidence_threshold: float = 0.90
    selected_event_refs: tuple[EventRef, ...] = ()
    previous_selected_event_refs: tuple[EventRef, ...] = ()
    selection_delta_count: int = 0
    readiness_reason: str = "Review matches until the selection stabilizes."
    can_select_similar: bool = False


@dataclass(frozen=True, slots=True)
class FindSimilarSelection:
    """Final selected similar events from a reviewed match profile."""

    event_refs: tuple[EventRef, ...]
    reviewed_event_refs: tuple[EventRef, ...]
    positive_event_refs: tuple[EventRef, ...]
    negative_event_refs: tuple[EventRef, ...]
    confidence_threshold: float
    match_profile: FindSimilarMatchProfile


@dataclass(frozen=True, slots=True)
class FindSimilarCandidate:
    """One timeline event available to the interactive Find Similar flow."""

    event_ref: EventRef
    label: str
    start_seconds: float
    end_seconds: float
    audio_path: str | None
    timeline_index: int
    embedding: tuple[float, ...] = ()
    preview_shape: tuple[float, ...] = ()
    initial_score: float = 0.0
    model_score: float | None = None
    passes_confidence: bool = True
    score: float = 0.0
    review_label: ReviewLabel = ReviewLabel.UNKNOWN
    is_anchor: bool = False


@dataclass(frozen=True, slots=True)
class FindSimilarReviewSession:
    """Current reviewed candidate set and ranker state for one Find Similar dialog."""

    anchor_ref: EventRef
    scope_mode: str
    candidates: tuple[FindSimilarCandidate, ...]
    phase: FindSimilarReviewPhase = FindSimilarReviewPhase.CHOOSE_EXAMPLES
    confidence_threshold: float = 0.90
    match_profile: FindSimilarMatchProfile = field(default_factory=FindSimilarMatchProfile)
    positive_count: int = 0
    negative_count: int = 0
    skipped_count: int = 0
    next_candidate_ref: EventRef | None = None

    @property
    def ranked_candidates(self) -> tuple[FindSimilarCandidate, ...]:
        """Return non-anchor candidates ordered by confidence for review."""

        return tuple(
            sorted(
                (candidate for candidate in self.candidates if not candidate.is_anchor),
                key=lambda candidate: (
                    candidate.score,
                    -candidate.timeline_index,
                ),
                reverse=True,
            )
        )

    @property
    def top_candidates(self) -> tuple[FindSimilarCandidate, ...]:
        """Return the auto-populated top candidate lane."""

        return self.ranked_candidates[:TOP_CANDIDATE_LIMIT]

    @property
    def review_candidates(self) -> tuple[FindSimilarCandidate, ...]:
        """Return the first-ranked candidates intended for human approval."""

        return self.top_candidates[:REVIEW_CANDIDATE_LIMIT]

    @property
    def matched_candidates(self) -> tuple[FindSimilarCandidate, ...]:
        """Return anchor and confirmed positives in timeline order."""

        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE
        )

    @property
    def seed_event_refs(self) -> tuple[EventRef, ...]:
        """Return the positive training seeds in timeline order."""

        return tuple(candidate.event_ref for candidate in self.matched_candidates)

    @property
    def negative_event_refs(self) -> tuple[EventRef, ...]:
        """Return rejected examples in timeline order."""

        return tuple(
            candidate.event_ref
            for candidate in self.candidates
            if candidate.review_label == ReviewLabel.NEGATIVE
        )

    @property
    def model_result_event_refs(self) -> tuple[EventRef, ...]:
        """Return model-selected event refs in timeline order."""

        return self.match_profile.selected_event_refs

    @property
    def model_result_candidates(self) -> tuple[FindSimilarCandidate, ...]:
        """Return model-selected candidates in timeline order."""

        selected_keys = {
            _event_ref_key(event_ref) for event_ref in self.match_profile.selected_event_refs
        }
        return tuple(
            candidate
            for candidate in self.candidates
            if _event_ref_key(candidate.event_ref) in selected_keys
        )

    @property
    def valid_candidate_count(self) -> int:
        """Return candidates with usable audio embeddings."""

        return sum(1 for candidate in self.candidates if candidate.embedding)

    @property
    def required_seed_count(self) -> int:
        """Return the release seed count needed before training."""

        valid_count = sum(1 for candidate in self.review_candidates if candidate.embedding) + int(
            any(candidate.is_anchor and candidate.embedding for candidate in self.candidates)
        )
        if valid_count < 5:
            return max(2, valid_count)
        return 5

    @property
    def required_example_count(self) -> int:
        """Return positive examples needed before final selection."""

        return self.required_seed_count

    @property
    def has_enough_profile_evidence(self) -> bool:
        """Return whether the reviewed candidates have enough match evidence."""

        review_ref_keys = {
            _event_ref_key(candidate.event_ref) for candidate in self.review_candidates
        }
        review_seed_count = sum(
            1
            for candidate in self.candidates
            if candidate.is_anchor
            or (
                candidate.review_label == ReviewLabel.POSITIVE
                and _event_ref_key(candidate.event_ref) in review_ref_keys
            )
        )
        review_valid_count = sum(
            1 for candidate in self.review_candidates if candidate.embedding
        ) + int(any(candidate.is_anchor and candidate.embedding for candidate in self.candidates))
        if self.required_seed_count < 5:
            return review_seed_count >= 2 and review_seed_count >= review_valid_count
        return review_seed_count >= 5

    @property
    def can_select_similar(self) -> bool:
        """Return whether the live profile is stable enough to select events."""

        return self.match_profile.can_select_similar

    @property
    def can_train(self) -> bool:
        """Compatibility alias for older callers."""

        return self.can_select_similar


class FindSimilarReviewService:
    """Builds and reranks interactive Find Similar review sessions."""

    def start_session(
        self,
        *,
        presentation: TimelinePresentation,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        scope_mode: str = "song",
        seed_event_refs: Sequence[EventRef] | None = None,
    ) -> FindSimilarReviewSession:
        """Create an initial timeline-ordered review session from a selected event."""

        anchor_ref = EventRef(layer_id, take_id, event_id)
        candidates = _candidate_records(
            presentation,
            anchor_ref=anchor_ref,
            scope_mode=_coerce_scope_mode(scope_mode),
        )
        seed_ref_keys = {
            _event_ref_key(event_ref)
            for event_ref in (seed_event_refs or ())
            if not _same_event_ref(event_ref, anchor_ref)
        }
        if seed_ref_keys:
            candidates = tuple(
                (
                    replace(candidate, review_label=ReviewLabel.POSITIVE)
                    if _event_ref_key(candidate.event_ref) in seed_ref_keys
                    else candidate
                )
                for candidate in candidates
            )
        return self._rerank(
            FindSimilarReviewSession(
                anchor_ref=anchor_ref,
                scope_mode=_coerce_scope_mode(scope_mode),
                candidates=candidates,
            )
        )

    def mark_candidate(
        self,
        session: FindSimilarReviewSession,
        event_ref: EventRef,
        label: ReviewLabel,
    ) -> FindSimilarReviewSession:
        """Apply a human label and return a reranked session."""

        candidates = tuple(
            (
                replace(candidate, review_label=label)
                if _same_event_ref(candidate.event_ref, event_ref) and not candidate.is_anchor
                else candidate
            )
            for candidate in session.candidates
        )
        return self._rerank(replace(session, candidates=candidates))

    def set_confidence_threshold(
        self,
        session: FindSimilarReviewSession,
        threshold: float,
    ) -> FindSimilarReviewSession:
        """Apply a confidence threshold while preserving candidate order."""

        return self._rerank(
            replace(session, confidence_threshold=_clamp01(threshold)),
        )

    def train_review_model(
        self,
        session: FindSimilarReviewSession,
        *,
        min_positive_seeds: int = 5,
    ) -> FindSimilarReviewSession:
        """Compatibility shim: recompute the live match profile."""

        del min_positive_seeds
        return self._rerank(session)

    def select_similar_events(
        self,
        session: FindSimilarReviewSession,
    ) -> FindSimilarSelection:
        """Return final similar-event refs from the current stable match profile."""

        refreshed = self._rerank(session)
        event_refs = (
            refreshed.match_profile.selected_event_refs
            if refreshed.can_select_similar
            else refreshed.seed_event_refs
        )
        reviewed_event_refs = tuple(
            candidate.event_ref
            for candidate in refreshed.candidates
            if candidate.is_anchor or candidate.review_label != ReviewLabel.UNKNOWN
        )
        return FindSimilarSelection(
            event_refs=event_refs,
            reviewed_event_refs=reviewed_event_refs,
            positive_event_refs=refreshed.seed_event_refs,
            negative_event_refs=refreshed.negative_event_refs,
            confidence_threshold=refreshed.confidence_threshold,
            match_profile=refreshed.match_profile,
        )

    def model_result_candidates(
        self,
        session: FindSimilarReviewSession,
    ) -> tuple[FindSimilarCandidate, ...]:
        """Return model-selected candidates for the current session."""

        return session.model_result_candidates

    def candidate_refs_above_confidence(
        self,
        session: FindSimilarReviewSession,
    ) -> tuple[EventRef, ...]:
        """Return selected refs above threshold in timeline order."""

        return tuple(candidate.event_ref for candidate in session.model_result_candidates)

    def _rerank(self, session: FindSimilarReviewSession) -> FindSimilarReviewSession:
        anchor = next((candidate for candidate in session.candidates if candidate.is_anchor), None)
        anchor_embedding = _vector(anchor.embedding if anchor is not None else ())
        positives = [
            _vector(candidate.embedding)
            for candidate in session.candidates
            if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE
        ]
        negatives = [
            _vector(candidate.embedding)
            for candidate in session.candidates
            if candidate.review_label == ReviewLabel.NEGATIVE
        ]
        positive_centroid = _centroid(positives) if positives else anchor_embedding
        negative_centroid = _centroid(negatives) if negatives else np.asarray((), dtype=np.float32)

        reranked: list[FindSimilarCandidate] = []
        for candidate in session.candidates:
            score = _score_candidate(
                candidate,
                anchor_embedding=anchor_embedding,
                positive_centroid=positive_centroid,
                negative_centroid=negative_centroid,
            )
            reranked.append(
                replace(
                    candidate,
                    initial_score=score,
                    model_score=score,
                    score=float(score),
                    passes_confidence=_candidate_passes_threshold(
                        candidate,
                        threshold=session.confidence_threshold,
                        score=float(score),
                    ),
                )
            )

        interim = replace(session, candidates=tuple(reranked))
        next_ref = _next_best_candidate_ref(interim.review_candidates)
        selected_event_refs = _selected_event_refs(
            tuple(reranked),
            threshold=session.confidence_threshold,
        )
        previous_selected_refs = session.match_profile.selected_event_refs
        delta_count = _selection_delta_count(previous_selected_refs, selected_event_refs)
        profile = FindSimilarMatchProfile(
            positive_centroid=tuple(float(value) for value in positive_centroid),
            negative_centroid=tuple(float(value) for value in negative_centroid),
            positive_count=sum(
                1
                for candidate in reranked
                if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE
            ),
            negative_count=sum(
                1 for candidate in reranked if candidate.review_label == ReviewLabel.NEGATIVE
            ),
            labeled_count=sum(
                1
                for candidate in reranked
                if candidate.is_anchor or candidate.review_label != ReviewLabel.UNKNOWN
            ),
            confidence_threshold=session.confidence_threshold,
            selected_event_refs=selected_event_refs,
            previous_selected_event_refs=previous_selected_refs,
            selection_delta_count=delta_count,
            readiness_reason="",
            can_select_similar=False,
        )
        readiness_reason, can_select = _profile_readiness(interim, profile)
        profile = replace(
            profile,
            readiness_reason=readiness_reason,
            can_select_similar=can_select,
        )
        return replace(
            interim,
            candidates=tuple(reranked),
            match_profile=profile,
            positive_count=sum(
                1 for candidate in reranked if candidate.review_label == ReviewLabel.POSITIVE
            ),
            negative_count=sum(
                1 for candidate in reranked if candidate.review_label == ReviewLabel.NEGATIVE
            ),
            skipped_count=sum(
                1 for candidate in reranked if candidate.review_label == ReviewLabel.SKIPPED
            ),
            next_candidate_ref=next_ref,
        )


def save_find_similar_review_model(
    session: FindSimilarReviewSession,
    *,
    output_dir: Path | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Save a local review model artifact containing positive and negative review state."""

    timestamp = created_at or datetime.now(timezone.utc)
    root = output_dir or ensure_find_similar_models_dir()
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / f"review-{timestamp.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.json"
    positives = [
        candidate
        for candidate in session.candidates
        if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE
    ]
    negatives = [
        candidate
        for candidate in session.candidates
        if candidate.review_label == ReviewLabel.NEGATIVE
    ]
    payload = {
        "schema": REVIEW_MODEL_SCHEMA,
        "model_id": uuid4().hex,
        "created_at": timestamp.isoformat(),
        "anchor_event_ref": _event_ref_payload(session.anchor_ref),
        "scope_mode": session.scope_mode,
        "settings": {"sample_count": 64, "padding_ms": 20.0},
        "confidence_threshold": session.confidence_threshold,
        "can_select_similar": session.match_profile.can_select_similar,
        "readiness_reason": session.match_profile.readiness_reason,
        "positive_event_refs": [
            _event_ref_payload(candidate.event_ref) for candidate in positives
        ],
        "negative_event_refs": [
            _event_ref_payload(candidate.event_ref) for candidate in negatives
        ],
        "positive_centroid": _centroid_payload(positives),
        "negative_centroid": _centroid_payload(negatives),
    }
    model_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return model_path


def score_saved_review_model(
    artifact_path: Path,
    candidate_embedding: tuple[float, ...],
) -> float:
    """Score a candidate embedding with a saved v2 review model or legacy mini-model."""

    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if payload.get("schema") == REVIEW_MODEL_SCHEMA:
        positive_centroid = _vector(payload.get("positive_centroid") or ())
        negative_centroid = _vector(payload.get("negative_centroid") or ())
        candidate = _vector(candidate_embedding)
        return _clamp01(
            _cosine(positive_centroid, candidate) - 0.45 * _cosine(negative_centroid, candidate)
        )
    legacy_payload = load_timbre_mini_model(Path(artifact_path))
    return compare_timbre_fingerprint_similarity(
        tuple(float(value) for value in legacy_payload["centroid"]),
        candidate_embedding,
    )


def _candidate_records(
    presentation: TimelinePresentation,
    *,
    anchor_ref: EventRef,
    scope_mode: str,
) -> tuple[FindSimilarCandidate, ...]:
    layer = _find_layer(presentation, anchor_ref.layer_id)
    if layer is None:
        return ()
    layers = _scope_layers(presentation, layer=layer, scope_mode=scope_mode)
    records: list[FindSimilarCandidate] = []
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    all_events = _iter_events(layers)
    if scope_mode == "take":
        all_events = tuple(
            row
            for row in all_events
            if row[0].layer_id == anchor_ref.layer_id and row[1] == anchor_ref.take_id
        )
    elif scope_mode == "selected_layers_main":
        all_events = tuple(
            row
            for row in all_events
            if row[3] is None
            or row[3].is_main
            or row[1] == (row[0].main_take_id or TakeId("main"))
        )
    for timeline_index, (candidate_layer, take_id, event, take) in enumerate(all_events):
        event_ref = EventRef(candidate_layer.layer_id, take_id, event.event_id)
        audio_path = _event_audio_path(candidate_layer, take)
        embedding = _embedding_for_event(
            audio_path=audio_path,
            start_seconds=float(event.start),
            end_seconds=float(event.end),
            audio_cache=audio_cache,
        )
        records.append(
            FindSimilarCandidate(
                event_ref=event_ref,
                label=event.label or str(event.event_id),
                start_seconds=float(event.start),
                end_seconds=float(event.end),
                audio_path=audio_path,
                timeline_index=timeline_index,
                embedding=embedding,
                preview_shape=_preview_shape(embedding),
                is_anchor=_same_event_ref(event_ref, anchor_ref),
            )
        )
    return tuple(records)


def _scope_layers(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    scope_mode: str,
) -> tuple[LayerPresentation, ...]:
    if scope_mode == "song":
        return tuple(presentation.layers)
    if scope_mode == "selected_layers_main":
        selected_ids = set(presentation.selected_layer_ids or [layer.layer_id])
        return tuple(
            candidate for candidate in presentation.layers if candidate.layer_id in selected_ids
        )
    return (layer,)


def _iter_events(
    layers: Sequence[LayerPresentation],
) -> tuple[tuple[LayerPresentation, TakeId, EventPresentation, TakeLanePresentation | None], ...]:
    rows: list[
        tuple[LayerPresentation, TakeId, EventPresentation, TakeLanePresentation | None]
    ] = []
    for layer in layers:
        main_take_id = layer.main_take_id or TakeId("main")
        rows.extend((layer, main_take_id, event, None) for event in layer.events)
        rows.extend(
            (layer, take.take_id, event, take)
            for take in layer.takes
            if take.take_id != main_take_id
            for event in take.events
        )
    return tuple(rows)


def _event_audio_path(layer: LayerPresentation, take: TakeLanePresentation | None) -> str | None:
    if take is not None and take.source_audio_path:
        return take.source_audio_path
    return layer.source_audio_path


def _embedding_for_event(
    *,
    audio_path: str | None,
    start_seconds: float,
    end_seconds: float,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[float, ...]:
    if not audio_path or not Path(audio_path).exists():
        return ()
    end = end_seconds if end_seconds > start_seconds else start_seconds + 0.12
    return (
        build_timbre_fingerprint_preview(
            audio_path=audio_path,
            start_seconds=start_seconds,
            end_seconds=end,
            settings=TimbreFingerprintSettings(sample_count=64, padding_ms=20.0),
            audio_cache=audio_cache,
        )
        or ()
    )


def _preview_shape(embedding: tuple[float, ...]) -> tuple[float, ...]:
    if not embedding:
        return ()
    return audio_shape_preview(np.asarray(embedding, dtype=np.float32), sample_count=32)


def _score_candidate(
    candidate: FindSimilarCandidate,
    *,
    anchor_embedding: np.ndarray,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
) -> float:
    if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE:
        return 1.0
    if candidate.review_label == ReviewLabel.NEGATIVE:
        return 0.0
    candidate_embedding = _vector(candidate.embedding)
    if candidate_embedding.size == 0:
        return 0.0
    score = 0.72 * _cosine(positive_centroid, candidate_embedding)
    score += 0.28 * _cosine(anchor_embedding, candidate_embedding)
    if negative_centroid.size:
        score -= 0.42 * _cosine(negative_centroid, candidate_embedding)
    if candidate.review_label == ReviewLabel.SKIPPED:
        score *= 0.85
    return _clamp01(score)


def _trained_model_score(
    candidate: FindSimilarCandidate,
    *,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
) -> float:
    if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE:
        return 1.0
    if candidate.review_label == ReviewLabel.NEGATIVE:
        return 0.0
    candidate_embedding = _vector(candidate.embedding)
    if candidate_embedding.size == 0:
        return 0.0
    score = _cosine(positive_centroid, candidate_embedding)
    if negative_centroid.size:
        score -= 0.50 * _cosine(negative_centroid, candidate_embedding)
    if candidate.review_label == ReviewLabel.SKIPPED:
        score *= 0.75
    return _clamp01(score)


def _candidate_passes_threshold(
    candidate: FindSimilarCandidate,
    *,
    threshold: float,
    score: float,
) -> bool:
    if candidate.is_anchor or candidate.review_label == ReviewLabel.POSITIVE:
        return True
    if candidate.review_label in {ReviewLabel.NEGATIVE, ReviewLabel.SKIPPED}:
        return False
    return score >= threshold


def _selected_event_refs(
    candidates: tuple[FindSimilarCandidate, ...],
    *,
    threshold: float,
) -> tuple[EventRef, ...]:
    return tuple(
        candidate.event_ref
        for candidate in sorted(candidates, key=lambda row: row.timeline_index)
        if candidate.is_anchor
        or candidate.review_label == ReviewLabel.POSITIVE
        or (
            candidate.review_label == ReviewLabel.UNKNOWN
            and candidate.embedding
            and candidate.score >= threshold
        )
    )


def _selection_delta_count(
    previous_refs: tuple[EventRef, ...],
    current_refs: tuple[EventRef, ...],
) -> int:
    previous_keys = {_event_ref_key(event_ref) for event_ref in previous_refs}
    current_keys = {_event_ref_key(event_ref) for event_ref in current_refs}
    return len(previous_keys.symmetric_difference(current_keys))


def _profile_readiness(
    session: FindSimilarReviewSession,
    profile: FindSimilarMatchProfile,
) -> tuple[str, bool]:
    required = session.required_example_count
    if not session.has_enough_profile_evidence:
        remaining = max(0, required - profile.positive_count)
        if remaining > 0:
            return f"Need {remaining} more match{'es' if remaining != 1 else ''}.", False
        return "Review matches until the selection stabilizes.", False
    if profile.selection_delta_count > 1:
        return "Selection still changing; review the next candidate.", False
    result_count = len(profile.selected_event_refs)
    if result_count == 0:
        return "Review matches until the selection stabilizes.", False
    return f"Ready to select {result_count} similar events.", True


def _next_best_candidate_ref(candidates: tuple[FindSimilarCandidate, ...]) -> EventRef | None:
    unknowns = [
        candidate
        for candidate in candidates
        if not candidate.is_anchor and candidate.review_label == ReviewLabel.UNKNOWN
    ]
    if not unknowns:
        return None
    return max(
        unknowns, key=lambda candidate: (candidate.score, -candidate.timeline_index)
    ).event_ref


def _centroid(vectors: Sequence[np.ndarray]) -> np.ndarray:
    valid = [vector for vector in vectors if vector.size]
    if not valid:
        return np.asarray((), dtype=np.float32)
    return _unit_array(np.mean(np.vstack(valid), axis=0).astype(np.float32))


def _centroid_payload(candidates: Sequence[FindSimilarCandidate]) -> list[float]:
    centroid = _centroid([_vector(candidate.embedding) for candidate in candidates])
    return [float(value) for value in centroid]


def _vector(values: object) -> np.ndarray:
    if values is None:
        return np.asarray((), dtype=np.float32)
    arr = np.asarray(tuple(float(value) for value in values), dtype=np.float32).reshape(-1)
    return _unit_array(arr)


def _unit_array(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm > 1e-9:
        return (arr / norm).astype(np.float32)
    return arr.astype(np.float32, copy=False)


def _cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.size == 0 or candidate.size == 0:
        return 0.0
    if reference.size != candidate.size:
        candidate = np.interp(
            np.linspace(0.0, 1.0, reference.size),
            np.linspace(0.0, 1.0, candidate.size),
            candidate,
        ).astype(np.float32)
        candidate = _unit_array(candidate)
    return max(0.0, min(1.0, float(np.dot(reference, candidate))))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _find_layer(presentation: TimelinePresentation, layer_id: LayerId) -> LayerPresentation | None:
    return next((layer for layer in presentation.layers if layer.layer_id == layer_id), None)


def _same_event_ref(left: EventRef, right: EventRef) -> bool:
    return (
        left.layer_id == right.layer_id
        and left.take_id == right.take_id
        and left.event_id == right.event_id
    )


def _event_ref_key(event_ref: EventRef) -> tuple[str, str, str]:
    return (str(event_ref.layer_id), str(event_ref.take_id), str(event_ref.event_id))


def _event_ref_payload(event_ref: EventRef) -> dict[str, str]:
    return {
        "layer_id": str(event_ref.layer_id),
        "take_id": str(event_ref.take_id),
        "event_id": str(event_ref.event_id),
    }


def _coerce_scope_mode(scope_mode: str) -> str:
    valid_modes = {"song", "take", "layer", "selected_layers_main"}
    return scope_mode if scope_mode in valid_modes else "song"


__all__ = [
    "FindSimilarCandidate",
    "FindSimilarMatchProfile",
    "FindSimilarReviewPhase",
    "FindSimilarReviewService",
    "FindSimilarReviewSession",
    "FindSimilarSelection",
    "REVIEW_MODEL_SCHEMA",
    "REVIEW_CANDIDATE_LIMIT",
    "ReviewLabel",
    "TOP_CANDIDATE_LIMIT",
    "save_find_similar_review_model",
    "score_saved_review_model",
]
