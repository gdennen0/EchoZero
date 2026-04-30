"""Review domain models for phone-first Foundry verification sessions.
Exists because model outputs and detections need durable human review state.
Connects review import, persistence, and HTTP serving through typed records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping


class ReviewPolarity(StrEnum):
    """Marks whether an item is being reviewed as a positive or negative example."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ReviewOutcome(StrEnum):
    """Tracks whether an item is still pending review or has been judged by a human."""

    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"


class ReviewDecisionKind(StrEnum):
    """Captures the reusable correction shape for downstream consumers."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    RELABELED = "relabeled"
    BOUNDARY_CORRECTED = "boundary_corrected"
    MISSED_EVENT_ADDED = "missed_event_added"


class ReviewSurface(StrEnum):
    """Names the operator surface that emitted one review decision."""

    PHONE_REVIEW = "phone_review"
    TIMELINE_FIX_MODE = "timeline_fix_mode"
    IMPORTED_SESSION = "imported_session"


@dataclass(slots=True)
class ReviewDecisionProvenance:
    """Explicit workflow and source linkage for one durable review decision."""

    surface: ReviewSurface
    workflow: str | None = None
    operator_action: str | None = None
    queue_session_ref: str | None = None
    project_ref: str | None = None
    song_ref: str | None = None
    version_ref: str | None = None
    layer_ref: str | None = None
    event_ref: str | None = None
    source_event_ref: str | None = None
    model_ref: str | None = None


@dataclass(slots=True)
class ReviewTrainingEligibility:
    """Declares which downstream training lanes may consume this review."""

    allows_positive_signal: bool = False
    allows_negative_signal: bool = False
    requires_materialized_correction: bool = False


@dataclass(slots=True)
class ReviewDecision:
    """Shared human-review decision details for one reviewed item."""

    kind: ReviewDecisionKind
    corrected_label: str | None = None
    review_note: str | None = None
    original_start_ms: float | None = None
    original_end_ms: float | None = None
    corrected_start_ms: float | None = None
    corrected_end_ms: float | None = None
    created_event_ref: str | None = None
    provenance: ReviewDecisionProvenance | None = None
    training_eligibility: ReviewTrainingEligibility = field(default_factory=ReviewTrainingEligibility)


@dataclass(slots=True)
class ReviewItem:
    """A single audio-backed verification target inside a review session."""

    item_id: str
    audio_path: str
    predicted_label: str
    target_class: str
    polarity: ReviewPolarity
    score: float | None = None
    source_provenance: dict[str, Any] = field(default_factory=dict)
    review_outcome: ReviewOutcome = ReviewOutcome.PENDING
    review_decision: ReviewDecision | None = None
    corrected_label: str | None = None
    review_note: str | None = None
    reviewed_at: datetime | None = None


@dataclass(slots=True)
class ReviewSession:
    """A persisted queue of review items for fast manual verification."""

    id: str
    name: str
    items: list[ReviewItem] = field(default_factory=list)
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def class_map(self) -> list[str]:
        """List the distinct target classes present in the session."""
        return sorted({item.target_class for item in self.items})


@dataclass(slots=True)
class ReviewSignal:
    """Canonical durable review record for one explicitly committed decision."""

    id: str
    session_id: str
    item_id: str
    audio_path: str
    predicted_label: str
    target_class: str
    polarity: ReviewPolarity
    score: float | None = None
    source_provenance: dict[str, Any] = field(default_factory=dict)
    review_outcome: ReviewOutcome = ReviewOutcome.PENDING
    review_decision: ReviewDecision | None = None
    corrected_label: str | None = None
    review_note: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ReviewCommitContext:
    """Shared context required to commit one explicit review signal."""

    session_id: str
    session_name: str
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_review_session(self, *, reviewed_at: datetime) -> ReviewSession:
        """Build a lightweight review session view for downstream writeback and datasets."""
        return ReviewSession(
            id=self.session_id,
            name=self.session_name,
            items=[],
            source_ref=self.source_ref,
            metadata=dict(self.metadata),
            created_at=reviewed_at,
            updated_at=reviewed_at,
        )


@dataclass(slots=True)
class ExplicitReviewCommit:
    """Reusable payload for one explicit review commit across producer surfaces."""

    item_id: str
    audio_path: str
    predicted_label: str
    target_class: str
    polarity: ReviewPolarity
    score: float | None = None
    source_provenance: dict[str, Any] = field(default_factory=dict)
    review_outcome: ReviewOutcome = ReviewOutcome.PENDING
    review_decision: ReviewDecision | None = None
    corrected_label: str | None = None
    review_note: str | None = None
    reviewed_at: datetime | None = None
    signal_id: str | None = None


@dataclass(slots=True)
class ReviewCommitCommand:
    """Canonical review-commit command shared across producer surfaces."""

    context: ReviewCommitContext
    commit: ExplicitReviewCommit
    apply_project_writeback: bool = True


def build_review_decision(
    outcome: ReviewOutcome,
    *,
    corrected_label: str | None,
    review_note: str | None,
    decision_kind: ReviewDecisionKind | str | None = None,
    original_start_ms: float | None = None,
    original_end_ms: float | None = None,
    corrected_start_ms: float | None = None,
    corrected_end_ms: float | None = None,
    created_event_ref: str | None = None,
    provenance: ReviewDecisionProvenance | None = None,
    training_eligibility: ReviewTrainingEligibility | None = None,
) -> ReviewDecision | None:
    """Normalize review fields into one reusable decision payload."""
    if outcome == ReviewOutcome.PENDING:
        return None
    normalized_kind = _normalize_decision_kind(
        outcome,
        decision_kind=decision_kind,
        corrected_label=corrected_label,
        review_note=review_note,
        original_start_ms=original_start_ms,
        original_end_ms=original_end_ms,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
        created_event_ref=created_event_ref,
    )
    return ReviewDecision(
        kind=normalized_kind,
        corrected_label=corrected_label,
        review_note=review_note,
        original_start_ms=original_start_ms,
        original_end_ms=original_end_ms,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
        created_event_ref=created_event_ref,
        provenance=provenance,
        training_eligibility=training_eligibility or build_training_eligibility(normalized_kind),
    )


def build_review_provenance(
    source_provenance: Mapping[str, Any] | None,
    *,
    surface: ReviewSurface | str = ReviewSurface.PHONE_REVIEW,
    workflow: str | None = "manual_review",
    operator_action: str | None = None,
    queue_session_ref: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> ReviewDecisionProvenance:
    """Merge source provenance with explicit review metadata for one decision."""
    source_payload = source_provenance or {}
    explicit_payload = payload or {}
    normalized_surface = explicit_payload.get("surface", surface)
    return ReviewDecisionProvenance(
        surface=ReviewSurface(str(normalized_surface)),
        workflow=_coerce_mapping_text(explicit_payload, "workflow") or workflow,
        operator_action=(
            _coerce_mapping_text(explicit_payload, "operator_action", "operatorAction")
            or operator_action
        ),
        queue_session_ref=(
            _coerce_mapping_text(explicit_payload, "queue_session_ref", "queueSessionRef")
            or queue_session_ref
        ),
        project_ref=_coerce_mapping_text(explicit_payload, "project_ref", "projectRef")
        or _coerce_mapping_text(source_payload, "project_ref", "projectRef"),
        song_ref=_coerce_mapping_text(explicit_payload, "song_ref", "songRef")
        or _coerce_mapping_text(source_payload, "song_ref", "songRef"),
        version_ref=_coerce_mapping_text(explicit_payload, "version_ref", "versionRef")
        or _coerce_mapping_text(source_payload, "version_ref", "versionRef"),
        layer_ref=_coerce_mapping_text(explicit_payload, "layer_ref", "layerRef")
        or _coerce_mapping_text(source_payload, "layer_ref", "layerRef"),
        event_ref=_coerce_mapping_text(explicit_payload, "event_ref", "eventRef")
        or _coerce_mapping_text(source_payload, "event_ref", "eventRef"),
        source_event_ref=_coerce_mapping_text(explicit_payload, "source_event_ref", "sourceEventRef")
        or _coerce_mapping_text(source_payload, "source_event_ref", "sourceEventRef"),
        model_ref=_coerce_mapping_text(explicit_payload, "model_ref", "modelRef")
        or _coerce_mapping_text(explicit_payload, "bundle_ref", "bundleRef")
        or _coerce_mapping_text(source_payload, "model_ref", "modelRef")
        or _coerce_mapping_text(source_payload, "bundle_ref", "bundleRef"),
    )


def build_training_eligibility(kind: ReviewDecisionKind) -> ReviewTrainingEligibility:
    """Return the downstream training semantics for one review decision kind."""
    if kind == ReviewDecisionKind.VERIFIED:
        return ReviewTrainingEligibility(allows_positive_signal=True)
    if kind == ReviewDecisionKind.REJECTED:
        return ReviewTrainingEligibility(allows_negative_signal=True)
    if kind == ReviewDecisionKind.RELABELED:
        return ReviewTrainingEligibility(
            allows_positive_signal=True,
            allows_negative_signal=True,
        )
    if kind == ReviewDecisionKind.BOUNDARY_CORRECTED:
        return ReviewTrainingEligibility(
            allows_positive_signal=True,
            allows_negative_signal=True,
            requires_materialized_correction=True,
        )
    return ReviewTrainingEligibility(allows_positive_signal=True)


def _normalize_decision_kind(
    outcome: ReviewOutcome,
    *,
    decision_kind: ReviewDecisionKind | str | None,
    corrected_label: str | None,
    review_note: str | None,
    original_start_ms: float | None,
    original_end_ms: float | None,
    corrected_start_ms: float | None,
    corrected_end_ms: float | None,
    created_event_ref: str | None,
) -> ReviewDecisionKind:
    if decision_kind is not None:
        return ReviewDecisionKind(str(decision_kind))
    if outcome == ReviewOutcome.CORRECT:
        return ReviewDecisionKind.VERIFIED
    if created_event_ref:
        return ReviewDecisionKind.MISSED_EVENT_ADDED
    if any(value is not None for value in (original_start_ms, original_end_ms, corrected_start_ms, corrected_end_ms)):
        return ReviewDecisionKind.BOUNDARY_CORRECTED
    if corrected_label or review_note:
        return ReviewDecisionKind.RELABELED
    return ReviewDecisionKind.REJECTED


def _coerce_mapping_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
