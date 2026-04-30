"""Canonical review-event state helpers for app and Foundry flows.
Exists because timeline fix and phone review must read and mutate one event-review schema.
Connects runtime events, persisted domain events, and review queues through shared metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from echozero.foundry.domain.review import (
    ReviewDecision,
    ReviewDecisionKind,
    ReviewOutcome,
    ReviewSurface,
    build_review_decision,
    build_review_provenance,
)

_VALID_PROMOTION_STATES = {"promoted", "demoted"}
_VALID_REVIEW_STATES = {"unreviewed", "corrected", "signed_off"}


@dataclass(frozen=True, slots=True)
class CanonicalEventReviewState:
    """Normalized review state derived from one canonical event metadata payload."""

    origin_kind: str
    promotion_state: str
    review_state: str
    review_outcome: ReviewOutcome
    decision_kind: ReviewDecisionKind | None
    original_label: str | None
    corrected_label: str | None
    review_note: str | None
    reviewed_at: datetime | None
    original_start_ms: float | None
    original_end_ms: float | None
    corrected_start_ms: float | None
    corrected_end_ms: float | None
    created_event_ref: str | None
    surface: ReviewSurface | None
    workflow: str | None
    operator_action: str | None


def normalize_review_label(value: object, *, default: str = "event") -> str:
    """Return the canonical lower-case review label for one event class value."""

    text = str(value or "").strip().lower()
    return text or default


def canonical_review_state(
    *,
    origin: object,
    metadata: Mapping[str, Any] | None,
) -> CanonicalEventReviewState:
    """Read one canonical event-review state from persisted metadata."""

    metadata_payload = dict(metadata or {})
    review_payload = _review_payload(metadata_payload)
    origin_kind = _origin_kind(origin)
    promotion_state = _promotion_state(review_payload, metadata_payload)
    review_state = _review_state(review_payload, metadata_payload)
    review_outcome = _review_outcome(review_payload, metadata_payload, review_state=review_state)
    corrected_label = _first_text(review_payload, "corrected_label", "correctedLabel")
    original_label = _first_text(review_payload, "original_label", "originalLabel")
    original_start_ms = _first_float(review_payload, "original_start_ms", "originalStartMs")
    original_end_ms = _first_float(review_payload, "original_end_ms", "originalEndMs")
    corrected_start_ms = _first_float(review_payload, "corrected_start_ms", "correctedStartMs")
    corrected_end_ms = _first_float(review_payload, "corrected_end_ms", "correctedEndMs")
    created_event_ref = _first_text(review_payload, "created_event_ref", "createdEventRef")
    decision_kind = _decision_kind(
        review_payload=review_payload,
        promotion_state=promotion_state,
        review_state=review_state,
        origin_kind=origin_kind,
        original_label=original_label,
        corrected_label=corrected_label,
        original_start_ms=original_start_ms,
        original_end_ms=original_end_ms,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
        created_event_ref=created_event_ref,
    )
    return CanonicalEventReviewState(
        origin_kind=origin_kind,
        promotion_state=promotion_state,
        review_state=review_state,
        review_outcome=review_outcome,
        decision_kind=decision_kind,
        original_label=original_label,
        corrected_label=corrected_label,
        review_note=_first_text(review_payload, "review_note", "reviewNote"),
        reviewed_at=_first_datetime(review_payload, "reviewed_at", "reviewedAt"),
        original_start_ms=original_start_ms,
        original_end_ms=original_end_ms,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
        created_event_ref=created_event_ref,
        surface=_surface(review_payload),
        workflow=_first_text(review_payload, "workflow"),
        operator_action=_first_text(review_payload, "operator_action", "operatorAction"),
    )


def updated_review_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    promotion_state: str,
    review_state: str,
    review_outcome: ReviewOutcome | str,
    decision_kind: ReviewDecisionKind | str | None,
    original_label: str | None,
    corrected_label: str | None,
    review_note: str | None,
    reviewed_at: datetime | str | None,
    original_start_ms: float | None = None,
    original_end_ms: float | None = None,
    corrected_start_ms: float | None = None,
    corrected_end_ms: float | None = None,
    created_event_ref: str | None = None,
    surface: ReviewSurface | str | None = None,
    workflow: str | None = None,
    operator_action: str | None = None,
) -> dict[str, Any]:
    """Return event metadata updated with one canonical review-state payload."""

    next_metadata = dict(metadata or {})
    review_payload = _review_payload(next_metadata)
    normalized_outcome = ReviewOutcome(str(review_outcome))
    normalized_decision = None if decision_kind is None else ReviewDecisionKind(str(decision_kind))
    review_payload["schema"] = "echozero.event_review.v1"
    review_payload["promotion_state"] = _validated_promotion_state(promotion_state)
    review_payload["review_state"] = _validated_review_state(review_state)
    review_payload["review_outcome"] = normalized_outcome.value
    _assign_optional(review_payload, "decision_kind", None if normalized_decision is None else normalized_decision.value)
    _assign_optional(review_payload, "original_label", _clean_text(original_label))
    _assign_optional(review_payload, "corrected_label", _clean_text(corrected_label))
    _assign_optional(review_payload, "review_note", _clean_text(review_note))
    _assign_optional(review_payload, "reviewed_at", _iso_text(reviewed_at))
    _assign_optional(review_payload, "original_start_ms", original_start_ms)
    _assign_optional(review_payload, "original_end_ms", original_end_ms)
    _assign_optional(review_payload, "corrected_start_ms", corrected_start_ms)
    _assign_optional(review_payload, "corrected_end_ms", corrected_end_ms)
    _assign_optional(review_payload, "created_event_ref", _clean_text(created_event_ref))
    _assign_optional(review_payload, "surface", None if surface is None else ReviewSurface(str(surface)).value)
    _assign_optional(review_payload, "workflow", _clean_text(workflow))
    _assign_optional(review_payload, "operator_action", _clean_text(operator_action))
    next_metadata["review"] = review_payload
    # Review semantics live under metadata["review"]; legacy top-level duplicates are removed
    # to keep processor outputs and app/foundry mutation lanes cleanly separated.
    next_metadata.pop("promotion_state", None)
    next_metadata.pop("review_state", None)
    next_metadata.pop("review_outcome", None)
    next_metadata.pop("review_decision_kind", None)
    return next_metadata


def build_review_decision_from_state(
    *,
    state: CanonicalEventReviewState,
    source_provenance: Mapping[str, Any],
) -> ReviewDecision | None:
    """Build one typed review decision from canonical event-review metadata."""

    if state.review_outcome == ReviewOutcome.PENDING or state.decision_kind is None:
        return None
    return build_review_decision(
        state.review_outcome,
        corrected_label=state.corrected_label if state.review_outcome == ReviewOutcome.INCORRECT else None,
        review_note=state.review_note if state.review_outcome == ReviewOutcome.INCORRECT else None,
        decision_kind=state.decision_kind,
        original_start_ms=state.original_start_ms if state.review_outcome == ReviewOutcome.INCORRECT else None,
        original_end_ms=state.original_end_ms if state.review_outcome == ReviewOutcome.INCORRECT else None,
        corrected_start_ms=state.corrected_start_ms if state.review_outcome == ReviewOutcome.INCORRECT else None,
        corrected_end_ms=state.corrected_end_ms if state.review_outcome == ReviewOutcome.INCORRECT else None,
        created_event_ref=state.created_event_ref if state.review_outcome == ReviewOutcome.INCORRECT else None,
        provenance=build_review_provenance(
            source_provenance,
            surface=state.surface or ReviewSurface.PHONE_REVIEW,
            workflow=state.workflow or "canonical_event_review",
            operator_action=state.operator_action,
        ),
    )


def _review_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = metadata.get("review")
    return dict(payload) if isinstance(payload, dict) else {}


def _origin_kind(origin: object) -> str:
    normalized = str(origin or "").strip().lower()
    if normalized in {"manual_added", "manual", "user", "ma3_pull"}:
        return "manual_added"
    return "model_detected"


def _promotion_state(review_payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    review_value = _clean_text(review_payload.get("promotion_state"))
    if review_value in _VALID_PROMOTION_STATES:
        return review_value
    legacy_value = _clean_text(metadata.get("promotion_state"))
    if legacy_value in _VALID_PROMOTION_STATES:
        return legacy_value
    detection_payload = metadata.get("detection")
    if isinstance(detection_payload, Mapping):
        detection_value = _clean_text(detection_payload.get("promotion_state"))
        if detection_value in _VALID_PROMOTION_STATES:
            return detection_value
        threshold_passed = _coerce_optional_bool(detection_payload.get("threshold_passed"))
        if threshold_passed is not None:
            return "promoted" if threshold_passed else "demoted"
    return "promoted"


def _review_state(review_payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    review_value = _clean_text(review_payload.get("review_state"))
    if review_value in _VALID_REVIEW_STATES:
        return review_value
    legacy_value = _clean_text(metadata.get("review_state"))
    if legacy_value in _VALID_REVIEW_STATES:
        return legacy_value
    return "unreviewed"


def _review_outcome(
    review_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    review_state: str,
) -> ReviewOutcome:
    for candidate in (
        _clean_text(review_payload.get("review_outcome")),
        _clean_text(metadata.get("review_outcome")),
    ):
        if candidate is None:
            continue
        try:
            return ReviewOutcome(candidate)
        except ValueError:
            continue
    if review_state == "signed_off":
        return ReviewOutcome.CORRECT
    if review_state == "corrected":
        return ReviewOutcome.INCORRECT
    return ReviewOutcome.PENDING


def _decision_kind(
    *,
    review_payload: Mapping[str, Any],
    promotion_state: str,
    review_state: str,
    origin_kind: str,
    original_label: str | None,
    corrected_label: str | None,
    original_start_ms: float | None,
    original_end_ms: float | None,
    corrected_start_ms: float | None,
    corrected_end_ms: float | None,
    created_event_ref: str | None,
) -> ReviewDecisionKind | None:
    explicit = _clean_text(review_payload.get("decision_kind"))
    if explicit is not None:
        try:
            return ReviewDecisionKind(explicit)
        except ValueError:
            pass
    if review_state == "signed_off":
        return ReviewDecisionKind.VERIFIED
    if review_state != "corrected":
        return None
    if promotion_state == "demoted":
        return ReviewDecisionKind.REJECTED
    if created_event_ref is not None or origin_kind == "manual_added":
        return ReviewDecisionKind.MISSED_EVENT_ADDED
    if _timing_changed(
        original_start_ms=original_start_ms,
        original_end_ms=original_end_ms,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
    ):
        return ReviewDecisionKind.BOUNDARY_CORRECTED
    if corrected_label is not None and corrected_label != original_label:
        return ReviewDecisionKind.RELABELED
    return ReviewDecisionKind.RELABELED


def _timing_changed(
    *,
    original_start_ms: float | None,
    original_end_ms: float | None,
    corrected_start_ms: float | None,
    corrected_end_ms: float | None,
) -> bool:
    if (
        original_start_ms is None
        or original_end_ms is None
        or corrected_start_ms is None
        or corrected_end_ms is None
    ):
        return False
    return (
        float(original_start_ms) != float(corrected_start_ms)
        or float(original_end_ms) != float(corrected_end_ms)
    )


def _surface(review_payload: Mapping[str, Any]) -> ReviewSurface | None:
    text = _clean_text(review_payload.get("surface"))
    if text is None:
        return None
    try:
        return ReviewSurface(text)
    except ValueError:
        return None


def _validated_promotion_state(value: str) -> str:
    normalized = _clean_text(value)
    if normalized not in _VALID_PROMOTION_STATES:
        raise ValueError(f"Unsupported promotion_state: {value!r}")
    return normalized


def _validated_review_state(value: str) -> str:
    normalized = _clean_text(value)
    if normalized not in _VALID_REVIEW_STATES:
        raise ValueError(f"Unsupported review_state: {value!r}")
    return normalized


def _assign_optional(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        mapping.pop(key, None)
        return
    mapping[key] = value


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iso_text(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return _clean_text(value)


def _first_text(mapping: object, *keys: str) -> str | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = _clean_text(mapping.get(key))
        if value is not None:
            return value
    return None


def _first_float(mapping: object, *keys: str) -> float | None:
    text = _first_text(mapping, *keys)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_datetime(mapping: object, *keys: str) -> datetime | None:
    text = _first_text(mapping, *keys)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None
