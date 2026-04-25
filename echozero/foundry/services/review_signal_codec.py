"""Review signal codec for imported payloads and phone-review API snapshots.
Exists because review-session service should orchestrate session flow, not nested decision JSON shaping.
Connects the hardened review contract to imported review files and the mobile snapshot API.
"""

from __future__ import annotations

from typing import Any

from echozero.foundry.domain.review import (
    ReviewDecision,
    ReviewOutcome,
    ReviewSurface,
    ReviewTrainingEligibility,
    build_review_decision,
    build_review_provenance,
)


def coerce_review_decision(
    row: dict,
    *,
    outcome: ReviewOutcome,
    corrected_label: str | None,
    review_note: str | None,
    source_provenance: dict[str, object],
) -> ReviewDecision | None:
    """Build one typed review decision from imported JSON-compatible payloads."""
    payload = row.get("review_decision", row.get("reviewDecision"))
    if not isinstance(payload, dict) or not payload.get("kind"):
        return build_review_decision(
            outcome,
            corrected_label=corrected_label,
            review_note=review_note,
            provenance=build_review_provenance(
                source_provenance,
                surface=ReviewSurface.IMPORTED_SESSION,
                workflow="import_review_payload",
            ),
        )
    provenance_payload = payload.get("provenance")
    training_payload = payload.get("training_eligibility", payload.get("trainingEligibility"))
    return build_review_decision(
        outcome,
        corrected_label=_first_text(payload, "corrected_label", "correctedLabel") or corrected_label,
        review_note=_first_text(payload, "review_note", "reviewNote") or review_note,
        decision_kind=_first_text(payload, "kind"),
        original_start_ms=_first_float(payload, "original_start_ms", "originalStartMs"),
        original_end_ms=_first_float(payload, "original_end_ms", "originalEndMs"),
        corrected_start_ms=_first_float(payload, "corrected_start_ms", "correctedStartMs"),
        corrected_end_ms=_first_float(payload, "corrected_end_ms", "correctedEndMs"),
        created_event_ref=_first_text(payload, "created_event_ref", "createdEventRef"),
        provenance=build_review_provenance(
            source_provenance,
            surface=_resolve_surface(provenance_payload),
            workflow=_resolve_workflow(provenance_payload),
            operator_action=_resolve_operator_action(provenance_payload),
            payload=provenance_payload if isinstance(provenance_payload, dict) else None,
        ),
        training_eligibility=_coerce_training_eligibility(training_payload),
    )


def serialize_review_decision(decision: ReviewDecision | None) -> dict[str, object] | None:
    """Return one API-ready review decision payload."""
    if decision is None:
        return None
    return {
        "kind": decision.kind.value,
        "correctedLabel": decision.corrected_label,
        "reviewNote": decision.review_note,
        "originalStartMs": decision.original_start_ms,
        "originalEndMs": decision.original_end_ms,
        "correctedStartMs": decision.corrected_start_ms,
        "correctedEndMs": decision.corrected_end_ms,
        "createdEventRef": decision.created_event_ref,
        "provenance": (
            {
                "surface": decision.provenance.surface.value,
                "workflow": decision.provenance.workflow,
                "operatorAction": decision.provenance.operator_action,
                "queueSessionRef": decision.provenance.queue_session_ref,
                "projectRef": decision.provenance.project_ref,
                "songRef": decision.provenance.song_ref,
                "versionRef": decision.provenance.version_ref,
                "layerRef": decision.provenance.layer_ref,
                "eventRef": decision.provenance.event_ref,
                "sourceEventRef": decision.provenance.source_event_ref,
                "modelRef": decision.provenance.model_ref,
            }
            if decision.provenance is not None
            else None
        ),
        "trainingEligibility": {
            "allowsPositiveSignal": decision.training_eligibility.allows_positive_signal,
            "allowsNegativeSignal": decision.training_eligibility.allows_negative_signal,
            "requiresMaterializedCorrection": decision.training_eligibility.requires_materialized_correction,
        },
    }


def _first_text(mapping: object, *keys: str) -> str | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_float(mapping: object, *keys: str) -> float | None:
    text = _first_text(mapping, *keys)
    if text is None:
        return None
    return float(text)


def _resolve_surface(payload: object) -> ReviewSurface:
    if not isinstance(payload, dict):
        return ReviewSurface.IMPORTED_SESSION
    surface = _first_text(payload, "surface")
    if surface is None:
        return ReviewSurface.IMPORTED_SESSION
    return ReviewSurface(surface)


def _resolve_workflow(payload: object) -> str:
    if not isinstance(payload, dict):
        return "import_review_payload"
    return _first_text(payload, "workflow") or "import_review_payload"


def _resolve_operator_action(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _first_text(payload, "operator_action", "operatorAction")


def _coerce_training_eligibility(payload: object) -> ReviewTrainingEligibility | None:
    if not isinstance(payload, dict):
        return None
    return ReviewTrainingEligibility(
        allows_positive_signal=bool(
            payload.get("allows_positive_signal", payload.get("allowsPositiveSignal", False))
        ),
        allows_negative_signal=bool(
            payload.get("allows_negative_signal", payload.get("allowsNegativeSignal", False))
        ),
        requires_materialized_correction=bool(
            payload.get(
                "requires_materialized_correction",
                payload.get("requiresMaterializedCorrection", False),
            )
        ),
    )
