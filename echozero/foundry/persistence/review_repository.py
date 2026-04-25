"""ReviewSessionRepository persists Foundry review queues to JSON state.
Exists because phone review progress must survive server restarts and reloads.
Connects review services to the same foundry/state envelope pattern as other lanes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from echozero.foundry.domain.review import (
    ReviewDecision,
    ReviewDecisionKind,
    ReviewItem,
    ReviewOutcome,
    ReviewPolarity,
    ReviewSession,
    ReviewSurface,
    ReviewTrainingEligibility,
    build_review_provenance,
    build_review_decision,
    build_training_eligibility,
)

from .repositories import _read_state, _write_state


class ReviewSessionRepository:
    """Stores and loads review sessions from the local Foundry state directory."""

    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "review_sessions.json"
        self._schema = "foundry.state.review_sessions.v1"

    def save(self, session: ReviewSession) -> ReviewSession:
        """Persist a review session and return the saved record."""
        rows = _read_state(self._path, self._schema)
        rows[session.id] = {
            "id": session.id,
            "name": session.name,
            "source_ref": session.source_ref,
            "metadata": session.metadata,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "items": [
                {
                    "item_id": item.item_id,
                    "audio_path": item.audio_path,
                    "predicted_label": item.predicted_label,
                    "target_class": item.target_class,
                    "polarity": item.polarity.value,
                    "score": item.score,
                    "source_provenance": item.source_provenance,
                    "review_outcome": item.review_outcome.value,
                    "review_decision": serialize_review_decision_state(
                        item.review_decision
                        or build_review_decision(
                            item.review_outcome,
                            corrected_label=item.corrected_label,
                            review_note=item.review_note,
                            provenance=build_review_provenance(
                                item.source_provenance,
                                queue_session_ref=session.id,
                            ),
                        )
                    ),
                    "corrected_label": item.corrected_label,
                    "review_note": item.review_note,
                    "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
                }
                for item in session.items
            ],
        }
        _write_state(self._path, self._schema, rows)
        return session

    def get(self, session_id: str) -> ReviewSession | None:
        """Load a review session by id when it exists."""
        row = _read_state(self._path, self._schema).get(session_id)
        if row is None:
            return None
        return ReviewSession(
            id=row["id"],
            name=row["name"],
            source_ref=row.get("source_ref"),
            metadata=row.get("metadata", {}),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            items=[
                ReviewItem(
                    item_id=item["item_id"],
                    audio_path=item["audio_path"],
                    predicted_label=item["predicted_label"],
                    target_class=item["target_class"],
                    polarity=ReviewPolarity(item["polarity"]),
                    score=item.get("score"),
                    source_provenance=item.get("source_provenance", {}),
                    review_outcome=ReviewOutcome(item.get("review_outcome", ReviewOutcome.PENDING.value)),
                    review_decision=deserialize_review_decision_state(
                        item.get("review_decision"),
                        outcome=ReviewOutcome(item.get("review_outcome", ReviewOutcome.PENDING.value)),
                        corrected_label=item.get("corrected_label"),
                        review_note=item.get("review_note"),
                        source_provenance=item.get("source_provenance", {}),
                        queue_session_ref=row["id"],
                    ),
                    corrected_label=item.get("corrected_label"),
                    review_note=item.get("review_note"),
                    reviewed_at=datetime.fromisoformat(item["reviewed_at"]) if item.get("reviewed_at") else None,
                )
                for item in row.get("items", [])
            ],
        )

    def list(self) -> list[ReviewSession]:
        """Return all persisted review sessions."""
        sessions: list[ReviewSession] = []
        for session_id in _read_state(self._path, self._schema).keys():
            session = self.get(session_id)
            if session is not None:
                sessions.append(session)
        return sessions

def serialize_review_decision_state(decision: ReviewDecision | None) -> dict[str, Any] | None:
    """Persist one review decision in the Foundry state-file shape."""
    if decision is None:
        return None
    return {
        "kind": decision.kind.value,
        "corrected_label": decision.corrected_label,
        "review_note": decision.review_note,
        "original_start_ms": decision.original_start_ms,
        "original_end_ms": decision.original_end_ms,
        "corrected_start_ms": decision.corrected_start_ms,
        "corrected_end_ms": decision.corrected_end_ms,
        "created_event_ref": decision.created_event_ref,
        "provenance": (
            {
                "surface": decision.provenance.surface.value,
                "workflow": decision.provenance.workflow,
                "operator_action": decision.provenance.operator_action,
                "queue_session_ref": decision.provenance.queue_session_ref,
                "project_ref": decision.provenance.project_ref,
                "song_ref": decision.provenance.song_ref,
                "version_ref": decision.provenance.version_ref,
                "layer_ref": decision.provenance.layer_ref,
                "event_ref": decision.provenance.event_ref,
                "source_event_ref": decision.provenance.source_event_ref,
                "model_ref": decision.provenance.model_ref,
            }
            if decision.provenance is not None
            else None
        ),
        "training_eligibility": {
            "allows_positive_signal": decision.training_eligibility.allows_positive_signal,
            "allows_negative_signal": decision.training_eligibility.allows_negative_signal,
            "requires_materialized_correction": (
                decision.training_eligibility.requires_materialized_correction
            ),
        },
    }


def deserialize_review_decision_state(
    payload: dict | None,
    *,
    outcome: ReviewOutcome,
    corrected_label: str | None,
    review_note: str | None,
    source_provenance: dict[str, Any],
    queue_session_ref: str,
) -> ReviewDecision | None:
    """Load one persisted review decision, filling compatibility defaults."""
    if isinstance(payload, dict) and payload.get("kind"):
        provenance_payload = payload.get("provenance")
        training_payload = payload.get("training_eligibility")
        return ReviewDecision(
            kind=ReviewDecisionKind(str(payload["kind"])),
            corrected_label=payload.get("corrected_label"),
            review_note=payload.get("review_note"),
            original_start_ms=_optional_float(payload.get("original_start_ms")),
            original_end_ms=_optional_float(payload.get("original_end_ms")),
            corrected_start_ms=_optional_float(payload.get("corrected_start_ms")),
            corrected_end_ms=_optional_float(payload.get("corrected_end_ms")),
            created_event_ref=_optional_text(payload.get("created_event_ref")),
            provenance=build_review_provenance(
                source_provenance,
                queue_session_ref=queue_session_ref,
                surface=_load_surface(provenance_payload),
                workflow=_load_workflow(provenance_payload),
                operator_action=_load_operator_action(provenance_payload),
                payload=provenance_payload if isinstance(provenance_payload, dict) else None,
            ),
            training_eligibility=_deserialize_training_eligibility(
                training_payload,
                kind=ReviewDecisionKind(str(payload["kind"])),
            ),
        )
    return build_review_decision(
        outcome,
        corrected_label=corrected_label,
        review_note=review_note,
        provenance=build_review_provenance(
            source_provenance,
            queue_session_ref=queue_session_ref,
        ),
    )


def _deserialize_training_eligibility(
    payload: Any,
    *,
    kind: ReviewDecisionKind,
) -> ReviewTrainingEligibility:
    if not isinstance(payload, dict):
        return build_training_eligibility(kind)
    return ReviewTrainingEligibility(
        allows_positive_signal=bool(payload.get("allows_positive_signal", False)),
        allows_negative_signal=bool(payload.get("allows_negative_signal", False)),
        requires_materialized_correction=bool(payload.get("requires_materialized_correction", False)),
    )


def _load_surface(payload: Any) -> ReviewSurface:
    if not isinstance(payload, dict) or not payload.get("surface"):
        return ReviewSurface.PHONE_REVIEW
    return ReviewSurface(str(payload["surface"]))


def _load_workflow(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "manual_review"
    return _optional_text(payload.get("workflow")) or "manual_review"


def _load_operator_action(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _optional_text(payload.get("operator_action", payload.get("operatorAction")))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
