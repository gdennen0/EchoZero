"""Shared mapping helpers for producer review payloads.
Exists because timeline and phone review should emit one normalized commit contract.
Connects producer payload shapes to ReviewCommitCommand with canonical provenance keys.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from echozero.foundry.domain.review import (
    ExplicitReviewCommit,
    ReviewCommitCommand,
    ReviewCommitContext,
    ReviewItem,
    ReviewSession,
)


_CANONICAL_REF_KEYS = {
    "project_ref": "projectRef",
    "song_ref": "songRef",
    "version_ref": "versionRef",
    "layer_ref": "layerRef",
    "event_ref": "eventRef",
    "source_event_ref": "sourceEventRef",
    "model_ref": "modelRef",
    "bundle_ref": "bundleRef",
    "take_ref": "takeRef",
    "audio_ref": "audioRef",
    "source_audio_ref": "sourceAudioRef",
}

_CANONICAL_REVIEW_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "session_id": ("sessionId",),
    "item_id": ("itemId",),
    "outcome": ("reviewOutcome",),
    "corrected_label": ("correctedLabel",),
    "review_note": ("reviewNote",),
    "decision_kind": ("decisionKind",),
    "original_start_ms": ("originalStartMs",),
    "original_end_ms": ("originalEndMs",),
    "corrected_start_ms": ("correctedStartMs",),
    "corrected_end_ms": ("correctedEndMs",),
    "created_event_ref": ("createdEventRef",),
    "operator_action": ("operatorAction",),
    "queue_source_kind": ("queueSourceKind",),
}


def normalize_source_provenance(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return source provenance with canonical key aliases populated."""
    normalized = dict(payload or {})
    for canonical, camel in _CANONICAL_REF_KEYS.items():
        if canonical in normalized:
            continue
        value = normalized.get(camel)
        if value is not None:
            normalized[canonical] = value
    return normalized


def normalize_review_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return review payload with canonical snake_case keys populated."""
    normalized = dict(payload or {})
    for canonical, aliases in _CANONICAL_REVIEW_PAYLOAD_KEYS.items():
        if canonical in normalized:
            continue
        for alias in aliases:
            value = normalized.get(alias)
            if value is not None:
                normalized[canonical] = value
                break
    return normalized


def build_review_commit_context(session: ReviewSession) -> ReviewCommitContext:
    """Build canonical commit context from a review session."""
    return ReviewCommitContext(
        session_id=session.id,
        session_name=session.name,
        source_ref=session.source_ref,
        metadata=dict(session.metadata),
    )


def build_explicit_commit_from_item(item: ReviewItem) -> ExplicitReviewCommit:
    """Build canonical explicit commit payload from one reviewed item."""
    return ExplicitReviewCommit(
        item_id=item.item_id,
        audio_path=item.audio_path,
        predicted_label=item.predicted_label,
        target_class=item.target_class,
        polarity=item.polarity,
        score=item.score,
        source_provenance=normalize_source_provenance(item.source_provenance),
        review_outcome=item.review_outcome,
        review_decision=item.review_decision,
        corrected_label=item.corrected_label,
        review_note=item.review_note,
        reviewed_at=item.reviewed_at,
    )


def build_review_commit_command(
    *,
    context: ReviewCommitContext,
    commit: ExplicitReviewCommit,
    apply_project_writeback: bool = True,
) -> ReviewCommitCommand:
    """Build one normalized command for the shared review pipeline boundary."""
    normalized_commit = replace(
        commit,
        source_provenance=normalize_source_provenance(commit.source_provenance),
    )
    return ReviewCommitCommand(
        context=context,
        commit=normalized_commit,
        apply_project_writeback=apply_project_writeback,
    )
