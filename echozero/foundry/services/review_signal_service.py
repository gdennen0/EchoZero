"""ReviewSignalService emits canonical durable review records.
Exists because review sessions manage work queues, not durable review truth.
Connects explicit review commits from session flow to Foundry signal persistence.
The commit hot path never performs dataset extraction; extraction is explicit-only.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from echozero.foundry.domain.review import (
    ExplicitReviewCommit,
    ReviewCommitContext,
    ReviewOutcome,
    ReviewSignal,
)
from echozero.foundry.persistence.review_signal_repository import ReviewSignalRepository
from echozero.foundry.services.review_writeback_service import ReviewWritebackService


class ReviewSignalService:
    """Upserts one durable review signal for each explicit review commit."""

    def __init__(
        self,
        root: Path,
        repository: ReviewSignalRepository | None = None,
        writeback_service: ReviewWritebackService | None = None,
    ):
        self._repo = repository or ReviewSignalRepository(root)
        self._writeback = writeback_service or ReviewWritebackService()

    def record_explicit_review(
        self,
        context: ReviewCommitContext,
        commit: ExplicitReviewCommit,
        *,
        apply_project_writeback: bool = True,
    ) -> ReviewSignal:
        """Persist one explicit review commit from any producer surface."""
        if commit.review_outcome == ReviewOutcome.PENDING or commit.review_decision is None:
            raise ValueError("Explicit review signals require a committed non-pending decision")
        reviewed_at = commit.reviewed_at or datetime.now(UTC)
        signal_id = commit.signal_id or self.build_signal_id(context.session_id, commit.item_id)
        existing = self._repo.get(signal_id)
        signal = ReviewSignal(
            id=signal_id,
            session_id=context.session_id,
            item_id=commit.item_id,
            audio_path=commit.audio_path,
            predicted_label=commit.predicted_label,
            target_class=commit.target_class,
            polarity=commit.polarity,
            score=commit.score,
            source_provenance=dict(commit.source_provenance),
            review_outcome=commit.review_outcome,
            review_decision=commit.review_decision,
            corrected_label=commit.corrected_label,
            review_note=commit.review_note,
            reviewed_at=reviewed_at,
            created_at=existing.created_at if existing is not None else reviewed_at,
            updated_at=reviewed_at,
        )
        normalized_source_provenance = _json_safe_value(signal.source_provenance)
        if not isinstance(normalized_source_provenance, dict):
            normalized_source_provenance = {"value": normalized_source_provenance}
        signal.source_provenance = normalized_source_provenance
        session = context.as_review_session(reviewed_at=reviewed_at)
        if apply_project_writeback:
            signal.source_provenance["project_writeback"] = _json_safe_value(
                self._writeback.apply_review_signal(session, signal)
            )
        else:
            signal.source_provenance["project_writeback"] = {
                "status": "deferred",
                "reason": "skipped_by_producer",
            }
        signal.source_provenance["dataset_materialization"] = {
            "status": "deferred",
            "reason": "explicit_extraction_only",
            "next_step": "use_review_extraction_service",
        }
        return self._repo.save(signal)

    @staticmethod
    def build_signal_id(session_id: str, item_id: str) -> str:
        """Return the stable signal id for one review-queue item."""
        return f"rsig_{session_id}_{item_id}"


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_safe_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)
