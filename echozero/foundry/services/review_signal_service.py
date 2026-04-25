"""ReviewSignalService emits canonical durable review records.
Exists because review sessions manage work queues, not durable review truth.
Connects explicit review commits from session flow to Foundry signal persistence.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from echozero.foundry.domain.review import ReviewItem, ReviewOutcome, ReviewSession, ReviewSignal
from echozero.foundry.persistence.review_signal_repository import ReviewSignalRepository
from echozero.foundry.services.dataset_service import DatasetService
from echozero.foundry.services.review_writeback_service import ReviewWritebackService


class ReviewSignalService:
    """Upserts one durable review signal for each explicit review commit."""

    def __init__(
        self,
        root: Path,
        repository: ReviewSignalRepository | None = None,
        dataset_service: DatasetService | None = None,
        writeback_service: ReviewWritebackService | None = None,
    ):
        self._repo = repository or ReviewSignalRepository(root)
        self._datasets = dataset_service or DatasetService(root)
        self._writeback = writeback_service or ReviewWritebackService()

    def record_session_item_review(self, session: ReviewSession, item: ReviewItem) -> ReviewSignal:
        """Persist the canonical review signal for one explicitly reviewed queue item."""
        if item.review_outcome == ReviewOutcome.PENDING or item.review_decision is None:
            raise ValueError("Explicit review signals require a committed non-pending decision")
        reviewed_at = item.reviewed_at or datetime.now(UTC)
        signal_id = self.build_signal_id(session.id, item.item_id)
        existing = self._repo.get(signal_id)
        signal = ReviewSignal(
            id=signal_id,
            session_id=session.id,
            item_id=item.item_id,
            audio_path=item.audio_path,
            predicted_label=item.predicted_label,
            target_class=item.target_class,
            polarity=item.polarity,
            score=item.score,
            source_provenance=dict(item.source_provenance),
            review_outcome=item.review_outcome,
            review_decision=item.review_decision,
            corrected_label=item.corrected_label,
            review_note=item.review_note,
            reviewed_at=reviewed_at,
            created_at=existing.created_at if existing is not None else reviewed_at,
            updated_at=reviewed_at,
        )
        signal.source_provenance = _json_safe_value(signal.source_provenance)
        signal.source_provenance["project_writeback"] = _json_safe_value(
            self._writeback.apply_review_signal(session, signal)
        )
        signal.source_provenance["dataset_materialization"] = _json_safe_value(
            self._datasets.materialize_review_signal(session, signal)
        )
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
