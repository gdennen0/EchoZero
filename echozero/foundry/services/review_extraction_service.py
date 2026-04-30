"""Explicit review-to-dataset extraction service.
Exists because review mutation and dataset extraction should remain separate lanes.
Connects durable review truth to versioned review dataset exports on explicit demand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from echozero.foundry.domain.review import ReviewCommitContext, ReviewSignal
from echozero.foundry.services.dataset_service import DatasetService


class ReviewExtractionService:
    """Handles explicit review dataset extraction operations."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_service: DatasetService | None = None,
    ) -> None:
        self._datasets = dataset_service or DatasetService(root)

    def extract_project_review_dataset(
        self,
        project_path: str | Path,
        *,
        project_ref: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        queue_source_kind: str = "ez_project",
    ):
        """Extract one project-backed review dataset version on explicit trigger."""
        return self._datasets.export_project_review_dataset(
            project_path,
            project_ref=project_ref,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            queue_source_kind=queue_source_kind,
        )

    def extract_review_signal(
        self,
        context: ReviewCommitContext,
        signal: ReviewSignal,
    ) -> dict[str, object]:
        """Extract one review signal into dataset state on explicit trigger."""
        reviewed_at = signal.reviewed_at or datetime.now(UTC)
        return self._datasets.materialize_review_signal(
            context.as_review_session(reviewed_at=reviewed_at),
            signal,
        )
