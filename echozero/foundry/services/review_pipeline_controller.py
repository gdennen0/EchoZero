"""Shared controller for explicit review commits across producer surfaces.
Exists because timeline fix mode and phone review should use one mutation boundary.
Connects explicit review commands to canonical writeback and durable signal persistence.
"""

from __future__ import annotations

from pathlib import Path

from echozero.foundry.domain.review import ReviewCommitCommand, ReviewSignal
from echozero.foundry.services.review_signal_service import ReviewSignalService


class ReviewPipelineController:
    """Canonical review-mutation boundary used by all producer surfaces."""

    def __init__(
        self,
        root: Path,
        *,
        signal_service: ReviewSignalService | None = None,
    ) -> None:
        self._signals = signal_service or ReviewSignalService(root)

    def commit(self, command: ReviewCommitCommand) -> ReviewSignal:
        """Commit one explicit review decision through the shared review boundary."""
        return self._signals.record_explicit_review(
            command.context,
            command.commit,
            apply_project_writeback=command.apply_project_writeback,
        )
