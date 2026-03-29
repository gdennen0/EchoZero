"""
SetlistProcessor: Coordinates sequential processing of all songs in a setlist.
Exists because batch analysis of an entire setlist is the primary user workflow.
Wraps Orchestrator with per-song error isolation and progress reporting.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from echozero.persistence.entities import PipelineConfig
from echozero.persistence.session import ProjectSession
from echozero.result import Result, is_ok
from echozero.services.orchestrator import AnalysisResult, Orchestrator


@dataclass(frozen=True)
class SetlistResult:
    """Summary of processing an entire setlist."""

    total: int
    succeeded: int
    failed: int
    results: list[Result[AnalysisResult]]
    duration_ms: float


class SetlistProcessor:
    """Coordinates sequential processing of all songs in a setlist."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator

    def process_setlist(
        self,
        session: ProjectSession,
        config_ids: list[str],
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> SetlistResult:
        """Process all songs sequentially using persisted PipelineConfig IDs.

        Args:
            session: The project session
            config_ids: List of PipelineConfig IDs to execute
            on_progress: Callback(message, current_index, total_count)

        Returns SetlistResult with per-song results and summary.
        """
        start_time = time.monotonic()
        results: list[Result[AnalysisResult]] = []
        succeeded = 0
        failed = 0
        total = len(config_ids)

        for i, config_id in enumerate(config_ids):
            if on_progress:
                on_progress(f"Processing song {i + 1}/{total}", i, total)

            result = self._orchestrator.execute(
                session=session,
                config_id=config_id,
            )
            results.append(result)
            if is_ok(result):
                succeeded += 1
            else:
                failed += 1

        duration_ms = (time.monotonic() - start_time) * 1000

        return SetlistResult(
            total=total,
            succeeded=succeeded,
            failed=failed,
            results=results,
            duration_ms=duration_ms,
        )
