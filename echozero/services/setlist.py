"""
SetlistProcessor: Coordinates sequential processing of all songs in a setlist.
Exists because batch analysis of an entire setlist is the primary user workflow.
Wraps AnalysisService with per-song error isolation and progress reporting.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from echozero.persistence.entities import SongPipelineConfig
from echozero.persistence.session import ProjectSession
from echozero.result import Result, is_ok
from echozero.services.analysis import AnalysisResult, AnalysisService


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

    def __init__(self, analysis_service: AnalysisService) -> None:
        self._analysis = analysis_service

    def process_setlist(
        self,
        session: ProjectSession,
        configs: list[SongPipelineConfig],
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> SetlistResult:
        """Process all songs sequentially. Continue on error.

        Args:
            session: The project session
            configs: Per-song pipeline configs (song_version_id + pipeline_id + bindings)
            on_progress: Callback(message, current_index, total_count)

        Returns SetlistResult with per-song results and summary.
        """
        start_time = time.monotonic()
        results: list[Result[AnalysisResult]] = []
        succeeded = 0
        failed = 0
        total = len(configs)

        for i, config in enumerate(configs):
            if on_progress:
                on_progress(f"Processing song {i + 1}/{total}", i, total)

            result = self._analysis.analyze(
                session=session,
                song_version_id=config.song_version_id,
                pipeline_id=config.pipeline_id,
                bindings=config.bindings,
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
