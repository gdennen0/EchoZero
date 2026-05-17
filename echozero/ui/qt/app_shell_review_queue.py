"""Deferred timeline-review persistence queue for the Qt app shell.
Exists because fix-mode review should flip local event state immediately without blocking on durable review artifacts.
Connects app-shell review commits to background signal persistence and optional sample export work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from echozero.foundry.domain.review import ExplicitReviewCommit, ReviewCommitContext
from echozero.foundry.services.review_audio_clip_service import ReviewAudioClipService
from echozero.foundry.services.review_commit_mapper import build_review_commit_command
from echozero.foundry.services.review_pipeline_controller import ReviewPipelineController
from echozero.ui.qt.timeline_review_sample_export import safe_export_timeline_review_sample

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeferredTimelineReviewSampleExport:
    """One optional follow-up sample export tied to a persisted review signal."""

    class_label: str
    source_audio_path: str
    start_seconds: float
    end_seconds: float
    event_id: str
    decision_kind: object


@dataclass(slots=True)
class DeferredTimelineReviewPersistenceEntry:
    """One durable review signal commit plus optional export metadata."""

    context: ReviewCommitContext
    commit: ExplicitReviewCommit
    sample_export: DeferredTimelineReviewSampleExport | None = None
    apply_project_writeback: bool = True


@dataclass(slots=True)
class DeferredTimelineReviewPersistence:
    """One batch of queued review persistence work sharing one background controller."""

    root: Path
    entries: tuple[DeferredTimelineReviewPersistenceEntry, ...]


class TimelineReviewPersistenceQueue:
    """Serial background worker for timeline-review persistence and sample export."""

    def __init__(self) -> None:
        self._queue: Queue[DeferredTimelineReviewPersistence] = Queue()
        self._stop_requested = Event()
        self._thread: Thread | None = None

    def enqueue(self, work: DeferredTimelineReviewPersistence) -> None:
        self._ensure_thread()
        self._queue.put(work)

    def flush(self) -> None:
        if self._thread is None:
            return
        self._queue.join()

    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        if self._thread is None:
            return
        self._queue.join()
        self._stop_requested.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(
            target=self._run,
            name="TimelineReviewPersistenceQueue",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            if self._stop_requested.is_set() and self._queue.empty():
                return
            try:
                work = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._process(work)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                logger.warning(
                    "Deferred timeline review persistence failed: %s",
                    exc,
                    exc_info=True,
                )
            finally:
                self._queue.task_done()

    @staticmethod
    def _process(work: DeferredTimelineReviewPersistence) -> None:
        controller = ReviewPipelineController(Path(work.root).resolve())
        clip_service = ReviewAudioClipService()
        for entry in work.entries:
            signal = controller.commit(
                build_review_commit_command(
                    context=entry.context,
                    commit=entry.commit,
                    apply_project_writeback=entry.apply_project_writeback,
                )
            )
            sample_export = entry.sample_export
            if sample_export is None:
                continue
            safe_export_timeline_review_sample(
                signal=signal,
                class_label=sample_export.class_label,
                source_audio_path=sample_export.source_audio_path,
                start_seconds=sample_export.start_seconds,
                end_seconds=sample_export.end_seconds,
                event_id=sample_export.event_id,
                decision_kind=sample_export.decision_kind,
                clip_service=clip_service,
            )
