"""Application-owned background pipeline run execution.
Exists to keep object-action execution and transient run state out of Qt widgets.
Connects prepared object-action requests to background orchestrator runs and app-visible status.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from echozero.result import is_err, unwrap
from echozero.services.orchestrator import AnalysisService

if TYPE_CHECKING:
    from echozero.application.session.models import Session
    from echozero.persistence.session import ProjectStorage

_ACTIVE_PIPELINE_RUN_STATUSES = frozenset({"queued", "resolving", "running", "persisting"})
_FINAL_PIPELINE_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(slots=True, frozen=True)
class PreparedPipelineRun:
    """Resolved object-action run request ready for the orchestrator."""

    action_id: str
    workflow_id: str
    pipeline_template_id: str
    config_id: str
    display_label: str
    object_id: str
    object_type: str
    source_layer_id: str | None
    song_id: str | None = None
    song_version_id: str | None = None
    runtime_bindings: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PipelineRunState:
    """Transient application-owned state for one pipeline run."""

    run_id: str
    action_id: str
    workflow_id: str
    display_label: str
    object_id: str
    object_type: str
    source_layer_id: str | None
    song_id: str | None
    song_version_id: str | None
    status: str
    message: str
    percent: float | None
    started_at: float
    finished_at: float | None = None
    can_cancel: bool = False
    error: str | None = None
    exception: BaseException | None = None
    output_layer_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class PipelineRunNotification:
    """Coalesced update emitted to runtime adapters when run state changes."""

    revision: int
    refresh_presentation: bool = False


@dataclass(slots=True, frozen=True)
class _PipelineRunRequest:
    action_id: str
    params: dict[str, object] | None
    object_id: object | None
    object_type: str | None
    persist_scope: str | None


class PipelineRunService:
    """Execute prepared object actions off-thread and track transient run state."""

    def __init__(
        self,
        *,
        project_storage_getter: Callable[[], ProjectStorage],
        session_getter: Callable[[], Session],
        analysis_service: AnalysisService,
        prepare_run: Callable[[str, dict[str, object] | None, object | None, str | None, str | None], PreparedPipelineRun],
        persist_generated_source_layer_id: Callable[..., None],
        max_workers: int = 4,
    ) -> None:
        self._project_storage_getter = project_storage_getter
        self._session_getter = session_getter
        self._analysis_service = analysis_service
        self._prepare_run = prepare_run
        self._persist_generated_source_layer_id = persist_generated_source_layer_id
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="ez-pipeline-run",
        )
        self._futures: dict[str, Future[None]] = {}
        self._subject_run_ids: dict[tuple[str, str, str], str] = {}
        self._revision = 0
        self._pending_refresh = False

    @property
    def session(self) -> Session:
        return self._session_getter()

    def request_run(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> str:
        """Queue one object-action run and return its `run_id` immediately."""

        session = self.session
        requested_object_id = str(object_id or "")
        requested_object_type = str(object_type or "object")
        requested_song_id = (
            str(session.active_song_id) if session.active_song_id is not None else None
        )
        requested_song_version_id = (
            str(session.active_song_version_id)
            if session.active_song_version_id is not None
            else None
        )
        subject_key = (
            self._subject_scope_key(
                song_id=requested_song_id,
                song_version_id=requested_song_version_id,
            ),
            requested_object_type,
            requested_object_id,
        )
        run_id = f"pipeline_run_{uuid.uuid4().hex[:12]}"
        request = _PipelineRunRequest(
            action_id=action_id,
            params=dict(params) if params is not None else None,
            object_id=object_id,
            object_type=object_type,
            persist_scope=persist_scope,
        )
        initial_state = PipelineRunState(
            run_id=run_id,
            action_id=action_id,
            workflow_id="",
            display_label=self._display_label_from_action_id(action_id),
            object_id=requested_object_id,
            object_type=requested_object_type,
            source_layer_id=requested_object_id or None,
            song_id=requested_song_id,
            song_version_id=requested_song_version_id,
            status="queued",
            message="Queued",
            percent=0.0,
            started_at=time.time(),
        )
        with self._lock:
            active_run_id = self._subject_run_ids.get(subject_key)
            active_state = self.session.pipeline_runs.get(active_run_id) if active_run_id is not None else None
            if active_state is not None and self.is_active(active_state):
                raise RuntimeError(
                    f"{action_id} is already running for {requested_object_type} '{requested_object_id}'."
                )
            self._subject_run_ids[subject_key] = run_id
            self._store_state(initial_state)
            self._futures[run_id] = self._executor.submit(
                self._execute_requested_run,
                run_id,
                request,
                subject_key,
            )
        return run_id

    def wait_for_run(self, run_id: str, *, timeout: float | None = None) -> PipelineRunState:
        """Block until the target run has finished and return the final state."""

        with self._lock:
            future = self._futures.get(run_id)
        if future is not None:
            future.result(timeout=timeout)
        state = self.get_run(run_id)
        if state is None:
            raise ValueError(f"Unknown pipeline run '{run_id}'.")
        return state

    def get_run(self, run_id: str) -> PipelineRunState | None:
        with self._lock:
            return self.session.pipeline_runs.get(run_id)

    def visible_run_for(
        self,
        *,
        action_id: str,
        object_id: object | None,
        object_type: str | None,
        song_id: object | None = None,
        song_version_id: object | None = None,
    ) -> PipelineRunState | None:
        """Return the latest run worth surfacing in plans for one object action."""

        requested_object_id = str(object_id or "")
        requested_object_type = str(object_type or "object")
        requested_song_id = str(song_id) if song_id is not None else None
        requested_song_version_id = (
            str(song_version_id) if song_version_id is not None else None
        )
        with self._lock:
            matching = [
                state
                for state in self.session.pipeline_runs.values()
                if state.action_id == action_id
                and state.object_type == requested_object_type
                and self._run_matches_context(
                    state,
                    song_id=requested_song_id,
                    song_version_id=requested_song_version_id,
                )
                and (
                    state.object_id == requested_object_id
                    or state.source_layer_id == requested_object_id
                )
            ]
        if not matching:
            return None
        latest = max(matching, key=lambda state: state.started_at)
        if latest.status in _ACTIVE_PIPELINE_RUN_STATUSES or latest.status == "failed":
            return latest
        return None

    def consume_updates_since(self, revision: int) -> PipelineRunNotification | None:
        """Return the latest coalesced notification after `revision`, if any."""

        with self._lock:
            if revision >= self._revision:
                return None
            notification = PipelineRunNotification(
                revision=self._revision,
                refresh_presentation=self._pending_refresh,
            )
            self._pending_refresh = False
            return notification

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    @staticmethod
    def is_active(state: PipelineRunState | None) -> bool:
        return state is not None and state.status in _ACTIVE_PIPELINE_RUN_STATUSES

    def _execute_requested_run(
        self,
        run_id: str,
        request: _PipelineRunRequest,
        subject_key: tuple[str, str, str],
    ) -> None:
        prepared: PreparedPipelineRun | None = None
        try:
            self._update_state(
                run_id,
                status="resolving",
                message="Resolving object action",
                percent=0.0,
            )
            prepared = self._prepare_run(
                request.action_id,
                request.params,
                request.object_id,
                request.object_type,
                request.persist_scope,
            )
            self._update_state(
                run_id,
                workflow_id=prepared.workflow_id,
                display_label=prepared.display_label,
                object_id=prepared.object_id,
                object_type=prepared.object_type,
                source_layer_id=prepared.source_layer_id,
                song_id=prepared.song_id,
                song_version_id=prepared.song_version_id,
                message="Preparing pipeline",
                percent=0.1,
            )
            result = self._analysis_service.execute(
                self._project_storage_getter(),
                prepared.config_id,
                runtime_bindings=prepared.runtime_bindings,
                on_progress=lambda message, percent: self._handle_progress(
                    run_id,
                    message=message,
                    percent=percent,
                ),
            )
            if is_err(result):
                raise RuntimeError(f"{request.action_id} failed: {result.error}")
            analysis_result = unwrap(result)
            self._persist_generated_source_layer_id(
                analysis_result=analysis_result,
                source_layer_id=prepared.source_layer_id,
            )
            self._update_state(
                run_id,
                status="completed",
                message="Complete",
                percent=1.0,
                finished_at=time.time(),
                output_layer_ids=tuple(analysis_result.layer_ids),
                refresh_presentation=True,
            )
        except Exception as exc:
            self._update_state(
                run_id,
                workflow_id=prepared.workflow_id if prepared is not None else None,
                display_label=prepared.display_label if prepared is not None else None,
                object_id=prepared.object_id if prepared is not None else None,
                object_type=prepared.object_type if prepared is not None else None,
                source_layer_id=prepared.source_layer_id if prepared is not None else None,
                song_id=prepared.song_id if prepared is not None else None,
                song_version_id=prepared.song_version_id if prepared is not None else None,
                status="failed",
                message=str(exc),
                error=str(exc),
                exception=exc,
                finished_at=time.time(),
            )
        finally:
            with self._lock:
                active_run_id = self._subject_run_ids.get(subject_key)
                if active_run_id == run_id:
                    self._subject_run_ids.pop(subject_key, None)

    def _handle_progress(self, run_id: str, *, message: str, percent: float) -> None:
        status = "running"
        if message in {"Loading configuration", "Preparing pipeline"}:
            status = "resolving"
        elif message == "Persisting results":
            status = "persisting"
        self._update_state(
            run_id,
            status=status,
            message=message,
            percent=round(float(percent), 3),
        )

    def _update_state(
        self,
        run_id: str,
        *,
        workflow_id: str | None = None,
        display_label: str | None = None,
        object_id: str | None = None,
        object_type: str | None = None,
        source_layer_id: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        status: str | None = None,
        message: str | None = None,
        percent: float | None = None,
        finished_at: float | None = None,
        error: str | None = None,
        exception: BaseException | None = None,
        output_layer_ids: tuple[str, ...] | None = None,
        refresh_presentation: bool = False,
    ) -> None:
        with self._lock:
            current = self.session.pipeline_runs.get(run_id)
            if current is None:
                return
            updated = replace(
                current,
                workflow_id=workflow_id if workflow_id is not None else current.workflow_id,
                display_label=display_label if display_label is not None else current.display_label,
                object_id=object_id if object_id is not None else current.object_id,
                object_type=object_type if object_type is not None else current.object_type,
                source_layer_id=source_layer_id if source_layer_id is not None else current.source_layer_id,
                song_id=song_id if song_id is not None else current.song_id,
                song_version_id=(
                    song_version_id
                    if song_version_id is not None
                    else current.song_version_id
                ),
                status=status if status is not None else current.status,
                message=message if message is not None else current.message,
                percent=percent if percent is not None else current.percent,
                finished_at=finished_at if finished_at is not None else current.finished_at,
                error=error if error is not None else current.error,
                exception=exception if exception is not None else current.exception,
                output_layer_ids=output_layer_ids if output_layer_ids is not None else current.output_layer_ids,
            )
            if updated == current:
                if refresh_presentation:
                    self._pending_refresh = True
                return
            self._store_state(updated, refresh_presentation=refresh_presentation)

    def _store_state(
        self,
        state: PipelineRunState,
        *,
        refresh_presentation: bool = False,
    ) -> None:
        self.session.pipeline_runs[state.run_id] = state
        self._revision += 1
        if refresh_presentation:
            self._pending_refresh = True

    @staticmethod
    def _display_label_from_action_id(action_id: str) -> str:
        normalized = str(action_id).split(".")[-1].replace("_", " ").strip()
        return normalized.title() if normalized else "Pipeline Run"

    @staticmethod
    def _subject_scope_key(
        *,
        song_id: str | None,
        song_version_id: str | None,
    ) -> str:
        if song_version_id:
            return f"version:{song_version_id}"
        if song_id:
            return f"song:{song_id}"
        return "global"

    @staticmethod
    def _run_matches_context(
        state: PipelineRunState,
        *,
        song_id: str | None,
        song_version_id: str | None,
    ) -> bool:
        if song_version_id is not None:
            return state.song_version_id == song_version_id
        if song_id is not None:
            return state.song_id == song_id
        return True
