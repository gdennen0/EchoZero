"""Application-owned background operation progress execution and state.
Exists to keep long-running object-action execution and status tracking out of widgets.
Connects object-action requests to orchestrator execution and app-visible progress banners.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from echozero.application.progress import (
    ACTIVE_OPERATION_PROGRESS_STATUSES,
    OperationProgressStatus,
    OperationProgressUpdate,
)
from echozero.result import is_err, unwrap
from echozero.services.orchestrator import Orchestrator

if TYPE_CHECKING:
    from echozero.application.session.models import Session
    from echozero.persistence.session import ProjectStorage


@dataclass(slots=True, frozen=True)
class PreparedOperation:
    """Resolved object-action operation request ready for the orchestrator."""

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
class OperationProgressState:
    """Transient application-owned progress state for one operation."""

    operation_id: str
    kind: str
    action_id: str
    workflow_id: str
    display_label: str
    object_id: str
    object_type: str
    source_layer_id: str | None
    song_id: str | None
    song_version_id: str | None
    status: OperationProgressStatus
    message: str
    fraction_complete: float | None
    started_at: float
    finished_at: float | None = None
    can_cancel: bool = False
    error: str | None = None
    exception: BaseException | None = None
    output_layer_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class OperationProgressNotification:
    """Coalesced update emitted to runtime adapters when operation state changes."""

    revision: int
    refresh_presentation: bool = False


@dataclass(slots=True, frozen=True)
class _OperationRequest:
    action_id: str
    params: dict[str, object] | None
    object_id: object | None
    object_type: str | None
    persist_scope: str | None


class OperationProgressService:
    """Execute prepared object actions off-thread and track transient operation state."""

    def __init__(
        self,
        *,
        project_storage_getter: Callable[[], ProjectStorage],
        session_getter: Callable[[], Session],
        analysis_service: Orchestrator,
        prepare_operation: Callable[[str, dict[str, object] | None, object | None, str | None, str | None], PreparedOperation],
        persist_generated_source_layer_id: Callable[..., None],
        max_workers: int = 4,
    ) -> None:
        self._project_storage_getter = project_storage_getter
        self._session_getter = session_getter
        self._analysis_service = analysis_service
        self._prepare_operation = prepare_operation
        self._persist_generated_source_layer_id = persist_generated_source_layer_id
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="ez-operation",
        )
        self._futures: dict[str, Future[None]] = {}
        self._subject_operation_ids: dict[tuple[str, str, str], str] = {}
        self._revision = 0
        self._pending_refresh = False

    @property
    def session(self) -> Session:
        return self._session_getter()

    def request_operation(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> str:
        """Queue one object-action operation and return its `operation_id` immediately."""

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
        operation_id = f"operation_{uuid.uuid4().hex[:12]}"
        request = _OperationRequest(
            action_id=action_id,
            params=dict(params) if params is not None else None,
            object_id=object_id,
            object_type=object_type,
            persist_scope=persist_scope,
        )
        initial_state = OperationProgressState(
            operation_id=operation_id,
            kind="pipeline",
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
            fraction_complete=0.0,
            started_at=time.time(),
        )
        with self._lock:
            active_operation_id = self._subject_operation_ids.get(subject_key)
            active_state = (
                self.session.operation_progress_by_id.get(active_operation_id)
                if active_operation_id is not None
                else None
            )
            if active_state is not None and self.is_active(active_state):
                raise RuntimeError(
                    f"{action_id} is already running for {requested_object_type} '{requested_object_id}'."
                )
            self._subject_operation_ids[subject_key] = operation_id
            self._store_state(initial_state)
            self._futures[operation_id] = self._executor.submit(
                self._execute_requested_operation,
                operation_id,
                request,
                subject_key,
            )
        return operation_id

    def wait_for_operation(
        self,
        operation_id: str,
        *,
        timeout: float | None = None,
    ) -> OperationProgressState:
        """Block until the target operation has finished and return the final state."""

        with self._lock:
            future = self._futures.get(operation_id)
        if future is not None:
            future.result(timeout=timeout)
        state = self.get_operation(operation_id)
        if state is None:
            raise ValueError(f"Unknown operation '{operation_id}'.")
        return state

    def get_operation(self, operation_id: str) -> OperationProgressState | None:
        with self._lock:
            return self.session.operation_progress_by_id.get(operation_id)

    def visible_operation_for(
        self,
        *,
        action_id: str,
        object_id: object | None,
        object_type: str | None,
        song_id: object | None = None,
        song_version_id: object | None = None,
    ) -> OperationProgressState | None:
        """Return the latest operation worth surfacing for one object action."""

        requested_object_id = str(object_id or "")
        requested_object_type = str(object_type or "object")
        requested_song_id = str(song_id) if song_id is not None else None
        requested_song_version_id = (
            str(song_version_id) if song_version_id is not None else None
        )
        with self._lock:
            matching = [
                state
                for state in self.session.operation_progress_by_id.values()
                if state.action_id == action_id
                and state.object_type == requested_object_type
                and self._operation_matches_context(
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
        if latest.status in ACTIVE_OPERATION_PROGRESS_STATUSES or latest.status == "failed":
            return latest
        return None

    def consume_updates_since(
        self,
        revision: int,
    ) -> OperationProgressNotification | None:
        """Return the latest coalesced notification after `revision`, if any."""

        with self._lock:
            if revision >= self._revision:
                return None
            notification = OperationProgressNotification(
                revision=self._revision,
                refresh_presentation=self._pending_refresh,
            )
            self._pending_refresh = False
            return notification

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    @staticmethod
    def is_active(state: OperationProgressState | None) -> bool:
        return state is not None and state.status in ACTIVE_OPERATION_PROGRESS_STATUSES

    def _execute_requested_operation(
        self,
        operation_id: str,
        request: _OperationRequest,
        subject_key: tuple[str, str, str],
    ) -> None:
        prepared: PreparedOperation | None = None
        try:
            self._update_state(
                operation_id,
                status="resolving",
                message="Resolving object action",
                fraction_complete=0.0,
            )
            prepared = self._prepare_operation(
                request.action_id,
                request.params,
                request.object_id,
                request.object_type,
                request.persist_scope,
            )
            self._update_state(
                operation_id,
                workflow_id=prepared.workflow_id,
                display_label=prepared.display_label,
                object_id=prepared.object_id,
                object_type=prepared.object_type,
                source_layer_id=prepared.source_layer_id,
                song_id=prepared.song_id,
                song_version_id=prepared.song_version_id,
                message="Preparing pipeline",
                fraction_complete=0.1,
            )
            result = self._analysis_service.execute(
                self._project_storage_getter(),
                prepared.config_id,
                runtime_bindings=prepared.runtime_bindings,
                on_progress=lambda update: self._handle_progress(
                    operation_id,
                    update=update,
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
                operation_id,
                status="completed",
                message="Complete",
                fraction_complete=1.0,
                finished_at=time.time(),
                output_layer_ids=tuple(analysis_result.layer_ids),
                refresh_presentation=True,
            )
        except Exception as exc:
            self._update_state(
                operation_id,
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
                active_operation_id = self._subject_operation_ids.get(subject_key)
                if active_operation_id == operation_id:
                    self._subject_operation_ids.pop(subject_key, None)

    def _handle_progress(
        self,
        operation_id: str,
        *,
        update: OperationProgressUpdate,
    ) -> None:
        stage_to_status: dict[str, OperationProgressStatus] = {
            "loading_configuration": "resolving",
            "preparing_pipeline": "resolving",
            "executing_pipeline": "running",
            "persisting_results": "persisting",
            "complete": "completed",
        }
        status = stage_to_status.get(update.stage, "running")
        self._update_state(
            operation_id,
            status=status,
            message=update.message,
            fraction_complete=update.fraction_complete,
        )

    def _update_state(
        self,
        operation_id: str,
        *,
        workflow_id: str | None = None,
        display_label: str | None = None,
        object_id: str | None = None,
        object_type: str | None = None,
        source_layer_id: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        status: OperationProgressStatus | None = None,
        message: str | None = None,
        fraction_complete: float | None = None,
        finished_at: float | None = None,
        error: str | None = None,
        exception: BaseException | None = None,
        output_layer_ids: tuple[str, ...] | None = None,
        refresh_presentation: bool = False,
    ) -> None:
        with self._lock:
            current = self.session.operation_progress_by_id.get(operation_id)
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
                fraction_complete=(
                    self._clamp_fraction(fraction_complete)
                    if fraction_complete is not None
                    else current.fraction_complete
                ),
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
        state: OperationProgressState,
        *,
        refresh_presentation: bool = False,
    ) -> None:
        self.session.operation_progress_by_id[state.operation_id] = state
        self._revision += 1
        if refresh_presentation:
            self._pending_refresh = True

    @staticmethod
    def _display_label_from_action_id(action_id: str) -> str:
        normalized = str(action_id).split(".")[-1].replace("_", " ").strip()
        return normalized.title() if normalized else "Operation"

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
    def _operation_matches_context(
        state: OperationProgressState,
        *,
        song_id: str | None,
        song_version_id: str | None,
    ) -> bool:
        if song_version_id is not None:
            return state.song_version_id == song_version_id
        if song_id is not None:
            return state.song_id == song_id
        return True

    @staticmethod
    def _clamp_fraction(fraction_complete: float) -> float:
        return max(0.0, min(1.0, float(fraction_complete)))
