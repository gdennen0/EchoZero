"""Object-action and pipeline-run helpers for the Qt app shell.
Exists to isolate action-session orchestration and pipeline-run refresh handling.
Connects app-shell runtime services to explicit object-action workflows.
"""

from __future__ import annotations

from typing import Protocol

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.object_actions import (
    ApplyCopySource,
    ObjectActionService,
    ObjectActionSettingsPlan,
    ObjectActionSettingsSession,
    RunSession,
    SaveAndRunSession,
    SaveSession,
)
from echozero.application.timeline.pipeline_run_service import PipelineRunService, PipelineRunState
from echozero.ui.qt.app_shell_timeline_state import surface_new_take_rows


class _PipelineRunShell(Protocol):
    _app: TimelineApplication
    _is_dirty: bool
    _last_pipeline_run_revision: int
    _pipeline_runs: PipelineRunService

    @property
    def session(self) -> Session: ...

    def presentation(self) -> TimelinePresentation: ...

    def _clear_history(self) -> None: ...

    def _refresh_from_storage(
        self,
        *,
        active_song_id: SongId | None,
        active_song_version_id: SongVersionId | None,
    ) -> None: ...


class _ObjectActionShell(Protocol):
    _is_dirty: bool
    _object_action_settings: ObjectActionService

    def _clear_history(self) -> None: ...

    def request_object_action_run(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> str: ...


def run_object_action(
    shell: _PipelineRunShell,
    action_id: str,
    params: dict[str, object] | None = None,
    *,
    object_id: object | None = None,
    object_type: str | None = None,
) -> TimelinePresentation:
    """Compatibility helper that blocks until one object action finishes."""
    run_id = request_object_action_run(
        shell,
        action_id,
        params,
        object_id=object_id,
        object_type=object_type,
    )
    state = wait_for_pipeline_run(shell, run_id)
    if state.exception is not None:
        raise state.exception
    if state.error:
        raise RuntimeError(state.error)
    updated = consume_pipeline_run_presentation_update(shell)
    if updated is not None:
        return updated
    return shell.presentation()


def request_object_action_run(
    shell: _PipelineRunShell,
    action_id: str,
    params: dict[str, object] | None = None,
    *,
    object_id: object | None = None,
    object_type: str | None = None,
    persist_scope: str | None = "version",
) -> str:
    """Queue one object-scoped action on the app-owned pipeline run service."""
    run_id = shell._pipeline_runs.request_run(
        action_id,
        params,
        object_id=object_id,
        object_type=object_type,
        persist_scope=persist_scope,
    )
    if persist_scope is not None:
        shell._is_dirty = True
    return run_id


def wait_for_pipeline_run(
    shell: _PipelineRunShell,
    run_id: str,
    *,
    timeout: float | None = None,
) -> PipelineRunState:
    return shell._pipeline_runs.wait_for_run(run_id, timeout=timeout)


def get_pipeline_run_state(shell: _PipelineRunShell, run_id: str) -> PipelineRunState | None:
    return shell._pipeline_runs.get_run(run_id)


def consume_pipeline_run_presentation_update(
    shell: _PipelineRunShell,
) -> TimelinePresentation | None:
    notification = shell._pipeline_runs.consume_updates_since(shell._last_pipeline_run_revision)
    if notification is None:
        return None
    shell._last_pipeline_run_revision = notification.revision
    if notification.refresh_presentation:
        prior_presentation = shell.presentation()
        shell._refresh_from_storage(
            active_song_id=shell.session.active_song_id,
            active_song_version_id=shell.session.active_song_version_id,
        )
        surface_new_take_rows(
            timeline=shell._app.timeline,
            prior_presentation=prior_presentation,
            current_presentation=shell.presentation(),
        )
        shell._is_dirty = True
        shell._clear_history()
    return shell.presentation()


def open_object_action_session(
    shell: _ObjectActionShell,
    action_id: str,
    params: dict[str, object] | None = None,
    *,
    object_id: object | None = None,
    object_type: str | None = None,
    scope: str = "version",
) -> ObjectActionSettingsSession:
    return shell._object_action_settings.open_session(
        action_id,
        params,
        object_id=object_id,
        object_type=object_type,
        scope=scope,
    )


def dispatch_object_action_command(
    shell: _ObjectActionShell,
    session_id: str,
    command: object,
) -> ObjectActionSettingsSession:
    if isinstance(command, (RunSession, SaveAndRunSession)):
        current = shell._object_action_settings.refresh_session(session_id)
        if current.scope != "version":
            raise ValueError(
                "Reruns use this version's effective settings. Switch to This Version to run."
            )
        saved = shell._object_action_settings.dispatch_command(session_id, SaveSession())
        shell._clear_history()
        shell.request_object_action_run(
            saved.action_id,
            saved.values,
            object_id=saved.object_id,
            object_type=saved.object_type,
            persist_scope=None,
        )
        shell._is_dirty = True
        return shell._object_action_settings.refresh_session(session_id)

    session = shell._object_action_settings.dispatch_command(session_id, command)
    if isinstance(command, (SaveSession, ApplyCopySource)):
        shell._is_dirty = True
        shell._clear_history()
    return session


def save_object_action_settings(
    shell: _ObjectActionShell,
    action_id: str,
    params: dict[str, object] | None = None,
    *,
    object_id: object | None = None,
    object_type: str | None = None,
    scope: str = "version",
) -> ObjectActionSettingsPlan:
    """Persist editable settings for one object action and return the refreshed plan."""
    shell._is_dirty = True
    shell._clear_history()
    return shell._object_action_settings.save(
        action_id,
        params,
        object_id=object_id,
        object_type=object_type,
        scope=scope,
    )


def describe_object_action(
    shell: _ObjectActionShell,
    action_id: str,
    params: dict[str, object] | None = None,
    *,
    object_id: object | None = None,
    object_type: str | None = None,
    scope: str = "version",
) -> ObjectActionSettingsPlan:
    """Describe editable settings and locked bindings for one object action."""
    return shell._object_action_settings.describe(
        action_id,
        params,
        object_id=object_id,
        object_type=object_type,
        scope=scope,
    )


def preview_object_action_settings_copy(
    shell: _ObjectActionShell,
    action_id: str,
    *,
    source_scope: str,
    target_scope: str,
    source_song_id: str | None = None,
    source_version_id: str | None = None,
    target_song_id: str | None = None,
    target_version_id: str | None = None,
    keys: list[str] | None = None,
) -> dict[str, object]:
    """Preview partial settings copy between song-default and version scopes."""
    return shell._object_action_settings.preview_copy(
        action_id,
        source_scope=source_scope,
        target_scope=target_scope,
        source_song_id=source_song_id,
        source_version_id=source_version_id,
        target_song_id=target_song_id,
        target_version_id=target_version_id,
        keys=keys,
    )


def apply_object_action_settings_copy(
    shell: _ObjectActionShell,
    action_id: str,
    *,
    source_scope: str,
    target_scope: str,
    source_song_id: str | None = None,
    source_version_id: str | None = None,
    target_song_id: str | None = None,
    target_version_id: str | None = None,
    keys: list[str] | None = None,
) -> dict[str, object]:
    """Apply partial settings copy between song-default and version scopes."""
    preview = shell._object_action_settings.apply_copy(
        action_id,
        source_scope=source_scope,
        target_scope=target_scope,
        source_song_id=source_song_id,
        source_version_id=source_version_id,
        target_song_id=target_song_id,
        target_version_id=target_version_id,
        keys=keys,
    )
    if preview["changes"]:
        shell._is_dirty = True
        shell._clear_history()
    return preview
