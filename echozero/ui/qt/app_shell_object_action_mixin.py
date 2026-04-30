"""Object-action facade mixin for the Qt app shell runtime.
Exists to isolate object-action sessions, operation access, and action shortcuts.
Connects AppShellRuntime to the app-owned object-action and operation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.ids import EventId, LayerId, SongId, SongVersionId, TakeId
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.object_actions import (
    ObjectActionService,
    ObjectActionSettingsPlan,
    ObjectActionSettingsSession,
)
from echozero.application.timeline.operation_progress_service import (
    OperationProgressService,
    OperationProgressState,
)
from echozero.ui.qt.app_shell_object_actions import (
    apply_object_action_settings_copy,
    consume_operation_presentation_update,
    describe_object_action,
    dispatch_object_action_command,
    get_operation_state,
    open_object_action_session,
    preview_object_action_settings_copy,
    request_object_action_run,
    run_object_action,
    save_object_action_settings,
    wait_for_operation,
)
from echozero.ui.qt.app_shell_runtime_support import RuntimeSupportShell, preview_event_clip


class AppShellObjectActionShell(RuntimeSupportShell, Protocol):
    _app: TimelineApplication
    _is_dirty: bool
    _last_pipeline_run_revision: int
    _object_action_settings: ObjectActionService
    _pipeline_runs: OperationProgressService

    @property
    def session(self) -> Session: ...

    def presentation(self) -> TimelinePresentation: ...

    def _clear_history(self) -> None: ...

    def _refresh_from_storage(
        self,
        *,
        active_song_id: SongId | None = None,
        active_song_version_id: SongVersionId | None = None,
    ) -> None: ...

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
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
    ) -> TimelinePresentation: ...


class AppShellObjectActionMixin:
    def run_object_action(
        self: AppShellObjectActionShell,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
    ) -> TimelinePresentation:
        return run_object_action(
            self,
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
        )

    def request_object_action_run(
        self: AppShellObjectActionShell,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> str:
        return request_object_action_run(
            self,
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            persist_scope=persist_scope,
        )

    def wait_for_operation(
        self: AppShellObjectActionShell,
        operation_id: str,
        *,
        timeout: float | None = None,
    ) -> OperationProgressState:
        return wait_for_operation(self, operation_id, timeout=timeout)

    def get_operation_state(
        self: AppShellObjectActionShell,
        operation_id: str,
    ) -> OperationProgressState | None:
        return get_operation_state(self, operation_id)

    def consume_operation_presentation_update(
        self: AppShellObjectActionShell,
    ) -> TimelinePresentation | None:
        return consume_operation_presentation_update(self)

    def open_object_action_session(
        self: AppShellObjectActionShell,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsSession:
        return open_object_action_session(
            self,
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )

    def dispatch_object_action_command(
        self: AppShellObjectActionShell,
        session_id: str,
        command: object,
    ) -> ObjectActionSettingsSession:
        return dispatch_object_action_command(self, session_id, command)

    def save_object_action_settings(
        self: AppShellObjectActionShell,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        return save_object_action_settings(
            self,
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )

    def describe_object_action(
        self: AppShellObjectActionShell,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        return describe_object_action(
            self,
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )

    def preview_object_action_settings_copy(
        self: AppShellObjectActionShell,
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
        return preview_object_action_settings_copy(
            self,
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
        self: AppShellObjectActionShell,
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
        return apply_object_action_settings_copy(
            self,
            action_id,
            source_scope=source_scope,
            target_scope=target_scope,
            source_song_id=source_song_id,
            source_version_id=source_version_id,
            target_song_id=target_song_id,
            target_version_id=target_version_id,
            keys=keys,
        )

    def extract_stems(
        self: AppShellObjectActionShell,
        layer_id: LayerId,
    ) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_stems",
            object_id=layer_id,
            object_type="layer",
        )

    def extract_drum_events(
        self: AppShellObjectActionShell,
        layer_id: LayerId,
    ) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_drum_events",
            object_id=layer_id,
            object_type="layer",
        )

    def extract_song_drum_events(
        self: AppShellObjectActionShell,
        layer_id: LayerId,
    ) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_song_drum_events",
            object_id=layer_id,
            object_type="layer",
        )

    def extract_song_sections(
        self: AppShellObjectActionShell,
        layer_id: LayerId,
    ) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_song_sections",
            object_id=layer_id,
            object_type="layer",
        )

    def classify_drum_events(
        self: AppShellObjectActionShell,
        layer_id: LayerId,
        model_path: str | Path,
    ) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.classify_drum_events",
            {"model_path": model_path},
            object_id=layer_id,
            object_type="layer",
        )

    def extract_classified_drums(
        self: AppShellObjectActionShell,
        layer_id: LayerId,
    ) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_classified_drums",
            object_id=layer_id,
            object_type="layer",
        )

    def preview_event_clip(
        self: AppShellObjectActionShell,
        *,
        layer_id: LayerId,
        take_id: TakeId | None = None,
        event_id: EventId,
    ) -> None:
        preview_event_clip(
            self,
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
        )
