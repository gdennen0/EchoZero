"""Qt app shell runtime for the canonical EchoZero desktop surface.
Exists to compose project storage, timeline application behavior, and runtime services.
Connects launcher and app-flow entrypoints to the Stage Zero shell contract.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import echozero.pipelines.templates  # noqa: F401
from echozero.application.presentation.models import (
    LayerPresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.settings import AppSettingsService, AudioOutputRuntimeConfig
from echozero.application.shared.enums import SyncMode
from echozero.application.shared.ids import (
    LayerId,
)
from echozero.application.sync.adapters import MA3SyncBridge
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.history import UndoHistory
from echozero.application.timeline.models import (
    Layer,
)
from echozero.application.timeline.operation_progress_service import (
    OperationProgressService,
)
from echozero.application.timeline.object_actions import (
    ObjectActionService,
)
from echozero.domain.types import AudioData
from echozero.foundry.review_server_controller import (
    ReviewServerController,
    ReviewServerLaunch,
)
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles  # noqa: F401
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.processors import (
    AudioFilterProcessor,
    DetectOnsetsProcessor,
    SongSectionsProcessor,
    LoadAudioProcessor,
    PyTorchAudioClassifyProcessor,
    SeparateAudioProcessor,
)
from echozero.processors.binary_drum_classify import BinaryDrumClassifyProcessor
from echozero.services.orchestrator import Orchestrator
from echozero.ui.qt.app_shell_editing_mixin import AppShellEditingMixin
from echozero.ui.qt.app_shell_history import (
    DEFAULT_HISTORY_LIMIT as _DEFAULT_HISTORY_LIMIT,
)
from echozero.ui.qt.app_shell_history import (
    clear_history as _clear_history,
)
from echozero.ui.qt.app_shell_history import (
    redo as _redo,
)
from echozero.ui.qt.app_shell_history import (
    run_undoable_operation as _run_undoable_operation,
)
from echozero.ui.qt.app_shell_history import (
    undo as _undo,
)
from echozero.ui.qt.app_shell_object_action_mixin import AppShellObjectActionMixin
from echozero.ui.qt.app_shell_project_mixin import AppShellProjectMixin
from echozero.ui.qt.app_shell_project_review import (
    bind_phone_review_server_to_current_project,
    clear_project_review_runtime_bridge,
)
from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application
from echozero.ui.qt.app_shell_runtime_support import (
    RuntimeAudioController as _RuntimeAudioController,
)
from echozero.ui.qt.app_shell_runtime_support import (
    apply_audio_output_config as _apply_audio_output_config,
)
from echozero.ui.qt.app_shell_runtime_support import (
    apply_ma3_osc_runtime_config as _apply_ma3_osc_runtime_config,
)
from echozero.ui.qt.app_shell_runtime_support import (
    build_object_action_services as _build_object_action_services,
)
from echozero.ui.qt.app_shell_runtime_support import (
    preview_event_clip as _preview_event_clip,
)
from echozero.ui.qt.app_shell_runtime_support import (
    require_layer as _require_layer,
)
from echozero.ui.qt.app_shell_runtime_support import (
    select_active_source_layer as _select_active_source_layer,
)
from echozero.ui.qt.app_shell_runtime_support import (
    shutdown as _shutdown_runtime,
)
from echozero.ui.qt.app_shell_runtime_support import (
    sync_runtime_audio_from_presentation as _sync_runtime_audio_from_presentation,
)
from echozero.ui.qt.app_shell_storage_sync import (
    materialize_draft_layers as _materialize_draft_layers,
)
from echozero.ui.qt.app_shell_storage_sync import (
    persist_manual_layer as _persist_manual_layer,
)
from echozero.ui.qt.app_shell_storage_sync import (
    sync_storage_backed_layers as _sync_storage_backed_layers,
)
from echozero.ui.qt.app_shell_storage_sync import (
    store_manual_layer as _store_manual_layer,
)
from echozero.ui.qt.app_shell_storage_sync import (
    sync_runtime_take_records as _sync_runtime_take_records,
)
from echozero.ui.qt.app_shell_storage_sync import (
    sync_storage_backed_timeline as _sync_storage_backed_timeline,
)
from echozero.ui.qt.app_shell_specialized_model import AppShellSpecializedModelMixin
_T = TypeVar("_T")


class AppShellRuntime(
    AppShellEditingMixin,
    AppShellProjectMixin,
    AppShellObjectActionMixin,
    AppShellSpecializedModelMixin,
):
    _object_action_settings: ObjectActionService
    _pipeline_runs: OperationProgressService

    def __init__(
        self,
        *,
        project_storage: ProjectStorage,
        project_path: Path | None = None,
        sync_bridge: MA3SyncBridge | None = None,
        sync_service: SyncService | None = None,
        analysis_service: Orchestrator | None = None,
        app_settings_service: AppSettingsService | None = None,
        audio_output_config: AudioOutputRuntimeConfig | None = None,
    ) -> None:
        self._sync_bridge = sync_bridge
        self._sync_service_override = sync_service
        self._analysis_service = analysis_service or _build_runtime_orchestrator()
        self._app_settings_service = app_settings_service
        self._review_server_controller = ReviewServerController()
        self._history = UndoHistory(limit=_DEFAULT_HISTORY_LIMIT)
        self._is_dirty = False
        self._draft_layers: list[Layer] = []
        self._staged_project_runtime_presentation: TimelinePresentation | None = None
        self._staged_layer_header_width_px: int | None = None
        self._app: TimelineApplication = build_runtime_timeline_application(
            project_storage=project_storage,
            sync_bridge=sync_bridge,
            sync_service=sync_service,
            audio_output_config=audio_output_config,
        )
        self.project_storage = project_storage
        self.project_path = Path(project_path) if project_path is not None else None
        self._last_pipeline_run_revision = 0
        self._build_object_action_services()

    @property
    def runtime_audio(self) -> _RuntimeAudioController | None:
        return cast(_RuntimeAudioController | None, self._app.runtime_audio)

    @runtime_audio.setter
    def runtime_audio(self, value: _RuntimeAudioController | None) -> None:
        self._app.runtime_audio = value

    @property
    def session(self) -> Session:
        return self._app.session

    @property
    def app_settings_service(self) -> AppSettingsService | None:
        return self._app_settings_service

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty or self.project_storage.is_dirty()

    def is_phone_review_service_enabled(self) -> bool:
        """Return whether project-backed phone review is enabled for this runtime."""

        return self._review_server_controller.is_enabled

    def enable_phone_review_service(self) -> ReviewServerLaunch | None:
        """Enable the phone review server control path for this runtime."""

        self._review_server_controller.enable()
        return bind_phone_review_server_to_current_project(self)

    def disable_phone_review_service(self) -> None:
        """Disable the phone review server control path for this runtime."""

        self._review_server_controller.disable()

    def presentation(self) -> TimelinePresentation:
        return self._app.presentation()

    def stage_project_runtime_presentation(
        self,
        presentation: TimelinePresentation | None,
        *,
        layer_header_width_px: int | None = None,
    ) -> None:
        """Stage one presentation snapshot to persist on the next project save."""

        self._staged_project_runtime_presentation = presentation
        self._staged_layer_header_width_px = (
            int(layer_header_width_px)
            if isinstance(layer_header_width_px, int) and layer_header_width_px > 0
            else None
        )

    def can_undo(self) -> bool:
        return self._history.can_undo()

    def can_redo(self) -> bool:
        return self._history.can_redo()

    def undo_label(self) -> str | None:
        return self._history.undo_label()

    def redo_label(self) -> str | None:
        return self._history.redo_label()

    def _build_object_action_services(self) -> None:
        _build_object_action_services(self)

    def _clear_history(self) -> None:
        _clear_history(self)

    def undo(self) -> TimelinePresentation:
        return _undo(self)

    def redo(self) -> TimelinePresentation:
        return _redo(self)

    def _run_undoable_operation(
        self,
        *,
        label: str,
        storage_backed: bool,
        mark_dirty: bool,
        operation: Callable[[], _T],
    ) -> _T:
        return _run_undoable_operation(
            self,
            label=label,
            storage_backed=storage_backed,
            mark_dirty=mark_dirty,
            operation=operation,
        )

    def _store_manual_layer(self, layer: Layer) -> None:
        _store_manual_layer(self, layer)

    def _persist_manual_layer(
        self,
        layer: Layer,
        *,
        song_version_id: str,
        order: int | None = None,
    ) -> None:
        _persist_manual_layer(
            self,
            layer,
            song_version_id=song_version_id,
            order=order,
        )

    def _materialize_draft_layers(self, *, song_version_id: str) -> None:
        _materialize_draft_layers(self, song_version_id=song_version_id)

    def _select_active_source_layer(self) -> None:
        _select_active_source_layer(self)

    def _sync_runtime_take_records(self, layer: Layer) -> None:
        _sync_runtime_take_records(self, layer)

    def _sync_storage_backed_timeline(self) -> None:
        _sync_storage_backed_timeline(self)

    def _sync_storage_backed_layers(self, layer_ids: list[LayerId]) -> None:
        _sync_storage_backed_layers(self, layer_ids=layer_ids)

    def shutdown(self) -> None:
        self._review_server_controller.stop()
        clear_project_review_runtime_bridge(self)
        _shutdown_runtime(self)

    def enable_sync(self, mode: SyncMode = SyncMode.MA3) -> SyncState:
        state = self._app.enable_sync(mode)
        self.session.sync_state = state
        return state

    def disable_sync(self) -> SyncState:
        state = self._app.disable_sync()
        self.session.sync_state = state
        return state

    def apply_audio_output_config(
        self,
        config: AudioOutputRuntimeConfig | None,
    ) -> None:
        _apply_audio_output_config(self, config)

    def apply_ma3_osc_runtime_config(self) -> bool:
        return _apply_ma3_osc_runtime_config(self)

    def _sync_runtime_audio_from_presentation(self, presentation: TimelinePresentation) -> None:
        _sync_runtime_audio_from_presentation(self, presentation)

    def _require_layer(self, layer_id: LayerId) -> LayerPresentation:
        return _require_layer(self, layer_id)


def build_app_shell(
    *,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
    analysis_service: Orchestrator | None = None,
    working_dir_root: Path | None = None,
    initial_project_name: str = "EchoZero Project",
    app_settings_service: AppSettingsService | None = None,
    audio_output_config: AudioOutputRuntimeConfig | None = None,
) -> AppShellRuntime:
    """Build the canonical in-memory app runtime used by the launcher and app-flow harness."""
    return AppShellRuntime(
        project_storage=ProjectStorage.create_new(
            name=initial_project_name,
            working_dir_root=working_dir_root,
        ),
        sync_bridge=sync_bridge,
        sync_service=sync_service,
        analysis_service=analysis_service,
        app_settings_service=app_settings_service,
        audio_output_config=audio_output_config,
    )


def _build_runtime_orchestrator() -> Orchestrator:
    return Orchestrator(
        get_registry(),
        {
            "LoadAudio": LoadAudioProcessor(),
            "AudioFilter": AudioFilterProcessor(),
            "SeparateAudio": SeparateAudioProcessor(),
            "DetectOnsets": DetectOnsetsProcessor(),
            "DetectSongSections": SongSectionsProcessor(),
            "PyTorchAudioClassify": PyTorchAudioClassifyProcessor(),
            "BinaryDrumClassify": BinaryDrumClassifyProcessor(),
        },
    )
