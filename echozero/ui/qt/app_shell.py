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
    SongId,
    SongVersionId,
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
from echozero.foundry.domain.review import ReviewPolarity
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
from echozero.ui.qt.app_shell_project_lifecycle import add_song_from_path as _add_song_from_path
from echozero.ui.qt.app_shell_project_lifecycle import add_song_version as _add_song_version
from echozero.ui.qt.app_shell_project_lifecycle import delete_song as _delete_song
from echozero.ui.qt.app_shell_project_lifecycle import delete_song_version as _delete_song_version
from echozero.ui.qt.app_shell_project_lifecycle import get_project_ma3_push_offset_seconds as _get_project_ma3_push_offset_seconds
from echozero.ui.qt.app_shell_project_lifecycle import list_ma3_timecode_pools as _list_ma3_timecode_pools
from echozero.ui.qt.app_shell_project_lifecycle import list_song_version_transfer_layers as _list_song_version_transfer_layers
from echozero.ui.qt.app_shell_project_lifecycle import move_song as _move_song
from echozero.ui.qt.app_shell_project_lifecycle import new_project as _new_project
from echozero.ui.qt.app_shell_project_lifecycle import open_project as _open_project
from echozero.ui.qt.app_shell_project_lifecycle import recover_project as _recover_project
from echozero.ui.qt.app_shell_project_lifecycle import refresh_from_storage as _refresh_from_storage
from echozero.ui.qt.app_shell_project_lifecycle import reorder_songs as _reorder_songs
from echozero.ui.qt.app_shell_project_lifecycle import save_project as _save_project
from echozero.ui.qt.app_shell_project_lifecycle import save_project_as as _save_project_as
from echozero.ui.qt.app_shell_project_lifecycle import select_song as _select_song
from echozero.ui.qt.app_shell_project_lifecycle import set_project_ma3_push_offset_seconds as _set_project_ma3_push_offset_seconds
from echozero.ui.qt.app_shell_project_lifecycle import set_song_version_ma3_timecode_pool as _set_song_version_ma3_timecode_pool
from echozero.ui.qt.app_shell_project_lifecycle import switch_song_version as _switch_song_version
from echozero.ui.qt.app_shell_project_review import (
    bind_phone_review_server_to_current_project,
    clear_project_review_runtime_bridge,
)
from echozero.ui.qt.app_shell_project_review import (
    ProjectReviewDatasetPaths,
    ProjectReviewLaunch,
    create_project_review_session,
    get_latest_project_review_dataset_version,
    latest_project_review_dataset_artifact_path,
    latest_project_review_dataset_folder,
    list_project_review_dataset_versions,
    open_project_review_session,
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
from echozero.ui.qt.timeline_review_sample_export import review_sample_export_root
_T = TypeVar("_T")


class StageZeroRuntimeController(
    AppShellEditingMixin,
    AppShellObjectActionMixin,
    AppShellSpecializedModelMixin,
):
    """Concrete Stage Zero runtime owner for shell lifecycle, session state, and collaborators."""

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

    def consume_sync_transport_update(self) -> dict[str, object] | None:
        bridge = self._sync_bridge
        if bridge is None:
            return None

        consume_latest = getattr(bridge, "consume_latest_transport_update", None)
        if callable(consume_latest):
            payload = consume_latest()
            if isinstance(payload, dict):
                return payload
            return None

        consume_next = getattr(bridge, "consume_transport_update", None)
        if callable(consume_next):
            payload = consume_next()
            if isinstance(payload, dict):
                return payload
        return None

    def prefers_low_latency_transport_poll(self) -> bool:
        """Hint UI runtime cadence when live MA3 transport sync is active."""

        bridge = self._sync_bridge
        if bridge is None:
            return False
        sync_state = getattr(self.session, "sync_state", None)
        mode_raw = getattr(sync_state, "mode", "")
        mode_value = getattr(mode_raw, "value", mode_raw)
        mode = str(mode_value or "").strip().lower()
        connected = bool(getattr(sync_state, "connected", False))
        return connected and mode == "ma3"

    def recent_ma3_osc_messages(self, *, limit: int = 12) -> list[dict[str, object]]:
        """Return one capped list of normalized inbound MA3 OSC messages."""

        bridge = self._sync_bridge
        if bridge is None:
            return []
        messages = getattr(bridge, "messages", None)
        if not isinstance(messages, list):
            return []

        max_items = max(1, int(limit))
        normalized: list[dict[str, object]] = []
        for message in messages[-max_items:]:
            message_type = str(getattr(message, "message_type", "") or "").strip()
            change = str(getattr(message, "change", "") or "").strip()
            raw_payload = str(getattr(message, "raw_payload", "") or "")
            timestamp = getattr(message, "timestamp", None)
            fields = getattr(message, "fields", {})
            normalized.append(
                {
                    "timestamp": timestamp,
                    "message_type": message_type,
                    "change": change,
                    "fields": fields if isinstance(fields, dict) else {},
                    "raw_payload": raw_payload,
                }
            )
        return normalized

    def clear_ma3_osc_messages(self) -> None:
        """Clear the runtime inbound MA3 OSC message history."""

        bridge = self._sync_bridge
        if bridge is None:
            return
        clear_messages = getattr(bridge, "clear_messages", None)
        if callable(clear_messages):
            clear_messages()
            return
        messages = getattr(bridge, "messages", None)
        if isinstance(messages, list):
            messages.clear()

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

    def new_project(self, name: str = "EchoZero Project") -> None:
        _new_project(self, name=name)

    def save_project_as(self, path: str | Path) -> Path:
        return _save_project_as(self, path)

    def save_project(self) -> Path:
        return _save_project(self)

    def open_project(self, path: str | Path) -> None:
        _open_project(self, path)

    def recover_project(self, path: str | Path) -> None:
        _recover_project(self, path)

    def add_song_from_path(
        self,
        title: str,
        audio_path: str | Path,
        *,
        run_import_pipeline: bool | None = None,
        import_pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> TimelinePresentation:
        return _add_song_from_path(
            self,
            title,
            audio_path,
            run_import_pipeline=run_import_pipeline,
            import_pipeline_action_ids=import_pipeline_action_ids,
        )

    def select_song(self, song_id: str | SongId) -> TimelinePresentation:
        return _select_song(self, song_id)

    def switch_song_version(self, song_version_id: str | SongVersionId) -> TimelinePresentation:
        return _switch_song_version(self, song_version_id)

    def add_song_version(
        self,
        song_id: str | SongId,
        audio_path: str | Path,
        *,
        label: str | None = None,
        activate: bool = True,
        transfer_layers: bool = False,
        transfer_layer_ids: list[str] | None = None,
        run_import_pipeline: bool | None = None,
        import_pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> TimelinePresentation:
        return _add_song_version(
            self,
            song_id,
            audio_path,
            label=label,
            activate=activate,
            transfer_layers=transfer_layers,
            transfer_layer_ids=transfer_layer_ids,
            run_import_pipeline=run_import_pipeline,
            import_pipeline_action_ids=import_pipeline_action_ids,
        )

    def list_song_version_transfer_layers(self, song_id: str | SongId) -> list[tuple[str, str]]:
        return _list_song_version_transfer_layers(self, song_id)

    def reorder_songs(self, song_ids: list[str]) -> TimelinePresentation:
        return _reorder_songs(self, song_ids)

    def move_song(self, song_id: str | SongId, *, steps: int) -> TimelinePresentation:
        return _move_song(self, song_id, steps=steps)

    def delete_song(self, song_id: str | SongId) -> TimelinePresentation:
        return _delete_song(self, song_id)

    def delete_song_version(self, song_version_id: str | SongVersionId) -> TimelinePresentation:
        return _delete_song_version(self, song_version_id)

    def list_ma3_timecode_pools(self) -> list[tuple[int, str | None]]:
        return _list_ma3_timecode_pools(self)

    def set_song_version_ma3_timecode_pool(
        self,
        song_version_id: str | SongVersionId,
        timecode_pool_no: int | None,
    ) -> TimelinePresentation:
        return _set_song_version_ma3_timecode_pool(self, song_version_id, timecode_pool_no)

    def get_project_ma3_push_offset_seconds(self) -> float:
        return _get_project_ma3_push_offset_seconds(self)

    def set_project_ma3_push_offset_seconds(
        self,
        offset_seconds: float,
    ) -> TimelinePresentation:
        return _set_project_ma3_push_offset_seconds(self, offset_seconds)

    def _refresh_from_storage(
        self,
        *,
        active_song_id: object | None = None,
        active_song_version_id: object | None = None,
    ) -> None:
        _refresh_from_storage(
            self,
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )

    def create_project_review_session(
        self,
        *,
        name: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
        review_mode: str | None = None,
        questionable_score_threshold: float | None = None,
        item_limit: int | None = None,
    ):
        return create_project_review_session(
            self,
            name=name,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            polarity=polarity,
            review_mode=review_mode,
            questionable_score_threshold=questionable_score_threshold,
            item_limit=item_limit,
        )

    def open_project_review_session(
        self,
        *,
        name: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        polarity: ReviewPolarity = ReviewPolarity.POSITIVE,
        review_mode: str | None = None,
        questionable_score_threshold: float | None = None,
        item_limit: int | None = None,
    ) -> ProjectReviewLaunch:
        return open_project_review_session(
            self,
            name=name,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            polarity=polarity,
            review_mode=review_mode,
            questionable_score_threshold=questionable_score_threshold,
            item_limit=item_limit,
        )

    def list_project_review_dataset_versions(
        self,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> list[ProjectReviewDatasetPaths]:
        return list_project_review_dataset_versions(self, queue_source_kind=queue_source_kind)

    def get_latest_project_review_dataset_version(
        self,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> ProjectReviewDatasetPaths | None:
        return get_latest_project_review_dataset_version(self, queue_source_kind=queue_source_kind)

    def latest_project_review_dataset_folder(
        self,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> Path:
        return latest_project_review_dataset_folder(self, queue_source_kind=queue_source_kind)

    def latest_project_review_dataset_artifact_path(
        self,
        *,
        queue_source_kind: str | None = "ez_project",
    ) -> Path:
        return latest_project_review_dataset_artifact_path(
            self,
            queue_source_kind=queue_source_kind,
        )

    def review_sample_export_root(self) -> Path:
        return review_sample_export_root(self.project_storage.working_dir)

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
) -> StageZeroRuntimeController:
    """Build the canonical in-memory app runtime used by the launcher and app-flow harness."""
    return StageZeroRuntimeController(
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


AppShellRuntime = StageZeroRuntimeController


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
