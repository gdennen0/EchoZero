"""Qt app shell runtime for the canonical EchoZero desktop surface.
Exists to compose project storage, timeline application behavior, and runtime services.
Connects launcher and app-flow entrypoints to the Stage Zero shell contract.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar, cast

import echozero.pipelines.templates  # noqa: F401
from echozero.application.presentation.models import (
    LayerPresentation,
    TimelinePresentation,
)
from echozero.application.actions import (
    ActionCoalescing,
    ActionCoalescingMode,
    ActionCommand,
    ActionGateway,
    ActionPriority,
)
from echozero.application.audio_hardware import (
    AudioHardwareApplyResult,
    AudioHardwareCoordinator,
    AudioHardwareSnapshot,
    audio_hardware_diagnostics_from_playback_state,
    requested_audio_hardware_from_runtime_config,
    resolved_audio_hardware_from_playback_state,
)
from echozero.application.operations import OperationKind, OperationLane
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
from echozero.application.timeline.intents import Pause, Play, Seek, Stop, TimelineIntent
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
from echozero.models.runtime_bundle_selection import (
    resolve_installed_binary_drum_bundles,
)  # noqa: F401
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.processors import (
    AudioFilterProcessor,
    DetectNoteContourProcessor,
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
from echozero.ui.qt.app_shell_project_lifecycle import (
    export_active_song_package as _export_active_song_package,
)
from echozero.ui.qt.app_shell_project_lifecycle import (
    get_project_ma3_push_offset_seconds as _get_project_ma3_push_offset_seconds,
)
from echozero.ui.qt.app_shell_project_lifecycle import (
    list_ma3_timecode_pools as _list_ma3_timecode_pools,
)
from echozero.ui.qt.app_shell_project_lifecycle import (
    import_song_package_into_project as _import_song_package_into_project,
)
from echozero.ui.qt.app_shell_project_lifecycle import (
    list_song_version_transfer_layers as _list_song_version_transfer_layers,
)
from echozero.ui.qt.app_shell_project_lifecycle import move_song as _move_song
from echozero.ui.qt.app_shell_project_lifecycle import new_project as _new_project
from echozero.ui.qt.app_shell_project_lifecycle import open_project as _open_project
from echozero.ui.qt.app_shell_project_lifecycle import recover_project as _recover_project
from echozero.ui.qt.app_shell_project_lifecycle import (
    refresh_from_storage as _refresh_from_storage,
)
from echozero.ui.qt.app_shell_project_lifecycle import rename_song as _rename_song
from echozero.ui.qt.app_shell_project_lifecycle import reorder_songs as _reorder_songs
from echozero.ui.qt.app_shell_project_lifecycle import save_project as _save_project
from echozero.ui.qt.app_shell_project_lifecycle import save_project_as as _save_project_as
from echozero.ui.qt.app_shell_project_lifecycle import select_song as _select_song
from echozero.ui.qt.app_shell_project_lifecycle import (
    set_project_ma3_push_offset_seconds as _set_project_ma3_push_offset_seconds,
)
from echozero.ui.qt.app_shell_project_lifecycle import (
    set_song_version_beat_anchor_seconds as _set_song_version_beat_anchor_seconds,
)
from echozero.ui.qt.app_shell_project_lifecycle import (
    set_song_version_ma3_timecode_pool as _set_song_version_ma3_timecode_pool,
)
from echozero.ui.qt.app_shell_project_lifecycle import switch_song_version as _switch_song_version
from echozero.ui.qt.app_shell_project_runtime_state import (
    TimelineViewportRuntimeState,
    load_project_runtime_state,
)
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
from echozero.ui.qt.app_shell_review_queue import (
    DeferredTimelineReviewPersistence,
    TimelineReviewPersistenceQueue,
)
from echozero.ui.qt.app_shell_selection_model_improvement import (
    AppShellSelectionModelImprovementMixin,
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
from echozero.ui.qt.timeline_command_runtime import TimelineCommandRuntime
from echozero.ui.qt.app_shell_specialized_model import AppShellSpecializedModelMixin
from echozero.ui.qt.timeline_review_sample_export import review_sample_export_root

_T = TypeVar("_T")


class StageZeroRuntimeController(
    AppShellEditingMixin,
    AppShellObjectActionMixin,
    AppShellSelectionModelImprovementMixin,
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
        self._deferred_timeline_review_persistence = TimelineReviewPersistenceQueue()
        self._timeline_command_runtime = TimelineCommandRuntime()
        self._history = UndoHistory(limit=_DEFAULT_HISTORY_LIMIT)
        self._is_dirty = False
        self._draft_layers: list[Layer] = []
        self._deferred_storage_sync_all = False
        self._deferred_storage_layer_ids: set[LayerId] = set()
        self._event_clipboard = []
        self._video_playback_controller = None
        self._action_gateway = ActionGateway()
        self._audio_hardware_snapshot = AudioHardwareSnapshot(revision=0)
        self._staged_project_runtime_presentation: TimelinePresentation | None = None
        self._staged_layer_header_width_px: int | None = None
        runtime_state = load_project_runtime_state(project_storage)
        self._song_version_viewports: dict[str, TimelineViewportRuntimeState] = dict(
            runtime_state.song_version_viewports or {}
        )
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
    def audio_hardware_snapshot(self) -> AudioHardwareSnapshot:
        """Return the latest app-visible audio hardware snapshot."""

        return self._audio_hardware_snapshot

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty or self.project_storage.is_dirty()

    def dispatch(self, intent: TimelineIntent) -> TimelinePresentation:
        """Dispatch one timeline intent through the action gateway compatibility seam."""

        command = self._action_command_for_timeline_intent(intent)
        accepted = self._action_gateway.accept(command)
        try:
            presentation = super().dispatch(intent)
        except Exception as exc:
            self._action_gateway.fail(command.command_id, str(exc))
            raise
        self._action_gateway.complete(accepted.command_id)
        return presentation

    @staticmethod
    def _action_command_for_timeline_intent(intent: TimelineIntent) -> ActionCommand:
        is_transport = isinstance(intent, (Play, Pause, Stop, Seek))
        command_type = f"timeline.{intent.__class__.__name__}"
        coalescing = (
            ActionCoalescing(ActionCoalescingMode.KEEP_LATEST, "transport.seek")
            if isinstance(intent, Seek)
            else ActionCoalescing()
        )
        return ActionCommand(
            command_type=command_type,
            lane=OperationLane.TRANSPORT if is_transport else OperationLane.APP,
            priority=ActionPriority.USER_BLOCKING if is_transport else ActionPriority.NORMAL,
            source="app_shell",
            coalescing=coalescing,
            operation_kind=OperationKind.TRANSPORT if is_transport else OperationKind.PIPELINE,
        )

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
        return self._with_app_settings_output_channels(self._app.presentation())

    def _with_app_settings_output_channels(
        self,
        presentation: TimelinePresentation,
    ) -> TimelinePresentation:
        if self._app_settings_service is None:
            return presentation
        resolve_channel_count = getattr(
            self._app_settings_service,
            "resolve_audio_output_channel_count",
            None,
        )
        if not callable(resolve_channel_count):
            return presentation
        try:
            settings_channels = int(resolve_channel_count() or 0)
        except Exception:
            return presentation
        if settings_channels <= int(presentation.playback_output_channels or 0):
            return presentation
        return replace(
            presentation,
            playback_output_channels=min(16, max(1, settings_channels)),
        )

    def consume_sync_transport_update(self) -> dict[str, object] | None:
        bridge = self._sync_bridge
        if bridge is None:
            return None

        consume_latest = getattr(bridge, "consume_latest_transport_update", None)
        if callable(consume_latest):
            payload = consume_latest()
            if isinstance(payload, dict):
                return payload

        consume_next = getattr(bridge, "consume_transport_update", None)
        if callable(consume_next):
            payload = consume_next()
            if isinstance(payload, dict):
                return payload
        return None

    def apply_sync_transport_update(
        self,
        payload: dict[str, object] | None,
        *,
        current_playhead_seconds: float | None = None,
        current_is_playing: bool | None = None,
    ) -> TimelinePresentation:
        """Apply one external transport update through the timeline application."""

        return self._app.apply_external_transport_update(
            payload,
            current_playhead_seconds=current_playhead_seconds,
            current_is_playing=current_is_playing,
        )

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
        if presentation is not None:
            song_version_id = str(presentation.active_song_version_id or "").strip()
            if song_version_id:
                viewport = TimelineViewportRuntimeState(
                    pixels_per_second=max(1.0, float(presentation.pixels_per_second)),
                    scroll_x=max(0.0, float(presentation.scroll_x)),
                    scroll_y=max(0.0, float(presentation.scroll_y)),
                )
                self._song_version_viewports[song_version_id] = viewport
                if str(self._app.timeline.song_version_id) == song_version_id:
                    self._app.timeline.viewport.pixels_per_second = viewport.pixels_per_second
                    self._app.timeline.viewport.scroll_x = viewport.scroll_x
                    self._app.timeline.viewport.scroll_y = viewport.scroll_y
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
        defer_storage_sync: bool = False,
        storage_layer_ids: list[LayerId] | None = None,
        history_layer_ids: list[LayerId] | None = None,
    ) -> _T:
        should_defer_storage = bool(defer_storage_sync) or (
            bool(storage_backed) and self._should_defer_storage_sync_for_live_transport()
        )
        return _run_undoable_operation(
            self,
            label=label,
            storage_backed=storage_backed,
            mark_dirty=mark_dirty,
            operation=operation,
            defer_storage_sync=should_defer_storage,
            storage_layer_ids=storage_layer_ids,
            history_layer_ids=history_layer_ids,
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

    def _should_defer_storage_sync_for_live_transport(self) -> bool:
        runtime_audio = self.runtime_audio
        if runtime_audio is None:
            return False
        try:
            return bool(runtime_audio.is_playing())
        except Exception:
            return False

    def _defer_storage_backed_timeline_sync(self) -> None:
        self._deferred_storage_sync_all = True
        self._deferred_storage_layer_ids.clear()
        active_song_version_id = self.session.active_song_version_id
        if active_song_version_id is not None:
            self.project_storage.dirty_tracker.mark_dirty(str(active_song_version_id))

    def _defer_storage_backed_layers_sync(self, layer_ids: list[LayerId]) -> None:
        if self._deferred_storage_sync_all:
            return
        for layer_id in layer_ids:
            if layer_id is not None:
                self._deferred_storage_layer_ids.add(LayerId(str(layer_id)))
        active_song_version_id = self.session.active_song_version_id
        if active_song_version_id is not None:
            self.project_storage.dirty_tracker.mark_dirty(str(active_song_version_id))

    def _flush_deferred_storage_sync(self) -> None:
        if self._deferred_storage_sync_all:
            self._deferred_storage_sync_all = False
            self._deferred_storage_layer_ids.clear()
            self._sync_storage_backed_timeline()
            return
        if not self._deferred_storage_layer_ids:
            return
        layer_ids = list(self._deferred_storage_layer_ids)
        self._deferred_storage_layer_ids.clear()
        self._sync_storage_backed_layers(layer_ids)

    def new_project(self, name: str = "EchoZero Project") -> None:
        _new_project(self, name=name)

    def save_project_as(self, path: str | Path) -> Path:
        self._flush_deferred_storage_sync()
        self.flush_deferred_review_persistence()
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

    def rename_song(self, song_id: str | SongId, title: str) -> TimelinePresentation:
        return _rename_song(self, song_id, title)

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

    def export_active_song_package(
        self,
        path: str | Path,
        *,
        song_version_id: str | SongVersionId | None = None,
    ):
        return _export_active_song_package(self, path, song_version_id=song_version_id)

    def import_song_package(
        self,
        path: str | Path,
        *,
        target_song_id: str | SongId | None = None,
        activate_import: bool = False,
    ):
        return _import_song_package_into_project(
            self,
            path,
            target_song_id=target_song_id,
            activate_import=activate_import,
        )

    def list_ma3_timecode_pools(self) -> list[tuple[int, str | None]]:
        return _list_ma3_timecode_pools(self)

    def set_song_version_ma3_timecode_pool(
        self,
        song_version_id: str | SongVersionId,
        timecode_pool_no: int | None,
    ) -> TimelinePresentation:
        return _set_song_version_ma3_timecode_pool(self, song_version_id, timecode_pool_no)

    def set_song_version_beat_anchor_seconds(
        self,
        song_version_id: str | SongVersionId,
        beat_anchor_seconds: float,
    ) -> TimelinePresentation:
        return _set_song_version_beat_anchor_seconds(
            self,
            song_version_id,
            beat_anchor_seconds,
        )

    def import_or_replace_song_video(self, video_path: str | Path) -> TimelinePresentation:
        """Import or replace the active song's video reference."""

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        if active_song_id is None:
            raise ValueError("Select a song before importing video.")
        self.project_storage.import_or_replace_song_video(str(active_song_id), Path(video_path))
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def import_or_replace_video_layer(
        self,
        video_path: str | Path,
        *,
        layer_id: str | LayerId | None = None,
    ) -> TimelinePresentation:
        """Import or replace a version-scoped video reference layer."""

        from echozero.persistence.video import import_video

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        if active_song_id is None or active_song_version_id is None:
            raise ValueError("Select a song version before importing video.")
        imported = import_video(Path(video_path), self.project_storage.working_dir)
        now = datetime.now(timezone.utc).isoformat()
        requested_layer_id = str(layer_id or "").strip()
        with self.project_storage.transaction():
            layer_record = (
                self.project_storage.layers.get(requested_layer_id)
                if requested_layer_id
                else None
            )
            if layer_record is None:
                requested_layer_id = f"layer_video_{uuid.uuid4().hex}"
                order_row = self.project_storage.db.execute(
                    'SELECT COALESCE(MAX("order"), 0) FROM layers WHERE song_version_id = ?',
                    (str(active_song_version_id),),
                ).fetchone()
                next_order = int(order_row[0] or 0) + 1
                self.project_storage.db.execute(
                    "INSERT INTO layers "
                    '(id, song_version_id, name, layer_type, color, "order", visible, locked, '
                    "parent_layer_id, source_pipeline, state_flags_json, provenance_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        requested_layer_id,
                        str(active_song_version_id),
                        "Video Reference",
                        "manual",
                        None,
                        next_order,
                        1,
                        0,
                        None,
                        json.dumps({"manual_video": True}),
                        json.dumps({"manual_kind": "reference", "reference_kind": "video"}),
                        "{}",
                        now,
                    ),
                )
                take_id = f"take_video_{uuid.uuid4().hex}"
                self.project_storage.db.execute(
                    "INSERT INTO takes "
                    "(id, layer_id, label, origin, is_main, is_archived, "
                    "source_json, data_json, created_at, notes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        take_id,
                        requested_layer_id,
                        "Video",
                        "user",
                        1,
                        0,
                        None,
                        json.dumps({"type": "EventData", "layers": []}),
                        now,
                        "Video reference layer.",
                    ),
                )
                object_id = f"object_{requested_layer_id}"
                content_id = f"content_{take_id}"
                self.project_storage.db.execute(
                    "INSERT INTO timeline_objects "
                    "(id, song_version_id, name, object_kind, main_content_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        object_id,
                        str(active_song_version_id),
                        "Video Reference",
                        "video_clip",
                        content_id,
                        now,
                    ),
                )
                self.project_storage.db.execute(
                    "INSERT INTO object_contents "
                    "(id, object_id, revision_id, content_kind, payload_json, "
                    "source_ref_json, analysis_build_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        content_id,
                        object_id,
                        f"revision_video_{imported.video_hash}",
                        "video_clip",
                        json.dumps(
                            _video_layer_payload_from_imported_video(imported, existing={})
                        ),
                        None,
                        None,
                        now,
                    ),
                )
            else:
                object_id = f"object_{requested_layer_id}"
                object_record = self.project_storage.timeline_objects.get(object_id)
                if object_record is None:
                    raise ValueError(f"Video layer object not found: {requested_layer_id}")
                content = self.project_storage.object_contents.get(object_record.main_content_id)
                if content is None or content.content_kind != "video_clip":
                    raise ValueError(f"Layer is not a video reference: {requested_layer_id}")
                payload = _video_layer_payload_from_imported_video(
                    imported,
                    existing=content.payload,
                )
                self.project_storage.db.execute(
                    "UPDATE object_contents SET revision_id = ?, payload_json = ? WHERE id = ?",
                    (
                        f"revision_video_{imported.video_hash}",
                        json.dumps(payload),
                        content.id,
                    ),
                )
            self.project_storage.dirty_tracker.mark_dirty(str(active_song_id))
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def remove_active_song_video(self) -> TimelinePresentation:
        """Remove the active song's video reference."""

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        if active_song_id is None:
            raise ValueError("Select a song before removing video.")
        self.project_storage.remove_song_video(str(active_song_id))
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def remove_video_layer(self, layer_id: str | LayerId) -> TimelinePresentation:
        """Remove one version-scoped video reference layer."""

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        resolved_layer_id = str(layer_id).strip()
        if active_song_id is None or active_song_version_id is None or not resolved_layer_id:
            raise ValueError("Select a video layer before removing video.")
        layer_record = self.project_storage.layers.get(resolved_layer_id)
        if layer_record is None:
            raise ValueError(f"Video layer not found: {resolved_layer_id}")
        with self.project_storage.transaction():
            for take in self.project_storage.takes.list_by_layer(resolved_layer_id):
                self.project_storage.takes.delete(str(take.id))
            self.project_storage.timeline_objects.delete(f"object_{resolved_layer_id}")
            self.project_storage.layers.delete(resolved_layer_id)
            self.project_storage.dirty_tracker.mark_dirty(str(active_song_id))
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def set_active_song_video_start_seconds(self, offset_seconds: float) -> TimelinePresentation:
        """Persist the active song version's video timeline offset."""

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        if active_song_version_id is None:
            raise ValueError("Select a song version before moving video.")
        self.project_storage.set_song_video_start_seconds(
            str(active_song_version_id),
            float(offset_seconds),
        )
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def set_video_layer_placement(
        self,
        layer_id: str | LayerId,
        *,
        start_seconds: float,
        trim_start_seconds: float,
        visible_duration_seconds: float,
        loop_enabled: bool,
    ) -> TimelinePresentation:
        """Persist placement fields on one version-scoped video layer."""

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        resolved_layer_id = str(layer_id).strip()
        if active_song_id is None or active_song_version_id is None or not resolved_layer_id:
            raise ValueError("Select a video layer before editing placement.")
        object_record = self.project_storage.timeline_objects.get(f"object_{resolved_layer_id}")
        if object_record is None:
            raise ValueError(f"Video layer object not found: {resolved_layer_id}")
        content = self.project_storage.object_contents.get(object_record.main_content_id)
        if content is None or content.content_kind != "video_clip":
            raise ValueError(f"Layer is not a video reference: {resolved_layer_id}")
        payload = dict(content.payload)
        payload.update(
            {
                "video_start_seconds": float(start_seconds),
                "video_trim_start_seconds": float(trim_start_seconds),
                "video_visible_duration_seconds": float(visible_duration_seconds),
                "video_loop_enabled": bool(loop_enabled),
            }
        )
        with self.project_storage.transaction():
            self.project_storage.db.execute(
                "UPDATE object_contents SET payload_json = ? WHERE id = ?",
                (json.dumps(payload), content.id),
            )
            self.project_storage.dirty_tracker.mark_dirty(str(active_song_id))
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def set_active_song_video_placement(
        self,
        *,
        start_seconds: float,
        trim_start_seconds: float,
        visible_duration_seconds: float,
        loop_enabled: bool,
    ) -> TimelinePresentation:
        """Persist the active song version's video trim, length, and loop state."""

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        if active_song_version_id is None:
            raise ValueError("Select a song version before editing video placement.")
        self.project_storage.set_song_video_placement(
            str(active_song_version_id),
            video_start_seconds=float(start_seconds),
            video_trim_start_seconds=float(trim_start_seconds),
            video_visible_duration_seconds=float(visible_duration_seconds),
            video_loop_enabled=bool(loop_enabled),
        )
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def set_active_song_video_loop_enabled(self, enabled: bool) -> TimelinePresentation:
        """Persist whether the active song version's video reference loops."""

        active_song_id = self.session.active_song_id
        active_song_version_id = self.session.active_song_version_id
        if active_song_version_id is None:
            raise ValueError("Select a song version before changing video loop state.")
        self.project_storage.set_song_video_loop_enabled(
            str(active_song_version_id),
            bool(enabled),
        )
        self._refresh_from_storage(
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._sync_video_playback_from_presentation()
        self._is_dirty = True
        return self.presentation()

    def open_video_window(self) -> None:
        """Open the synced video reference preview window."""

        from echozero.ui.qt.video_window import VideoPlaybackController

        if self._video_playback_controller is None:
            self._video_playback_controller = VideoPlaybackController(
                on_closed=self._on_video_window_closed
            )
            self._app.runtime_video = self._video_playback_controller
        self._video_playback_controller.sync_presentation(self.presentation())
        self._video_playback_controller.show()

    def update_runtime_video(self, song_seconds: float, is_playing: bool) -> None:
        """Update the video reference surface from the runtime audio clock."""

        if self._video_playback_controller is None:
            return
        self._app.update_runtime_video(
            song_seconds=float(song_seconds),
            is_playing=bool(is_playing),
            presentation=self.presentation(),
        )

    def _sync_video_playback_from_presentation(self) -> None:
        if self._video_playback_controller is None:
            return
        self._video_playback_controller.sync_presentation(self.presentation())

    def _on_video_window_closed(self) -> None:
        if self._video_playback_controller is not None:
            stop = getattr(self._video_playback_controller, "stop", None)
            if callable(stop):
                stop()
        self._video_playback_controller = None
        if self._app.runtime_video is not None:
            self._app.runtime_video = None

    def close_video_windows(self) -> None:
        """Close any app-owned video preview windows."""

        if self._video_playback_controller is None:
            return
        close = getattr(self._video_playback_controller, "close", None)
        if callable(close):
            close()
        self._video_playback_controller = None
        if self._app.runtime_video is not None:
            self._app.runtime_video = None

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
        return review_sample_export_root()

    def timeline_review_sample_export_folder(self) -> Path:
        return self.review_sample_export_root()

    def enqueue_deferred_review_persistence(
        self,
        request: DeferredTimelineReviewPersistence,
    ) -> None:
        self._deferred_timeline_review_persistence.enqueue(request)

    def flush_deferred_review_persistence(self) -> None:
        self._deferred_timeline_review_persistence.flush()

    def shutdown(self) -> None:
        self._flush_deferred_storage_sync()
        self._review_server_controller.stop()
        self.close_video_windows()
        clear_project_review_runtime_bridge(self)
        self._deferred_timeline_review_persistence.shutdown()
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
        self._apply_audio_output_config_through_gateway(config)

    def _apply_audio_output_config_through_gateway(
        self,
        config: AudioOutputRuntimeConfig | None,
    ) -> None:
        request = requested_audio_hardware_from_runtime_config(config)
        command = ActionCommand(
            command_type="audio.hardware.apply",
            lane=OperationLane.PREPARE,
            priority=ActionPriority.USER_BLOCKING,
            source="app_settings",
            coalescing=ActionCoalescing(
                ActionCoalescingMode.KEEP_LATEST,
                "audio.hardware.apply",
            ),
            operation_kind=OperationKind.PLAYBACK,
            diagnostics={"requested_device_id": request.device_id or "system_default"},
        )
        accepted = self._action_gateway.accept(command)

        def apply_request(_request):
            _apply_audio_output_config(self, config)
            snapshot_state = getattr(self.runtime_audio, "snapshot_state", None)
            state = (
                snapshot_state(self.presentation())
                if callable(snapshot_state)
                else self.session.playback_state
            )
            resolved = resolved_audio_hardware_from_playback_state(state)
            if resolved is None:
                raise RuntimeError("Audio hardware did not report a resolved output stream.")
            return AudioHardwareApplyResult(
                resolved=resolved,
                diagnostics=audio_hardware_diagnostics_from_playback_state(state),
            )

        coordinator = AudioHardwareCoordinator(
            apply_request,
            initial_snapshot=self._audio_hardware_snapshot,
        )
        try:
            self._audio_hardware_snapshot = coordinator.apply_request(
                request,
                request_id=command.command_id,
                operation_id=accepted.operation_id,
                generation_id=command.command_id,
            )
        except Exception as exc:
            self._action_gateway.fail(command.command_id, str(exc))
            raise
        if (
            self._audio_hardware_snapshot.operation is not None
            and self._audio_hardware_snapshot.operation.error
        ):
            self._action_gateway.fail(
                command.command_id,
                self._audio_hardware_snapshot.operation.error,
            )
            return
        self._action_gateway.complete(command.command_id)

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


def _video_layer_payload_from_imported_video(imported: object, *, existing: dict) -> dict:
    payload = dict(existing)
    metadata = getattr(imported, "metadata")
    payload.update(
        {
            "video_file": getattr(imported, "video_file"),
            "video_hash": getattr(imported, "video_hash"),
            "duration_seconds": getattr(metadata, "duration_seconds"),
            "extracted_audio_file": getattr(imported, "extracted_audio_file", None),
            "extracted_audio_hash": getattr(imported, "extracted_audio_hash", None),
            "width": getattr(metadata, "width", None),
            "height": getattr(metadata, "height", None),
            "fps": getattr(metadata, "fps", None),
        }
    )
    payload.setdefault("video_start_seconds", 0.0)
    payload.setdefault("video_trim_start_seconds", 0.0)
    payload.setdefault("video_visible_duration_seconds", None)
    payload.setdefault("video_loop_enabled", False)
    return payload


def _build_runtime_orchestrator() -> Orchestrator:
    return Orchestrator(
        get_registry(),
        {
            "LoadAudio": LoadAudioProcessor(),
            "AudioFilter": AudioFilterProcessor(),
            "SeparateAudio": SeparateAudioProcessor(),
            "DetectNoteContour": DetectNoteContourProcessor(),
            "DetectOnsets": DetectOnsetsProcessor(),
            "DetectSongSections": SongSectionsProcessor(),
            "PyTorchAudioClassify": PyTorchAudioClassifyProcessor(),
            "BinaryDrumClassify": BinaryDrumClassifyProcessor(),
        },
    )
