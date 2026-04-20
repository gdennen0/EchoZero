from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import echozero.pipelines.templates  # noqa: F401
from echozero.application.presentation.models import (
    EventPresentation,
    LayerStatusPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.shared.enums import LayerKind, PlaybackStatus, SyncMode
from echozero.application.shared.ids import LayerId, ProjectId, SessionId, SongId, SongVersionId
from echozero.application.sync.adapters import MA3SyncBridge
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
)
from echozero.application.timeline.models import (
    Layer,
    LayerPresentationHints,
    LayerProvenance,
    LayerStatus,
    LayerSyncState,
    Timeline,
)
from echozero.application.timeline.object_actions import (
    ApplyCopySource,
    ChangeSessionScope,
    ObjectActionService,
    ObjectActionSettingsPlan,
    ObjectActionSettingsSession,
    PreviewCopySource,
    ReplaceSessionValues,
    RunSession,
    SaveAndRunSession,
    SaveSession,
    SetSessionFieldValue,
)
from echozero.domain.types import AudioData
from echozero.models.paths import ensure_installed_models_dir
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.processors import (
    DetectOnsetsProcessor,
    LoadAudioProcessor,
    PyTorchAudioClassifyProcessor,
    SeparateAudioProcessor,
)
from echozero.processors.binary_drum_classify import BinaryDrumClassifyProcessor
from echozero.services.orchestrator import AnalysisService
from echozero.ui.qt.app_shell_project_timeline import (
    AudioPresentationFields as _AudioPresentationFields,
)
from echozero.ui.qt.app_shell_project_timeline import (
    TimelinePresentationOverlay as _TimelinePresentationOverlay,
)
from echozero.ui.qt.app_shell_project_timeline import (
    apply_timeline_presentation_overlay as _apply_timeline_presentation_overlay,
)
from echozero.ui.qt.app_shell_project_timeline import (
    build_project_native_baseline_timeline as _build_project_native_baseline_timeline,
)
from echozero.ui.qt.app_shell_project_timeline import format_time as _format_time
from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture  # noqa: F401
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from echozero.ui.qt.timeline.style import TIMELINE_STYLE

_DIRTYING_INTENT_TYPES = (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
)


class AppShellRuntime:
    def __init__(
        self,
        *,
        project_storage: ProjectStorage,
        project_path: Path | None = None,
        sync_bridge: MA3SyncBridge | None = None,
        sync_service: SyncService | None = None,
        analysis_service: AnalysisService | None = None,
    ) -> None:
        self._sync_bridge = sync_bridge
        self._sync_service_override = sync_service
        self._analysis_service = analysis_service or _build_runtime_analysis_service()
        self._object_action_settings = ObjectActionService(
            project_storage_getter=lambda: self.project_storage,
            session_getter=lambda: self.session,
            presentation_getter=self.presentation,
            require_layer=self._require_layer,
            analysis_service=self._analysis_service,
        )
        self._is_dirty = False
        self._app = self._build_runtime_app(
            project_storage=project_storage,
            sync_bridge=sync_bridge,
            sync_service=sync_service,
        )
        self.project_storage = project_storage
        self.project_path = Path(project_path) if project_path is not None else None

    @property
    def runtime_audio(self):
        return self._app.runtime_audio

    @runtime_audio.setter
    def runtime_audio(self, value) -> None:
        self._app.runtime_audio = value

    @property
    def session(self) -> Session:
        return self._app.session

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty or self.project_storage.is_dirty()

    def presentation(self):
        return self._app.presentation()

    def add_layer(self, kind: LayerKind, title: str | None = None) -> TimelinePresentation:
        layer_kind = kind
        if not isinstance(layer_kind, LayerKind):
            try:
                layer_kind = LayerKind(str(layer_kind))
            except ValueError as exc:
                raise ValueError(f"Unsupported layer kind '{kind}'.") from exc

        layer_title = (title or "").strip()
        if not layer_title:
            layer_title = f"{layer_kind.value.title()} Layer"

        new_layer_id = LayerId(f"layer_{uuid.uuid4().hex[:12]}")
        timeline = self._app.timeline
        timeline.layers.append(
            Layer(
                id=new_layer_id,
                timeline_id=timeline.id,
                name=layer_title,
                kind=layer_kind,
                order_index=len(timeline.layers),
                status=LayerStatus(),
                provenance=LayerProvenance(),
                presentation_hints=LayerPresentationHints(),
                sync=LayerSyncState(),
            )
        )
        timeline.selection.selected_layer_id = new_layer_id
        timeline.selection.selected_layer_ids = [new_layer_id]
        timeline.selection.selected_take_id = None
        timeline.selection.selected_event_ids = []
        timeline.playback_target.layer_id = new_layer_id
        timeline.playback_target.take_id = None
        self._sync_runtime_audio_from_presentation(self.presentation())
        self._is_dirty = True
        return self.presentation()

    def dispatch(self, intent):
        presentation = self._app.dispatch(intent)
        if isinstance(intent, _DIRTYING_INTENT_TYPES):
            self._is_dirty = True
        return presentation

    def new_project(self, name: str = "EchoZero Project") -> None:
        working_dir_root = self.project_storage.working_dir.parent
        runtime_audio = self.runtime_audio
        self.project_storage.close()
        project_storage = ProjectStorage.create_new(
            name=name,
            working_dir_root=working_dir_root,
        )
        self.project_storage = project_storage
        self.project_path = None
        self._app = self._build_runtime_app(
            project_storage=project_storage,
            sync_bridge=self._sync_bridge,
            sync_service=self._sync_service_override,
            runtime_audio=runtime_audio,
        )
        self._is_dirty = False

    def save_project_as(self, path: str | Path) -> Path:
        target_path = Path(path)
        self.project_storage.save_as(target_path)
        self.project_path = target_path
        self._is_dirty = False
        return target_path

    def save_project(self) -> Path:
        if self.project_path is None:
            raise RuntimeError("save_project requires an existing project_path")
        return self.save_project_as(self.project_path)

    def open_project(self, path: str | Path) -> None:
        target_path = Path(path)
        working_dir_root = self.project_storage.working_dir.parent
        runtime_audio = self.runtime_audio
        prior_presentation = self.presentation()
        self.project_storage.close()
        project_storage = ProjectStorage.open(
            target_path,
            working_dir_root=working_dir_root,
        )
        self.project_storage = project_storage
        self.project_path = target_path
        self._app = self._build_runtime_app(
            project_storage=project_storage,
            sync_bridge=self._sync_bridge,
            sync_service=self._sync_service_override,
            runtime_audio=runtime_audio,
        )
        self._restore_timeline_targets(prior_presentation)
        self._is_dirty = False

    def add_song_from_path(self, title: str, audio_path: str | Path) -> TimelinePresentation:
        song, version = self.project_storage.import_song(
            title=title,
            audio_source=Path(audio_path),
        )
        self._refresh_from_storage(
            active_song_id=SongId(song.id),
            active_song_version_id=SongVersionId(version.id),
        )
        self._is_dirty = True
        return self.presentation()

    def run_object_action(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
    ) -> TimelinePresentation:
        """Execute one object-scoped action through the canonical runtime path."""
        self._object_action_settings.run(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
        )
        self._refresh_from_storage(
            active_song_id=self.session.active_song_id,
            active_song_version_id=self.session.active_song_version_id,
        )
        self._is_dirty = True
        return self.presentation()

    def open_object_action_session(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsSession:
        return self._object_action_settings.open_session(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )

    def dispatch_object_action_command(
        self,
        session_id: str,
        command: object,
    ) -> ObjectActionSettingsSession:
        session = self._object_action_settings.dispatch_command(session_id, command)
        if isinstance(command, (RunSession, SaveAndRunSession)):
            self._refresh_from_storage(
                active_song_id=self.session.active_song_id,
                active_song_version_id=self.session.active_song_version_id,
            )
            self._is_dirty = True
        elif isinstance(command, (SaveSession, ApplyCopySource)):
            self._is_dirty = True
        elif isinstance(
            command,
            (SetSessionFieldValue, ReplaceSessionValues, ChangeSessionScope, PreviewCopySource),
        ):
            self._is_dirty = self._is_dirty
        return session

    def save_object_action_settings(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        """Persist editable settings for one object action and return the refreshed plan."""
        self._is_dirty = True
        return self._object_action_settings.save(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )

    def describe_object_action(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        """Describe editable settings and locked bindings for one object action."""
        return self._object_action_settings.describe(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )

    def preview_object_action_settings_copy(
        self,
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
        return self._object_action_settings.preview_copy(
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
        self,
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
        preview = self._object_action_settings.apply_copy(
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
            self._is_dirty = True
        return preview

    def select_song(self, song_id: str | SongId) -> TimelinePresentation:
        song_record = self.project_storage.songs.get(str(song_id))
        if song_record is None:
            raise ValueError(f"SongRecord not found: {song_id}")

        active_version_id = (
            SongVersionId(song_record.active_version_id)
            if song_record.active_version_id is not None
            else None
        )
        self._refresh_from_storage(
            active_song_id=SongId(song_record.id),
            active_song_version_id=active_version_id,
        )
        return self.presentation()

    def switch_song_version(self, song_version_id: str | SongVersionId) -> TimelinePresentation:
        version_record = self.project_storage.song_versions.get(str(song_version_id))
        if version_record is None:
            raise ValueError(f"SongVersionRecord not found: {song_version_id}")

        song_record = self.project_storage.songs.get(version_record.song_id)
        if song_record is None:
            raise RuntimeError(f"SongRecord not found for SongVersionRecord '{version_record.id}'")

        if song_record.active_version_id != version_record.id:
            self.project_storage.songs.update(
                replace(song_record, active_version_id=version_record.id)
            )
            self.project_storage.commit()
            self.project_storage.dirty_tracker.mark_dirty(song_record.id)
            self._is_dirty = True

        self._refresh_from_storage(
            active_song_id=SongId(song_record.id),
            active_song_version_id=SongVersionId(version_record.id),
        )
        return self.presentation()

    def add_song_version(
        self,
        song_id: str | SongId,
        audio_path: str | Path,
        *,
        label: str | None = None,
        activate: bool = True,
    ) -> TimelinePresentation:
        song_record = self.project_storage.songs.get(str(song_id))
        if song_record is None:
            raise ValueError(f"SongRecord not found: {song_id}")

        version = self.project_storage.add_song_version(
            song_record.id,
            Path(audio_path),
            label=label,
            activate=activate,
        )
        updated_song = self.project_storage.songs.get(song_record.id)
        active_version_id = (
            SongVersionId(updated_song.active_version_id)
            if updated_song is not None and updated_song.active_version_id is not None
            else SongVersionId(version.id)
        )
        self._refresh_from_storage(
            active_song_id=SongId(song_record.id),
            active_song_version_id=active_version_id,
        )
        self._is_dirty = True
        return self.presentation()

    def extract_stems(self, layer_id) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_stems", object_id=layer_id, object_type="layer"
        )

    def extract_drum_events(self, layer_id) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_drum_events", object_id=layer_id, object_type="layer"
        )

    def classify_drum_events(self, layer_id, model_path: str | Path) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.classify_drum_events",
            {"model_path": model_path},
            object_id=layer_id,
            object_type="layer",
        )

    def extract_classified_drums(self, layer_id) -> TimelinePresentation:
        return self.run_object_action(
            "timeline.extract_classified_drums", object_id=layer_id, object_type="layer"
        )

    def preview_event_clip(self, *, layer_id, take_id=None, event_id) -> None:
        runtime_audio = self.runtime_audio
        if runtime_audio is None or not hasattr(runtime_audio, "preview_clip"):
            raise RuntimeError("This runtime does not support event clip preview.")

        clip = self._resolve_event_clip_preview(
            self.presentation(),
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
        )
        runtime_audio.preview_clip(
            clip["source_ref"],
            start_seconds=clip["start_seconds"],
            end_seconds=clip["end_seconds"],
            gain_db=clip["gain_db"],
        )

    def shutdown(self) -> None:
        if self.runtime_audio is not None:
            self.runtime_audio.shutdown()
        if self._sync_bridge is not None and hasattr(self._sync_bridge, "shutdown"):
            self._sync_bridge.shutdown()
        self.project_storage.close()

    def enable_sync(self, mode: SyncMode = SyncMode.MA3) -> SyncState:
        state = self._app.enable_sync(mode)
        self.session.sync_state = state
        return state

    def disable_sync(self) -> SyncState:
        state = self._app.disable_sync()
        self.session.sync_state = state
        return state

    @staticmethod
    def _build_runtime_app(
        *,
        project_storage: ProjectStorage,
        sync_bridge: MA3SyncBridge | None,
        sync_service: SyncService | None,
        runtime_audio=None,
    ) -> TimelineApplication:
        return build_runtime_timeline_application(
            project_storage=project_storage,
            sync_bridge=sync_bridge,
            sync_service=sync_service,
            runtime_audio=runtime_audio,
        )

    def _refresh_from_storage(
        self,
        *,
        active_song_id: SongId | None = None,
        active_song_version_id: SongVersionId | None = None,
    ) -> None:
        current_presentation = self.presentation()
        timeline, overlay, resolved_song_id, resolved_song_version_id = (
            _build_project_native_baseline_timeline(
                self.project_storage,
                active_song_id=active_song_id,
                active_song_version_id=active_song_version_id,
            )
        )
        self._app.presentation_enricher = (
            lambda presentation: _apply_timeline_presentation_overlay(
                presentation,
                overlay=overlay,
            )
        )
        self._app.replace_timeline(timeline)
        self._restore_timeline_targets(current_presentation)
        self.session.active_song_id = active_song_id or resolved_song_id
        self.session.active_song_version_id = active_song_version_id or resolved_song_version_id
        self.session.active_timeline_id = self._app.timeline.id
        self._sync_runtime_audio_from_presentation(self.presentation())

    def _sync_runtime_audio_from_presentation(self, presentation: TimelinePresentation) -> None:
        runtime_audio = self.runtime_audio
        if runtime_audio is None:
            return
        runtime_audio.build_for_presentation(presentation)
        if hasattr(runtime_audio, "snapshot_state"):
            self.session.playback_state = runtime_audio.snapshot_state(presentation)

    def _require_layer(self, layer_id) -> LayerPresentation:
        for layer in self.presentation().layers:
            if layer.layer_id == layer_id:
                return layer
        raise ValueError(f"Unknown layer_id: {layer_id}")

    @staticmethod
    def _resolve_event_clip_preview(
        presentation: TimelinePresentation,
        *,
        layer_id,
        take_id,
        event_id,
    ) -> dict[str, object]:
        for layer in presentation.layers:
            if layer.layer_id != layer_id:
                continue
            if take_id in (None, layer.main_take_id):
                for event in layer.events:
                    if event.event_id == event_id:
                        source_ref = layer.playback_source_ref or layer.source_audio_path
                        if not source_ref:
                            raise RuntimeError("Selected event does not have a source audio clip.")
                        return {
                            "source_ref": source_ref,
                            "start_seconds": float(event.start),
                            "end_seconds": float(event.end),
                            "gain_db": float(layer.gain_db),
                        }
            for take in layer.takes:
                if take.take_id != take_id:
                    continue
                for event in take.events:
                    if event.event_id == event_id:
                        source_ref = (
                            take.playback_source_ref
                            or take.source_audio_path
                            or layer.playback_source_ref
                            or layer.source_audio_path
                        )
                        if not source_ref:
                            raise RuntimeError("Selected event does not have a source audio clip.")
                        return {
                            "source_ref": source_ref,
                            "start_seconds": float(event.start),
                            "end_seconds": float(event.end),
                            "gain_db": float(layer.gain_db),
                        }
        raise ValueError(f"Unknown event clip preview target: {event_id}")

    def _require_active_song_version_id(self, action_name: str) -> str:
        song_version_id = self.session.active_song_version_id
        if song_version_id is None:
            raise RuntimeError(f"{action_name} requires an active song version.")
        return str(song_version_id)

    @staticmethod
    def _validate_drum_derived_audio_layer(layer: LayerPresentation, *, action_name: str) -> None:
        if layer.kind is not LayerKind.AUDIO:
            raise ValueError(
                f"{action_name} requires an audio layer, got {layer.kind.name.lower()}."
            )
        if not layer.source_audio_path:
            raise RuntimeError(
                f"{action_name} requires a source audio path on the selected layer."
            )

        title_lower = layer.title.lower()
        source_label = layer.status.source_label if layer.status is not None else ""
        source_label_lower = source_label.lower()
        badges = {str(badge).strip().lower() for badge in layer.badges}
        if (
            "drum" not in title_lower
            and "drums" not in badges
            and "drum" not in source_label_lower
        ):
            raise NotImplementedError(
                f"{action_name} currently runs only from drum-derived audio layers. "
                "Select a drums layer produced by stem separation."
            )

    def _restore_timeline_targets(self, prior_presentation: TimelinePresentation) -> None:
        current_presentation = self.presentation()
        selected_layer_id = self._resolve_preserved_selected_layer_id(
            prior_presentation, current_presentation
        )
        selected_take_id = self._resolve_take_id(
            current_presentation,
            layer_id=selected_layer_id,
            take_id=prior_presentation.selected_take_id,
        )
        active_playback_layer_id = self._resolve_preserved_active_playback_layer_id(
            prior_presentation,
            current_presentation,
        )
        active_playback_take_id = self._resolve_take_id(
            current_presentation,
            layer_id=active_playback_layer_id,
            take_id=prior_presentation.active_playback_take_id,
        )
        timeline = self._app.timeline
        timeline.selection.selected_layer_id = selected_layer_id
        timeline.selection.selected_layer_ids = (
            [selected_layer_id] if selected_layer_id is not None else []
        )
        timeline.selection.selected_take_id = selected_take_id
        timeline.selection.selected_event_ids = []
        timeline.playback_target.layer_id = active_playback_layer_id
        timeline.playback_target.take_id = active_playback_take_id

    @staticmethod
    def _resolve_preserved_selected_layer_id(
        prior_presentation: TimelinePresentation,
        current_presentation: TimelinePresentation,
    ) -> LayerId | None:
        if prior_presentation.selected_layer_id is not None and AppShellRuntime._has_layer(
            current_presentation, prior_presentation.selected_layer_id
        ):
            return prior_presentation.selected_layer_id
        return current_presentation.selected_layer_id

    @staticmethod
    def _resolve_preserved_active_playback_layer_id(
        prior_presentation: TimelinePresentation,
        current_presentation: TimelinePresentation,
    ) -> LayerId | None:
        if prior_presentation.active_playback_layer_id is not None and AppShellRuntime._has_layer(
            current_presentation, prior_presentation.active_playback_layer_id
        ):
            return prior_presentation.active_playback_layer_id
        if current_presentation.active_playback_layer_id is not None:
            return current_presentation.active_playback_layer_id
        return current_presentation.selected_layer_id

    @staticmethod
    def _resolve_take_id(
        presentation: TimelinePresentation,
        *,
        layer_id: LayerId | None,
        take_id: TakeId | None,
    ) -> TakeId | None:
        if layer_id is None or take_id is None:
            return None
        for layer in presentation.layers:
            if layer.layer_id != layer_id:
                continue
            if layer.main_take_id == take_id:
                return take_id
            if any(take.take_id == take_id for take in layer.takes):
                return take_id
            return None
        return None

    @staticmethod
    def _has_layer(presentation: TimelinePresentation, layer_id: LayerId) -> bool:
        return any(layer.layer_id == layer_id for layer in presentation.layers)


def build_app_shell(
    *,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
    analysis_service: AnalysisService | None = None,
    working_dir_root: Path | None = None,
    initial_project_name: str = "EchoZero Project",
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
    )


def _build_runtime_analysis_service() -> AnalysisService:
    return AnalysisService(
        get_registry(),
        {
            "LoadAudio": LoadAudioProcessor(),
            "SeparateAudio": SeparateAudioProcessor(),
            "DetectOnsets": DetectOnsetsProcessor(),
            "PyTorchAudioClassify": PyTorchAudioClassifyProcessor(),
            "BinaryDrumClassify": BinaryDrumClassifyProcessor(),
        },
    )
