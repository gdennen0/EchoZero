from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import echozero.pipelines.templates  # noqa: F401
from echozero.application.presentation.models import EventPresentation, LayerStatusPresentation, TakeActionPresentation, TakeLanePresentation
from echozero.application.presentation.models import TimelinePresentation
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
from echozero.application.timeline.models import Layer, LayerPresentationHints, LayerProvenance, LayerStatus, LayerSyncState, Timeline
from echozero.domain.types import AudioData
from echozero.persistence.session import ProjectStorage
from echozero.inference_eval.runtime_preflight import resolve_runtime_model_path
from echozero.models.paths import ensure_installed_models_dir
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles
from echozero.pipelines.registry import get_registry
from echozero.processors import DetectOnsetsProcessor, LoadAudioProcessor, PyTorchAudioClassifyProcessor, SeparateAudioProcessor
from echozero.processors.binary_drum_classify import BinaryDrumClassifyProcessor
from echozero.result import is_err
from echozero.runtime_models.bundle_compat import upgrade_installed_runtime_bundles
from echozero.services.orchestrator import AnalysisService
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture  # noqa: F401
from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application
from echozero.ui.qt.app_shell_project_timeline import (
    AudioPresentationFields as _AudioPresentationFields,
    TimelinePresentationOverlay as _TimelinePresentationOverlay,
    apply_timeline_presentation_overlay as _apply_timeline_presentation_overlay,
    build_project_native_baseline_timeline as _build_project_native_baseline_timeline,
    format_time as _format_time,
)
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from echozero.ui.qt.timeline.style import TIMELINE_STYLE


_DIRTYING_INTENT_TYPES = (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
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
            raise RuntimeError(
                f"SongRecord not found for SongVersionRecord '{version_record.id}'"
            )

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
        layer = self._require_layer(layer_id)
        if self.session.active_song_version_id is None:
            raise RuntimeError("extract_stems requires an active song version.")
        if layer.kind is not LayerKind.AUDIO:
            raise ValueError(f"extract_stems requires an audio layer, got {layer.kind.name.lower()}.")
        if str(layer.layer_id) != "source_audio":
            raise NotImplementedError(
                "extract_stems currently runs only from the imported song layer. "
                "Derived-audio reruns are deferred until arbitrary-layer pipeline input is wired."
            )
        result = self._analysis_service.analyze(
            self.project_storage,
            str(self.session.active_song_version_id),
            "stem_separation",
        )
        if is_err(result):
            raise RuntimeError(f"extract_stems failed: {result.error}")
        self._refresh_from_storage(
            active_song_id=self.session.active_song_id,
            active_song_version_id=self.session.active_song_version_id,
        )
        self._is_dirty = True
        return self.presentation()

    def extract_drum_events(self, layer_id) -> TimelinePresentation:
        layer = self._require_layer(layer_id)
        song_version_id = self._require_active_song_version_id("extract_drum_events")
        self._validate_drum_derived_audio_layer(layer, action_name="extract_drum_events")

        result = self._analysis_service.analyze(
            self.project_storage,
            song_version_id,
            "onset_detection",
            bindings={"audio_file": layer.source_audio_path},
        )
        if is_err(result):
            raise RuntimeError(f"extract_drum_events failed: {result.error}")
        self._refresh_from_storage(
            active_song_id=self.session.active_song_id,
            active_song_version_id=self.session.active_song_version_id,
        )
        self._is_dirty = True
        return self.presentation()

    def classify_drum_events(self, layer_id, model_path: str | Path) -> TimelinePresentation:
        layer = self._require_layer(layer_id)
        song_version_id = self._require_active_song_version_id("classify_drum_events")
        self._validate_drum_derived_audio_layer(layer, action_name="classify_drum_events")

        resolved_model_path = resolve_runtime_model_path(model_path)
        if not str(resolved_model_path).strip():
            raise ValueError("classify_drum_events requires a non-empty model path.")

        result = self._analysis_service.analyze(
            self.project_storage,
            song_version_id,
            "drum_classification",
            bindings={
                "audio_file": layer.source_audio_path,
                "classify_model_path": str(resolved_model_path),
            },
        )
        if is_err(result):
            raise RuntimeError(f"classify_drum_events failed: {result.error}")
        self._refresh_from_storage(
            active_song_id=self.session.active_song_id,
            active_song_version_id=self.session.active_song_version_id,
        )
        self._is_dirty = True
        return self.presentation()

    def extract_classified_drums(self, layer_id) -> TimelinePresentation:
        layer = self._require_layer(layer_id)
        song_version_id = self._require_active_song_version_id("extract_classified_drums")
        self._validate_drum_derived_audio_layer(layer, action_name="extract_classified_drums")

        upgrade_installed_runtime_bundles(ensure_installed_models_dir())
        bundles = resolve_installed_binary_drum_bundles()
        result = self._analysis_service.analyze(
            self.project_storage,
            song_version_id,
            "extract_classified_drums",
            bindings={
                "audio_file": layer.source_audio_path,
                "kick_model_path": str(bundles["kick"].manifest_path),
                "snare_model_path": str(bundles["snare"].manifest_path),
            },
        )
        if is_err(result):
            raise RuntimeError(f"extract_classified_drums failed: {result.error}")
        self._refresh_from_storage(
            active_song_id=self.session.active_song_id,
            active_song_version_id=self.session.active_song_version_id,
        )
        self._is_dirty = True
        return self.presentation()

    def shutdown(self) -> None:
        if self.runtime_audio is not None:
            self.runtime_audio.shutdown()
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
        timeline, overlay, resolved_song_id, resolved_song_version_id = _build_project_native_baseline_timeline(
            self.project_storage,
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )
        self._app.presentation_enricher = lambda presentation: _apply_timeline_presentation_overlay(
            presentation,
            overlay=overlay,
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
            raise RuntimeError(f"{action_name} requires a source audio path on the selected layer.")

        title_lower = layer.title.lower()
        source_label = (layer.status.source_label if layer.status is not None else "")
        source_label_lower = source_label.lower()
        badges = {str(badge).strip().lower() for badge in layer.badges}
        if "drum" not in title_lower and "drums" not in badges and "drum" not in source_label_lower:
            raise NotImplementedError(
                f"{action_name} currently runs only from drum-derived audio layers. "
                "Select a drums layer produced by stem separation."
            )

    def _restore_timeline_targets(self, prior_presentation: TimelinePresentation) -> None:
        current_presentation = self.presentation()
        selected_layer_id = self._resolve_preserved_selected_layer_id(prior_presentation, current_presentation)
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
        timeline.selection.selected_layer_ids = [selected_layer_id] if selected_layer_id is not None else []
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
