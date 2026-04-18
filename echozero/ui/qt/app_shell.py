from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import echozero.pipelines.templates  # noqa: F401
from echozero.application.presentation.models import EventPresentation, LayerStatusPresentation, TakeActionPresentation, TakeLanePresentation
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind, PlaybackStatus, SyncMode
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter, MA3SyncBridge
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.assembler import TimelineAssembler
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
from echozero.application.timeline.models import Event, Layer, LayerPresentationHints, LayerProvenance, LayerStatus, LayerSyncState, Take, Timeline
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.timeline.queries import TimelineQueries
from echozero.domain.types import AudioData, EventData, Event as DomainEvent
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
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import get_cached_waveform, register_waveform_from_audio_file


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


class AppRuntimeProfile(str, Enum):
    PRODUCTION = "production"
    TEST = "test"


@dataclass(slots=True)
class _AudioPresentationFields:
    waveform_key: str | None = None
    source_audio_path: str | None = None
    playback_source_ref: str | None = None


@dataclass(slots=True)
class _TimelinePresentationOverlay:
    project_title: str
    end_time_label: str
    layer_audio: dict[LayerId, _AudioPresentationFields]
    take_audio: dict[TakeId, _AudioPresentationFields]


class _RuntimeSessionService(SessionService):
    def __init__(self, session: Session):
        self._session = session

    def get_session(self) -> Session:
        return self._session

    def set_active_song(self, song_id):
        self._session.active_song_id = song_id
        return self._session

    def set_active_song_version(self, song_version_id):
        self._session.active_song_version_id = song_version_id
        return self._session

    def set_active_timeline(self, timeline_id):
        self._session.active_timeline_id = timeline_id
        return self._session


class _RuntimeTransportService(TransportService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> TransportState:
        return self._session.transport_state

    def play(self) -> TransportState:
        self._session.transport_state.is_playing = True
        return self._session.transport_state

    def pause(self) -> TransportState:
        self._session.transport_state.is_playing = False
        return self._session.transport_state

    def stop(self) -> TransportState:
        self._session.transport_state.is_playing = False
        self._session.transport_state.playhead = 0.0
        return self._session.transport_state

    def seek(self, position: float) -> TransportState:
        self._session.transport_state.playhead = max(0.0, position)
        return self._session.transport_state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._session.transport_state.loop_region = loop_region if enabled else None
        return self._session.transport_state


class _RuntimeMixerService(MixerService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> MixerState:
        return self._session.mixer_state

    def set_layer_state(self, layer_id, state: LayerMixerState) -> MixerState:
        self._session.mixer_state.layer_states[layer_id] = state
        return self._session.mixer_state

    def set_mute(self, layer_id, muted: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.mute = muted
        return self._session.mixer_state

    def set_solo(self, layer_id, soloed: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.solo = soloed
        return self._session.mixer_state

    def set_gain(self, layer_id, gain_db: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.gain_db = gain_db
        return self._session.mixer_state

    def set_pan(self, layer_id, pan: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.pan = pan
        return self._session.mixer_state

    def resolve_audibility(self, layers: list[Layer]) -> list[AudibilityState]:
        return [AudibilityState(layer_id=layer.id, is_audible=True, reason="normal") for layer in layers]


class _RuntimePlaybackService(PlaybackService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> PlaybackState:
        return self._session.playback_state

    def prepare(self, timeline) -> PlaybackState:
        return self._session.playback_state

    def update_runtime(self, timeline, transport: TransportState, audibility, sync: SyncState) -> PlaybackState:
        self._session.playback_state.status = (
            PlaybackStatus.PLAYING if transport.is_playing else PlaybackStatus.STOPPED
        )
        return self._session.playback_state

    def stop(self) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.STOPPED
        return self._session.playback_state


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
        if runtime_audio is None:
            runtime_audio = TimelineRuntimeAudioController()
        timeline, overlay, active_song_id, active_song_version_id = _build_project_native_baseline_timeline(
            project_storage
        )
        session = Session(
            id=SessionId(f"session_{project_storage.project.id}"),
            project_id=ProjectId(project_storage.project.id),
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
            active_timeline_id=timeline.id,
            transport_state=TransportState(
                is_playing=False,
                playhead=0.0,
            ),
            mixer_state=MixerState(),
            playback_state=PlaybackState(
                status=PlaybackStatus.STOPPED,
                backend_name="sounddevice",
            ),
            sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref="show_manager"),
        )

        runtime_sync_service: SyncService
        if sync_service is not None:
            runtime_sync_service = sync_service
        elif sync_bridge is not None:
            runtime_sync_service = MA3SyncAdapter(sync_bridge, state=session.sync_state, target_ref="show_manager")
        else:
            runtime_sync_service = InMemorySyncService(session.sync_state)

        assembler = TimelineAssembler()
        return TimelineApplication(
            timeline=timeline,
            session=session,
            orchestrator=TimelineOrchestrator(
                session_service=_RuntimeSessionService(session),
                transport_service=_RuntimeTransportService(session),
                mixer_service=_RuntimeMixerService(session),
                playback_service=_RuntimePlaybackService(session),
                sync_service=runtime_sync_service,
                assembler=assembler,
            ),
            queries=TimelineQueries(assembler=assembler),
            sync_service=runtime_sync_service,
            runtime_audio=runtime_audio,
            presentation_enricher=lambda presentation: _apply_timeline_presentation_overlay(
                presentation,
                overlay=overlay,
            ),
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
    profile: AppRuntimeProfile = AppRuntimeProfile.PRODUCTION,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
    analysis_service: AnalysisService | None = None,
    working_dir_root: Path | None = None,
    initial_project_name: str = "EchoZero Project",
) -> AppShellRuntime:
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


def _build_project_native_baseline_timeline(
    project_storage: ProjectStorage,
    *,
    active_song_id: SongId | None = None,
    active_song_version_id: SongVersionId | None = None,
) -> tuple[Timeline, _TimelinePresentationOverlay, SongId | None, SongVersionId | None]:
    project = project_storage.project
    songs = project_storage.songs.list_by_project(project.id)
    requested_song_id = str(active_song_id) if active_song_id is not None else None
    requested_version_id = str(active_song_version_id) if active_song_version_id is not None else None

    active_song = None
    active_version = None
    if requested_version_id is not None:
        active_version = project_storage.song_versions.get(requested_version_id)
        if active_version is not None:
            active_song = project_storage.songs.get(active_version.song_id)
    if active_song is None and requested_song_id is not None:
        active_song = next((song for song in songs if song.id == requested_song_id), None)
    if active_song is None:
        active_song = next((song for song in songs if song.active_version_id), None)
    if active_song is not None and active_version is None:
        if active_song.active_version_id is not None:
            active_version = project_storage.song_versions.get(active_song.active_version_id)

    # If requested IDs refer to a version for which no song is currently linked,
    # continue with project ordering to avoid breaking baseline startup flow.
    if active_song is None:
        return _build_empty_project_timeline(project_storage), _empty_overlay(project.name), None, None

    version = active_version
    if version is None and active_song.active_version_id is not None:
        version = project_storage.song_versions.get(active_song.active_version_id)
    if version is None:
        return (
            _build_empty_project_timeline(project_storage),
            _empty_overlay(project.name),
            SongId(active_song.id),
            None,
        )

    timeline_id = TimelineId(f"timeline_{project.id}")
    source_audio_path = _resolve_project_audio_path(project_storage, version.audio_file)
    waveform_key = _ensure_registered_waveform(f"song-{version.id}", source_audio_path)
    source_layer_id = LayerId("source_audio")
    source_take_id = TakeId(f"take_source_{version.id}")
    layers: list[Layer] = [
        Layer(
            id=source_layer_id,
            timeline_id=timeline_id,
            name=active_song.title,
            kind=LayerKind.AUDIO,
            order_index=0,
            takes=[
                Take(
                    id=source_take_id,
                    layer_id=source_layer_id,
                    name="Main",
                    source_ref="Imported track",
                )
            ],
            playback=replace(
                Layer(id=source_layer_id, timeline_id=timeline_id, name="", kind=LayerKind.AUDIO, order_index=0).playback,
                armed_source_ref=str(source_audio_path),
            ),
            presentation_hints=LayerPresentationHints(
                color=TIMELINE_STYLE.fixture.layer_color_tokens.get("song"),
            ),
        )
    ]
    layer_audio: dict[LayerId, _AudioPresentationFields] = {
        source_layer_id: _AudioPresentationFields(
            waveform_key=waveform_key,
            source_audio_path=str(source_audio_path),
            playback_source_ref=str(source_audio_path),
        )
    }
    take_audio: dict[TakeId, _AudioPresentationFields] = {
        source_take_id: _AudioPresentationFields(
            waveform_key=waveform_key,
            source_audio_path=str(source_audio_path),
            playback_source_ref=str(source_audio_path),
        )
    }
    for layer_record in project_storage.layers.list_by_version(version.id):
        layer, layer_fields, take_fields = _build_storage_layer(project_storage, timeline_id, layer_record)
        if layer is not None:
            layers.append(layer)
            layer_audio[layer.id] = layer_fields
            take_audio.update(take_fields)

    timeline = Timeline(
        id=timeline_id,
        song_version_id=SongVersionId(version.id),
        end=version.duration_seconds,
        layers=layers,
    )
    timeline.selection.selected_layer_id = source_layer_id
    timeline.selection.selected_layer_ids = [source_layer_id]
    timeline.playback_target.layer_id = source_layer_id

    return (
        timeline,
        _TimelinePresentationOverlay(
            project_title=project.name,
            end_time_label=_format_time(version.duration_seconds),
            layer_audio=layer_audio,
            take_audio=take_audio,
        ),
        SongId(active_song.id),
        SongVersionId(version.id),
    )


def _build_empty_project_timeline(project_storage: ProjectStorage) -> Timeline:
    project = project_storage.project
    timeline_id = TimelineId(f"timeline_{project.id}")
    return Timeline(
        id=timeline_id,
        song_version_id=SongVersionId("song_version_empty"),
        layers=[],
    )


def _resolve_project_audio_path(project_storage: ProjectStorage, audio_file: str) -> Path:
    raw_path = Path(audio_file)
    if raw_path.is_absolute():
        return raw_path
    return (project_storage.working_dir / raw_path).resolve()


def _ensure_registered_waveform(key: str, audio_path: Path) -> str | None:
    if not audio_path.exists():
        return None
    if get_cached_waveform(key) is None:
        register_waveform_from_audio_file(key, audio_path)
    return key


def _audio_presentation_fields(project_storage: ProjectStorage, take) -> _AudioPresentationFields:
    if not isinstance(take.data, AudioData):
        return _AudioPresentationFields()
    audio_path = _resolve_project_audio_path(project_storage, take.data.file_path)
    waveform_key = _ensure_registered_waveform(f"take-{take.id}", audio_path)
    return _AudioPresentationFields(
        waveform_key=waveform_key,
        source_audio_path=str(audio_path),
        playback_source_ref=str(audio_path),
    )


def _build_storage_layer(
    project_storage: ProjectStorage,
    timeline_id: TimelineId,
    layer_record,
) -> tuple[Layer | None, _AudioPresentationFields, dict[TakeId, _AudioPresentationFields]]:
    takes = project_storage.takes.list_by_layer(layer_record.id)
    if not takes:
        return None, _AudioPresentationFields(), {}
    main_take = next((take for take in takes if take.is_main), takes[0])
    main_kind = _take_kind(main_take)
    main_audio = _audio_presentation_fields(project_storage, main_take)
    take_audio: dict[TakeId, _AudioPresentationFields] = {}
    timeline_takes: list[Take] = []
    for take in takes:
        take_id = TakeId(str(take.id))
        timeline_takes.append(
            Take(
                id=take_id,
                layer_id=LayerId(str(layer_record.id)),
                name=take.label,
                events=_events_from_take(take),
                source_ref=_source_ref(take.source),
            )
        )
        take_audio[take_id] = _audio_presentation_fields(project_storage, take)

    layer_id = LayerId(str(layer_record.id))
    source_pipeline = layer_record.source_pipeline or {}
    provenance = layer_record.provenance or {}
    layer = Layer(
        id=layer_id,
        timeline_id=timeline_id,
        name=layer_record.name.title(),
        kind=main_kind,
        order_index=int(layer_record.order) + 1,
        takes=timeline_takes,
        playback=replace(
            Layer(id=layer_id, timeline_id=timeline_id, name="", kind=main_kind, order_index=0).playback,
            armed_source_ref=main_audio.playback_source_ref,
        ),
        status=LayerStatus(
            stale=bool(layer_record.state_flags.get("stale", False)),
            manually_modified=bool(layer_record.state_flags.get("manually_modified", False)),
            stale_reason=layer_record.state_flags.get("stale_reason"),
        ),
        provenance=LayerProvenance(
            source_layer_id=LayerId(str(provenance["source_layer_id"])) if provenance.get("source_layer_id") else None,
            source_song_version_id=SongVersionId(str(provenance["source_song_version_id"])) if provenance.get("source_song_version_id") else None,
            source_run_id=provenance.get("source_run_id"),
            pipeline_id=source_pipeline.get("pipeline_id") or provenance.get("pipeline_id"),
            output_name=source_pipeline.get("output_name") or provenance.get("output_name"),
        ),
        presentation_hints=LayerPresentationHints(
            visible=bool(layer_record.visible),
            locked=bool(layer_record.locked),
            expanded=len(timeline_takes) > 1,
            color=layer_record.color,
        ),
    )
    return layer, main_audio, take_audio


def _take_kind(take) -> LayerKind:
    if isinstance(take.data, EventData):
        return LayerKind.EVENT
    return LayerKind.AUDIO


def _events_from_take(take) -> list[Event]:
    if not isinstance(take.data, EventData):
        return []
    events: list[DomainEvent] = []
    for layer in take.data.layers:
        events.extend(layer.events)
    events.sort(key=lambda event: (event.time, event.duration, str(event.id)))
    return [
        Event(
            id=EventId(str(event.id)),
            take_id=TakeId(str(take.id)),
            start=float(event.time),
            end=float(event.time + max(event.duration, 0.08)),
            label=_event_label(event),
        )
        for event in events
    ]


def _event_label(event: DomainEvent) -> str:
    if isinstance(event.classifications, dict) and event.classifications:
        first_key = next(iter(event.classifications.keys()))
        value = event.classifications.get(first_key)
        if isinstance(value, str) and value.strip():
            return value.strip().title()
        if isinstance(first_key, str) and first_key.strip():
            return first_key.strip().replace("_", " ").title()
    return "Onset"


def _source_ref(source) -> str | None:
    if source is None:
        return None
    run_id = getattr(source, "run_id", "")
    block_type = getattr(source, "block_type", "")
    if run_id and block_type:
        return f"{block_type}:{str(run_id)[:8]}"
    if run_id:
        return str(run_id)
    return None


def _empty_overlay(project_title: str) -> _TimelinePresentationOverlay:
    return _TimelinePresentationOverlay(
        project_title=project_title,
        end_time_label="00:00.00",
        layer_audio={},
        take_audio={},
    )


def _apply_timeline_presentation_overlay(
    presentation: TimelinePresentation,
    *,
    overlay: _TimelinePresentationOverlay,
) -> TimelinePresentation:
    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        layer_fields = overlay.layer_audio.get(layer.layer_id, _AudioPresentationFields())
        takes = [
            replace(
                take,
                waveform_key=overlay.take_audio.get(take.take_id, _AudioPresentationFields()).waveform_key,
                source_audio_path=overlay.take_audio.get(take.take_id, _AudioPresentationFields()).source_audio_path,
                playback_source_ref=overlay.take_audio.get(take.take_id, _AudioPresentationFields()).playback_source_ref,
            )
            for take in layer.takes
        ]
        layers.append(
            replace(
                layer,
                badges=_layer_badges(layer.title, layer.kind),
                waveform_key=layer_fields.waveform_key,
                source_audio_path=layer_fields.source_audio_path,
                playback_source_ref=layer_fields.playback_source_ref,
                takes=takes,
            )
        )

    return replace(
        presentation,
        title=overlay.project_title,
        current_time_label=_format_time(presentation.playhead),
        end_time_label=overlay.end_time_label,
        layers=layers,
    )


def _layer_badges(name: str, kind: LayerKind) -> list[str]:
    badges = ["main", kind.value]
    if kind is LayerKind.AUDIO and name.strip().lower() != "imported song":
        badges.append("stem")
    if "drum" in name.strip().lower():
        badges.append("drums")
    return badges


def _format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"
