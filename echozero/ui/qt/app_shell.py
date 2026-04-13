from __future__ import annotations

from enum import Enum
from pathlib import Path

import echozero.pipelines.templates  # noqa: F401
from echozero.application.presentation.models import EventPresentation, LayerStatusPresentation, TakeActionPresentation, TakeLanePresentation
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.mixer.models import MixerState
from echozero.application.playback.models import PlaybackState
from echozero.application.session.models import Session
from echozero.application.shared.enums import LayerKind, PlaybackStatus, SyncMode
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter, MA3SyncBridge
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
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
    ToggleMute,
    ToggleSolo,
    TriggerTakeAction,
    TrimEvent,
)
from echozero.domain.types import AudioData, EventData, Event as DomainEvent
from echozero.persistence.audio import resolve_audio_path
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.processors import LoadAudioProcessor, SeparateAudioProcessor
from echozero.result import is_err
from echozero.services.orchestrator import AnalysisService
from echozero.ui.qt.timeline.demo_app import DemoTimelineApp, build_demo_app
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture  # noqa: F401
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import register_waveform_from_audio_file


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
    ToggleMute,
    ToggleSolo,
    TriggerTakeAction,
    TrimEvent,
)


class AppRuntimeProfile(str, Enum):
    PRODUCTION = "production"
    TEST = "test"
    DEMO = "demo"


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
        self._is_dirty = False

    def add_song_from_path(self, title: str, audio_path: str | Path) -> TimelinePresentation:
        song, version = self.project_storage.import_song(
            title=title,
            audio_source=Path(audio_path),
            default_templates=[],
        )
        self._refresh_from_storage(
            active_song_id=SongId(song.id),
            active_song_version_id=SongVersionId(version.id),
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
        self._require_layer(layer_id)
        raise NotImplementedError(
            "extract_drum_events is not wired to the canonical app-shell runtime yet. "
            "Next hook should execute drum-event extraction from the active drums stem layer."
        )

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
    ) -> DemoTimelineApp:
        presentation, active_song_id, active_song_version_id = _build_project_native_baseline_presentation(
            project_storage
        )
        session = Session(
            id=SessionId(f"session_{project_storage.project.id}"),
            project_id=ProjectId(project_storage.project.id),
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
            active_timeline_id=presentation.timeline_id,
            transport_state=TransportState(
                is_playing=presentation.is_playing,
                playhead=presentation.playhead,
                follow_mode=presentation.follow_mode,
            ),
            mixer_state=MixerState(),
            playback_state=PlaybackState(
                status=PlaybackStatus.PLAYING if presentation.is_playing else PlaybackStatus.STOPPED,
                backend_name="demo",
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

        return DemoTimelineApp(
            presentation_state=presentation,
            session=session,
            sync_service=runtime_sync_service,
            runtime_audio=runtime_audio,
        )

    def _refresh_from_storage(
        self,
        *,
        active_song_id: SongId | None = None,
        active_song_version_id: SongVersionId | None = None,
    ) -> None:
        presentation, resolved_song_id, resolved_song_version_id = _build_project_native_baseline_presentation(
            self.project_storage
        )
        self._app.presentation_state = presentation
        self.session.active_song_id = active_song_id or resolved_song_id
        self.session.active_song_version_id = active_song_version_id or resolved_song_version_id
        self.session.active_timeline_id = presentation.timeline_id

    def _require_layer(self, layer_id) -> LayerPresentation:
        for layer in self.presentation().layers:
            if layer.layer_id == layer_id:
                return layer
        raise ValueError(f"Unknown layer_id: {layer_id}")


def build_app_shell(
    *,
    profile: AppRuntimeProfile = AppRuntimeProfile.PRODUCTION,
    use_demo_fixture: bool = False,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
    analysis_service: AnalysisService | None = None,
    working_dir_root: Path | None = None,
    initial_project_name: str = "EchoZero Project",
) -> DemoTimelineApp | AppShellRuntime:
    effective_profile = AppRuntimeProfile.DEMO if use_demo_fixture else profile

    if effective_profile == AppRuntimeProfile.DEMO:
        return build_demo_app(sync_bridge=sync_bridge, sync_service=sync_service)

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
        },
    )


def _build_project_native_baseline_presentation(
    project_storage: ProjectStorage,
) -> tuple[TimelinePresentation, SongId | None, SongVersionId | None]:
    project = project_storage.project
    songs = project_storage.songs.list_by_project(project.id)
    active_song = next((song for song in songs if song.active_version_id), None)
    if active_song is None:
        return _build_empty_project_presentation(project_storage), None, None

    version = project_storage.song_versions.get(active_song.active_version_id)
    if version is None:
        return _build_empty_project_presentation(project_storage), SongId(active_song.id), None

    timeline_id = TimelineId(f"timeline_{project.id}")
    source_audio_path = resolve_audio_path(project_storage.working_dir, version.audio_file)
    waveform_key = f"song-{version.id}"
    if source_audio_path.exists():
        register_waveform_from_audio_file(waveform_key, source_audio_path)

    layers: list[LayerPresentation] = [
        LayerPresentation(
            layer_id=LayerId("source_audio"),
            title=active_song.title,
            subtitle=f"Imported song · {Path(version.audio_file).name}",
            kind=LayerKind.AUDIO,
            is_selected=True,
            color=TIMELINE_STYLE.fixture.layer_color_tokens.get("song"),
            badges=["main", "audio"],
            waveform_key=waveform_key if source_audio_path.exists() else None,
            source_audio_path=str(source_audio_path),
            status=LayerStatusPresentation(source_label="Imported track"),
        )
    ]
    for layer_record in project_storage.layers.list_by_version(version.id):
        layer = _build_storage_layer_presentation(project_storage, layer_record)
        if layer is not None:
            layers.append(layer)

    return (
        TimelinePresentation(
            timeline_id=timeline_id,
            title=project.name,
            layers=layers,
            selected_layer_id=layers[0].layer_id,
            selected_layer_ids=[layers[0].layer_id],
            current_time_label="00:00.00",
            end_time_label=_format_time(version.duration_seconds),
        ),
        SongId(active_song.id),
        SongVersionId(version.id),
    )


def _build_empty_project_presentation(project_storage: ProjectStorage) -> TimelinePresentation:
    project = project_storage.project
    timeline_id = TimelineId(f"timeline_{project.id}")
    default_layer_id = LayerId(f"layer_{project.id}_default")
    return TimelinePresentation(
        timeline_id=timeline_id,
        title=project.name,
        layers=[
            LayerPresentation(
                layer_id=default_layer_id,
                title="Layer 1",
                is_selected=True,
                is_expanded=True,
            )
        ],
        selected_layer_id=default_layer_id,
        selected_layer_ids=[default_layer_id],
        current_time_label="00:00.00",
        end_time_label="00:00.00",
    )


def _build_storage_layer_presentation(project_storage: ProjectStorage, layer_record) -> LayerPresentation | None:
    takes = project_storage.takes.list_by_layer(layer_record.id)
    if not takes:
        return None
    main_take = next((take for take in takes if take.is_main), takes[0])
    main_kind = _take_kind(main_take)
    take_rows: list[TakeLanePresentation] = []
    for take in takes:
        if take.is_main:
            continue
        take_rows.append(
            TakeLanePresentation(
                take_id=TakeId(str(take.id)),
                name=take.label,
                kind=_take_kind(take),
                events=_event_presentations_from_take(take),
                source_ref=_source_ref(take.source),
                actions=[
                    TakeActionPresentation(action_id="overwrite_main", label="Overwrite Main"),
                    TakeActionPresentation(action_id="merge_main", label="Merge Main"),
                ],
                source_audio_path=str(take.data.file_path) if isinstance(take.data, AudioData) else None,
            )
        )
    source_audio_path = str(main_take.data.file_path) if isinstance(main_take.data, AudioData) else None
    return LayerPresentation(
        layer_id=LayerId(str(layer_record.id)),
        title=layer_record.name.title(),
        subtitle=f"Derived layer · {layer_record.layer_type}",
        main_take_id=TakeId(str(main_take.id)),
        kind=main_kind,
        is_expanded=bool(take_rows),
        events=_event_presentations_from_take(main_take),
        takes=take_rows,
        visible=bool(layer_record.visible),
        locked=bool(layer_record.locked),
        color=layer_record.color,
        badges=_layer_badges(layer_record.name, main_kind),
        source_audio_path=source_audio_path,
        status=LayerStatusPresentation(
            stale=bool(layer_record.state_flags.get("stale", False)),
            manually_modified=bool(layer_record.state_flags.get("manually_modified", False)),
            source_label=_source_label(layer_record),
        ),
    )


def _take_kind(take) -> LayerKind:
    if isinstance(take.data, EventData):
        return LayerKind.EVENT
    return LayerKind.AUDIO


def _event_presentations_from_take(take) -> list[EventPresentation]:
    if not isinstance(take.data, EventData):
        return []
    events: list[DomainEvent] = []
    for layer in take.data.layers:
        events.extend(layer.events)
    events.sort(key=lambda event: (event.time, event.duration, str(event.id)))
    return [
        EventPresentation(
            event_id=EventId(str(event.id)),
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


def _source_label(layer_record) -> str:
    source = layer_record.source_pipeline or {}
    pipeline = source.get("pipeline_id", "pipeline")
    output_name = source.get("output_name", layer_record.name)
    return f"{pipeline} · {output_name}"


def _layer_badges(name: str, kind: LayerKind) -> list[str]:
    badges = ["main", kind.value]
    if kind is LayerKind.AUDIO:
        badges.append("stem")
    if "drum" in name.strip().lower():
        badges.append("drums")
    return badges


def _format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"
