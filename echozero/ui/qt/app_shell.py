from __future__ import annotations

from pathlib import Path

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.mixer.models import MixerState
from echozero.application.playback.models import PlaybackState
from echozero.application.session.models import Session
from echozero.application.shared.enums import PlaybackStatus, SyncMode
from echozero.application.shared.ids import LayerId, ProjectId, SessionId, TimelineId
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
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.timeline.demo_app import DemoTimelineApp, build_demo_app
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture


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


class AppShellRuntime:
    def __init__(
        self,
        *,
        project_storage: ProjectStorage,
        project_path: Path | None = None,
        sync_bridge: MA3SyncBridge | None = None,
        sync_service: SyncService | None = None,
    ) -> None:
        self._sync_bridge = sync_bridge
        self._sync_service_override = sync_service
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

    def shutdown(self) -> None:
        if self.runtime_audio is not None:
            self.runtime_audio.shutdown()
        self.project_storage.close()

    @staticmethod
    def _build_runtime_app(
        *,
        project_storage: ProjectStorage,
        sync_bridge: MA3SyncBridge | None,
        sync_service: SyncService | None,
        runtime_audio=None,
    ) -> DemoTimelineApp:
        presentation = _build_project_native_baseline_presentation(project_storage)
        session = Session(
            id=SessionId(f"session_{project_storage.project.id}"),
            project_id=ProjectId(project_storage.project.id),
            active_song_id=None,
            active_song_version_id=None,
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


def build_app_shell(
    *,
    use_demo_fixture: bool = False,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
    working_dir_root: Path | None = None,
    initial_project_name: str = "EchoZero Project",
) -> DemoTimelineApp | AppShellRuntime:
    if use_demo_fixture:
        return build_demo_app(sync_bridge=sync_bridge, sync_service=sync_service)

    return AppShellRuntime(
        project_storage=ProjectStorage.create_new(
            name=initial_project_name,
            working_dir_root=working_dir_root,
        ),
        sync_bridge=sync_bridge,
        sync_service=sync_service,
    )


def _build_project_native_baseline_presentation(project_storage: ProjectStorage) -> TimelinePresentation:
    project = project_storage.project
    timeline_id = TimelineId(f"timeline_{project.id}")
    default_layer_id = LayerId(f"layer_{project.id}_default")
    default_layer = LayerPresentation(
        layer_id=default_layer_id,
        title="Layer 1",
        is_selected=True,
        is_expanded=True,
    )
    return TimelinePresentation(
        timeline_id=timeline_id,
        title=project.name,
        layers=[default_layer],
        selected_layer_id=default_layer_id,
        selected_layer_ids=[default_layer_id],
        current_time_label="00:00.00",
        end_time_label="00:00.00",
    )
