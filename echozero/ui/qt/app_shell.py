from __future__ import annotations

from echozero.application.mixer.models import MixerState
from echozero.application.playback.models import PlaybackState
from echozero.application.session.models import Session
from echozero.application.shared.enums import PlaybackStatus, SyncMode
from echozero.application.shared.ids import ProjectId, SessionId, SongId, SongVersionId
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter, MA3SyncBridge
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
from echozero.ui.qt.timeline.demo_app import DemoTimelineApp, build_demo_app
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture


def build_app_shell(
    *,
    use_demo_fixture: bool = False,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
) -> DemoTimelineApp:
    if use_demo_fixture:
        return build_demo_app(sync_bridge=sync_bridge, sync_service=sync_service)

    presentation = load_realistic_timeline_fixture()
    session = Session(
        id=SessionId("session_demo"),
        project_id=ProjectId("project_demo"),
        active_song_id=SongId("song_demo"),
        active_song_version_id=SongVersionId("song_version_demo"),
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
    )
