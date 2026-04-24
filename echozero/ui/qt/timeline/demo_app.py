"""Demo-only timeline fixture app for tests and screenshots.
Exists to build synthetic timeline states without project/runtime wiring.
Never use this module in the canonical user launch or app-shell runtime path.
"""

from __future__ import annotations

from pathlib import Path

from echozero.application.mixer.models import MixerState
from echozero.application.playback.models import PlaybackState
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.enums import PlaybackStatus, SyncMode
from echozero.application.shared.ids import ProjectId, SessionId, SongId, SongVersionId
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter, MA3SyncBridge
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
from echozero.ui.qt.timeline.demo_app_runtime import DemoTimelineApp
from echozero.ui.qt.timeline.demo_app_services import (
    DemoMixerService,
    DemoPlaybackService,
    DemoSessionService,
    DemoTransportService,
)
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController


def build_demo_presentation() -> TimelinePresentation:
    """Build the synthetic fixture presentation used by timeline-only tests."""
    return load_realistic_timeline_fixture()


def build_demo_app(
    *,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
) -> DemoTimelineApp:
    """Build the synthetic demo app used by timeline fixture tests only."""
    presentation = build_demo_presentation()
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
        runtime_sync_service = MA3SyncAdapter(
            sync_bridge, state=session.sync_state, target_ref="show_manager"
        )
    else:
        runtime_sync_service = InMemorySyncService(session.sync_state)

    return DemoTimelineApp(
        presentation_state=presentation,
        session=session,
        sync_service=runtime_sync_service,
    )


def build_real_data_demo_app(
    audio_path: str | Path,
    *,
    working_root: str | Path,
    song_title: str = "Doechii Nissan Altima",
    runtime_audio: TimelineRuntimeAudioController | None = None,
) -> tuple[DemoTimelineApp, object]:
    """Build a demo app around real data fixtures for explicit demo/test flows only."""
    from echozero.ui.qt.timeline.real_data_fixture import build_real_data_presentation

    presentation, summary = build_real_data_presentation(
        audio_path=audio_path,
        working_root=working_root,
        song_title=song_title,
    )
    app = build_demo_app()
    app.presentation_state = presentation
    app.session.active_timeline_id = presentation.timeline_id
    app.session.transport_state.is_playing = presentation.is_playing
    app.session.transport_state.playhead = presentation.playhead
    app.runtime_audio = runtime_audio or TimelineRuntimeAudioController()
    app.runtime_audio.build_for_presentation(presentation)
    return app, summary


__all__ = [
    "DemoMixerService",
    "DemoPlaybackService",
    "DemoSessionService",
    "DemoTimelineApp",
    "DemoTransportService",
    "build_demo_app",
    "build_demo_presentation",
    "build_real_data_demo_app",
]
