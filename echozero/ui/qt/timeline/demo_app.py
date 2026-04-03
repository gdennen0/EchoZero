"""Runnable demo/testing loop for the new Stage Zero timeline shell."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from PyQt6.QtWidgets import QApplication

from echozero.application.mixer.models import AudibilityState, MixerState, LayerMixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import FollowMode, PlaybackStatus, SyncMode
from echozero.application.shared.ids import ProjectId, SessionId, SongId, SongVersionId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import Pause, Play, Seek, TimelineIntent, ToggleTakeSelector
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture
from echozero.ui.qt.timeline.widget import TimelineWidget


class DemoSessionService(SessionService):
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


class DemoTransportService(TransportService):
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


class DemoMixerService(MixerService):
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

    def resolve_audibility(self, layers) -> list[AudibilityState]:
        resolved: list[AudibilityState] = []
        for layer in layers:
            resolved.append(AudibilityState(layer_id=layer.layer_id, is_audible=not layer.muted, reason='normal'))
        return resolved


class DemoPlaybackService(PlaybackService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> PlaybackState:
        return self._session.playback_state

    def prepare(self, timeline) -> PlaybackState:
        return self._session.playback_state

    def update_runtime(self, timeline, transport: TransportState, audibility, sync: SyncState) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.PLAYING if transport.is_playing else PlaybackStatus.STOPPED
        return self._session.playback_state

    def stop(self) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.STOPPED
        return self._session.playback_state


class DemoSyncService(SyncService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> SyncState:
        return self._session.sync_state

    def connect(self) -> SyncState:
        self._session.sync_state.connected = True
        return self._session.sync_state

    def disconnect(self) -> SyncState:
        self._session.sync_state.connected = False
        self._session.sync_state.mode = SyncMode.NONE
        return self._session.sync_state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._session.sync_state.mode = mode
        return self._session.sync_state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport


@dataclass(slots=True)
class DemoTimelineApp:
    presentation_state: TimelinePresentation
    session: Session

    def presentation(self) -> TimelinePresentation:
        return self.presentation_state

    def dispatch(self, intent: TimelineIntent) -> TimelinePresentation:
        if isinstance(intent, Pause):
            self.session.transport_state.is_playing = False
        elif isinstance(intent, Play):
            self.session.transport_state.is_playing = True
        elif isinstance(intent, Seek):
            self.session.transport_state.playhead = max(0.0, intent.position)
        elif isinstance(intent, ToggleTakeSelector):
            layers = []
            for layer in self.presentation_state.layers:
                if layer.layer_id == intent.layer_id:
                    layers.append(replace(layer, is_expanded=not layer.is_expanded))
                else:
                    layers.append(layer)
            self.presentation_state = replace(self.presentation_state, layers=layers)
        self.presentation_state = replace(
            self.presentation_state,
            is_playing=self.session.transport_state.is_playing,
            playhead=self.session.transport_state.playhead,
            current_time_label=_fmt_time(self.session.transport_state.playhead),
        )
        return self.presentation_state


def _fmt_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def build_demo_presentation() -> TimelinePresentation:
    return load_realistic_timeline_fixture()


def build_demo_app() -> DemoTimelineApp:
    presentation = build_demo_presentation()
    session = Session(
        id=SessionId('session_demo'),
        project_id=ProjectId('project_demo'),
        active_song_id=SongId('song_demo'),
        active_song_version_id=SongVersionId('song_version_demo'),
        active_timeline_id=presentation.timeline_id,
        transport_state=TransportState(
            is_playing=presentation.is_playing,
            playhead=presentation.playhead,
            follow_mode=presentation.follow_mode,
        ),
        mixer_state=MixerState(),
        playback_state=PlaybackState(
            status=PlaybackStatus.PLAYING if presentation.is_playing else PlaybackStatus.STOPPED,
            backend_name='demo',
        ),
        sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref='show_manager'),
    )
    return DemoTimelineApp(presentation_state=presentation, session=session)


def main() -> int:
    app = QApplication(sys.argv)
    demo = build_demo_app()
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch)
    widget.resize(1440, 720)
    widget.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
