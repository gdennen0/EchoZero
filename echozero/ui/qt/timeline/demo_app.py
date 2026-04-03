"""Runnable demo/testing loop for the new Stage Zero timeline shell."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from PyQt6.QtWidgets import QApplication

from echozero.application.mixer.models import AudibilityState, MixerState, LayerMixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import FollowMode, LayerKind, PlaybackStatus, SyncMode
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import Pause, Play, Seek, TimelineIntent, ToggleTakeSelector
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService
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


def _event(event_id: str, start: float, end: float, label: str, color: str, selected: bool = False) -> EventPresentation:
    return EventPresentation(event_id=EventId(event_id), start=start, end=end, label=label, color=color, is_selected=selected)


def build_demo_presentation() -> TimelinePresentation:
    kick_main = [_event('e1', 0.50, 0.80, 'Kick', '#66a3ff'), _event('e2', 4.20, 4.45, 'Kick', '#66a3ff')]
    snare_main = [_event('e3', 1.20, 1.45, 'Snare', '#7fd1ae', selected=True), _event('e4', 4.90, 5.15, 'Snare', '#7fd1ae')]
    hihat_main = [_event('e5', 0.90, 1.00, 'Hat', '#f8c555'), _event('e6', 1.90, 2.00, 'Hat', '#f8c555')]
    clap_take = [_event('e7', 2.40, 2.55, 'Clap', '#ff8c78'), _event('e8', 6.20, 6.35, 'Clap', '#ff8c78')]

    layers = [
        LayerPresentation(
            layer_id=LayerId('layer_song'),
            title='Song',
            subtitle='Main song layer · imported audio',
            kind=LayerKind.AUDIO,
            is_selected=False,
            is_expanded=False,
            muted=False,
            soloed=False,
            color='#4da3ff',
            waveform_key='song-main',
            badges=['main', 'audio'],
            status=LayerStatusPresentation(source_label='Imported source', sync_label='No sync'),
        ),
        LayerPresentation(
            layer_id=LayerId('layer_drums'),
            title='Drums',
            subtitle='Stem main from separator',
            kind=LayerKind.AUDIO,
            is_selected=True,
            is_expanded=True,
            muted=False,
            soloed=True,
            color='#9b87f5',
            waveform_key='drums-main',
            badges=['main', 'audio'],
            status=LayerStatusPresentation(source_label='Derived from Song via stem_separation', sync_label='Main only'),
            takes=[TakeLanePresentation(take_id=TakeId('take_drums_rerun1'), name='Take 2 · separator rerun', kind=LayerKind.AUDIO, waveform_key='drums-take-2')],
        ),
        LayerPresentation(
            layer_id=LayerId('layer_bass'),
            title='Bass',
            subtitle='Stem main from separator',
            kind=LayerKind.AUDIO,
            is_selected=False,
            is_expanded=False,
            muted=False,
            soloed=False,
            color='#d68cff',
            waveform_key='bass-main',
            badges=['main', 'audio'],
            status=LayerStatusPresentation(source_label='Derived from Song via stem_separation', sync_label='No sync'),
        ),
        LayerPresentation(
            layer_id=LayerId('layer_vocals'),
            title='Vocals',
            subtitle='Stem main from separator',
            kind=LayerKind.AUDIO,
            is_selected=False,
            is_expanded=False,
            muted=False,
            soloed=False,
            color='#7dd3fc',
            waveform_key='vocals-main',
            badges=['main', 'audio'],
            status=LayerStatusPresentation(source_label='Derived from Song via stem_separation', sync_label='No sync'),
        ),
        LayerPresentation(
            layer_id=LayerId('layer_other'),
            title='Other',
            subtitle='Stem main from separator',
            kind=LayerKind.AUDIO,
            is_selected=False,
            is_expanded=False,
            muted=False,
            soloed=False,
            color='#94a3b8',
            waveform_key='other-main',
            badges=['main', 'audio'],
            status=LayerStatusPresentation(source_label='Derived from Song via stem_separation', sync_label='No sync'),
        ),
        LayerPresentation(
            layer_id=LayerId('layer_kick'),
            title='Kick',
            subtitle='Main classified events from Drums main',
            kind=LayerKind.EVENT,
            is_selected=False,
            is_expanded=True,
            muted=False,
            soloed=False,
            color='#66a3ff',
            events=kick_main,
            badges=['main', 'event'],
            status=LayerStatusPresentation(stale=True, manually_modified=True, source_label='Derived from Drums main', sync_label='Not synced'),
            takes=[TakeLanePresentation(take_id=TakeId('take_kick_rerun1'), name='Take 2 · classifier rerun', kind=LayerKind.EVENT, events=[_event('e9', 0.48, 0.76, 'Kick v2', '#66a3ff'), _event('e10', 4.18, 4.40, 'Kick v2', '#66a3ff')])],
        ),
        LayerPresentation(
            layer_id=LayerId('layer_snare'),
            title='Snare',
            subtitle='Main classified events from Drums main',
            kind=LayerKind.EVENT,
            is_selected=False,
            is_expanded=False,
            muted=False,
            soloed=False,
            color='#7fd1ae',
            events=snare_main,
            badges=['main', 'event'],
            status=LayerStatusPresentation(source_label='Derived from Drums main', sync_label='Not synced'),
        ),
        LayerPresentation(
            layer_id=LayerId('layer_hihat'),
            title='HiHat',
            subtitle='Main classified events from Drums main',
            kind=LayerKind.EVENT,
            is_selected=False,
            is_expanded=False,
            muted=False,
            soloed=False,
            color='#f8c555',
            events=hihat_main,
            badges=['main', 'event'],
            status=LayerStatusPresentation(source_label='Derived from Drums main', sync_label='Not synced'),
        ),
        LayerPresentation(
            layer_id=LayerId('layer_clap'),
            title='Clap',
            subtitle='Main classified events from Drums main',
            kind=LayerKind.EVENT,
            is_selected=False,
            is_expanded=True,
            muted=False,
            soloed=False,
            color='#ff8c78',
            events=[_event('e11', 2.50, 2.62, 'Clap', '#ff8c78')],
            badges=['main', 'event'],
            status=LayerStatusPresentation(source_label='Derived from Drums main', sync_label='Not synced'),
            takes=[TakeLanePresentation(take_id=TakeId('take_clap_rerun1'), name='Take 2 · threshold tweak', kind=LayerKind.EVENT, events=clap_take)],
        ),
        LayerPresentation(
            layer_id=LayerId('layer_sync'),
            title='MA3 Sync',
            subtitle='Main truth only',
            kind=LayerKind.SYNC,
            is_selected=False,
            is_expanded=False,
            muted=True,
            soloed=False,
            color='#ff8c78',
            events=[_event('e12', 0.75, 1.05, 'Cue 101', '#ff8c78'), _event('e13', 6.10, 6.35, 'Cue 102', '#ff8c78')],
            badges=['sync'],
            status=LayerStatusPresentation(source_label='Read/write main only', sync_label='Connected tc101'),
        ),
    ]

    return TimelinePresentation(
        timeline_id=TimelineId('timeline_demo'),
        title='Stage Zero Timeline',
        layers=layers,
        playhead=1.35,
        is_playing=True,
        follow_mode=FollowMode.PAGE,
        selected_layer_id=LayerId('layer_drums'),
        selected_take_id=None,
        selected_event_ids=[EventId('e3')],
        pixels_per_second=180.0,
        scroll_x=0.0,
        scroll_y=0.0,
        current_time_label=_fmt_time(1.35),
        end_time_label=_fmt_time(8.0),
    )


def build_demo_app() -> DemoTimelineApp:
    session = Session(
        id=SessionId('session_demo'),
        project_id=ProjectId('project_demo'),
        active_song_id=SongId('song_demo'),
        active_song_version_id=SongVersionId('song_version_demo'),
        active_timeline_id=TimelineId('timeline_demo'),
        transport_state=TransportState(is_playing=True, playhead=1.35, follow_mode=FollowMode.PAGE),
        mixer_state=MixerState(),
        playback_state=PlaybackState(status=PlaybackStatus.PLAYING, backend_name='demo'),
        sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref='show_manager'),
    )
    return DemoTimelineApp(presentation_state=build_demo_presentation(), session=session)


def main() -> int:
    app = QApplication(sys.argv)
    demo = build_demo_app()
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch)
    widget.resize(1440, 720)
    widget.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
