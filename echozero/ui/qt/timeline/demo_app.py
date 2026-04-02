"""Runnable demo/testing loop for the new Stage Zero timeline shell."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from PyQt6.QtWidgets import QApplication

from echozero.application.mixer.models import AudibilityState, MixerState, LayerMixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState, LayerPlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.project.models import Project
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import (
    FollowMode,
    LayerKind,
    PlaybackMode,
    PlaybackStatus,
    SyncMode,
)
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    ProjectId,
    SessionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.intents import TimelineIntent
from echozero.application.timeline.models import (
    Event,
    Layer,
    LayerPresentationHints,
    LayerSyncState,
    Take,
    Timeline,
    TimelineSelection,
    TimelineViewport,
)
from echozero.application.timeline.orchestrator import TimelineOrchestrator
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

    def resolve_audibility(self, layers: list[Layer]) -> list[AudibilityState]:
        any_solo = any(layer.mixer.solo for layer in layers)
        resolved: list[AudibilityState] = []
        for layer in layers:
            is_audible = not layer.mixer.mute and (layer.mixer.solo if any_solo else True)
            reason = "solo" if any_solo and layer.mixer.solo else ("muted" if layer.mixer.mute else "normal")
            resolved.append(AudibilityState(layer_id=layer.id, is_audible=is_audible, reason=reason))
        return resolved


class DemoPlaybackService(PlaybackService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> PlaybackState:
        return self._session.playback_state

    def prepare(self, timeline: Timeline) -> PlaybackState:
        return self._session.playback_state

    def update_runtime(self, timeline: Timeline, transport: TransportState, audibility: list[AudibilityState], sync: SyncState) -> PlaybackState:
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
    timeline: Timeline
    session: Session
    orchestrator: TimelineOrchestrator = field(init=False)

    def __post_init__(self) -> None:
        self.orchestrator = TimelineOrchestrator(
            session_service=DemoSessionService(self.session),
            transport_service=DemoTransportService(self.session),
            mixer_service=DemoMixerService(self.session),
            playback_service=DemoPlaybackService(self.session),
            sync_service=DemoSyncService(self.session),
            assembler=TimelineAssembler(),
        )

    def presentation(self) -> TimelinePresentation:
        return self.orchestrator.assembler.assemble(self.timeline, self.session)

    def dispatch(self, intent: TimelineIntent) -> TimelinePresentation:
        return self.orchestrator.handle(self.timeline, intent)


def build_demo_timeline():
    timeline_id = TimelineId("timeline_demo")
    layer_a = Layer(
        id=LayerId("layer_drums"),
        timeline_id=timeline_id,
        name="Drums",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[
            Take(
                id=TakeId("take_drums_main"),
                layer_id=LayerId("layer_drums"),
                name="Main",
                events=[
                    Event(id=EventId("e1"), take_id=TakeId("take_drums_main"), start=0.5, end=0.8, label="Kick", color="#66a3ff"),
                    Event(id=EventId("e2"), take_id=TakeId("take_drums_main"), start=1.2, end=1.45, label="Snare", color="#7fd1ae"),
                    Event(id=EventId("e3"), take_id=TakeId("take_drums_main"), start=2.0, end=2.35, label="Crash", color="#f8c555"),
                ],
            ),
            Take(
                id=TakeId("take_drums_alt"),
                layer_id=LayerId("layer_drums"),
                name="Alt",
                events=[
                    Event(id=EventId("e3a"), take_id=TakeId("take_drums_alt"), start=0.45, end=0.78, label="Kick Alt", color="#66a3ff"),
                    Event(id=EventId("e3b"), take_id=TakeId("take_drums_alt"), start=1.22, end=1.52, label="Snare Alt", color="#7fd1ae"),
                ],
            ),
        ],
        active_take_id=TakeId("take_drums_main"),
        mixer=LayerMixerState(gain_db=-1.5),
        playback=LayerPlaybackState(mode=PlaybackMode.EVENT_SLICE, enabled=True),
        presentation_hints=LayerPresentationHints(color="#66a3ff", take_selector_expanded=False),
    )

    layer_b = Layer(
        id=LayerId("layer_bass"),
        timeline_id=timeline_id,
        name="Bass",
        kind=LayerKind.AUDIO,
        order_index=1,
        takes=[
            Take(
                id=TakeId("take_bass_1"),
                layer_id=LayerId("layer_bass"),
                name="Stem A",
                events=[
                    Event(id=EventId("e4"), take_id=TakeId("take_bass_1"), start=0.0, end=3.5, label="Bass Stem", color="#d68cff"),
                ],
                source_ref="bass_stem.wav",
            )
        ],
        active_take_id=TakeId("take_bass_1"),
        mixer=LayerMixerState(solo=True, gain_db=-3.0),
        playback=LayerPlaybackState(mode=PlaybackMode.CONTINUOUS_AUDIO, enabled=True),
        presentation_hints=LayerPresentationHints(color="#d68cff"),
    )

    layer_c = Layer(
        id=LayerId("layer_sync"),
        timeline_id=timeline_id,
        name="MA3 Sync",
        kind=LayerKind.SYNC,
        order_index=2,
        takes=[
            Take(
                id=TakeId("take_sync_1"),
                layer_id=LayerId("layer_sync"),
                name="Main",
                events=[
                    Event(id=EventId("e5"), take_id=TakeId("take_sync_1"), start=0.75, end=1.05, label="Cue 101", color="#ff8c78"),
                    Event(id=EventId("e6"), take_id=TakeId("take_sync_1"), start=2.1, end=2.4, label="Cue 102", color="#ff8c78"),
                ],
            )
        ],
        active_take_id=TakeId("take_sync_1"),
        mixer=LayerMixerState(mute=True),
        playback=LayerPlaybackState(mode=PlaybackMode.NONE, enabled=False),
        sync=LayerSyncState(mode=SyncMode.MA3.value, connected=True, target_ref="show_manager", ma3_track_coord="tc101"),
        presentation_hints=LayerPresentationHints(color="#ff8c78", locked=True),
    )

    timeline = Timeline(
        id=timeline_id,
        song_version_id=SongVersionId("song_version_demo"),
        start=0.0,
        end=8.0,
        layers=[layer_a, layer_b, layer_c],
        selection=TimelineSelection(
            selected_layer_id=LayerId("layer_drums"),
            selected_take_id=TakeId("take_drums_main"),
            selected_event_ids=[EventId("e2")],
        ),
        viewport=TimelineViewport(pixels_per_second=180.0, scroll_x=0.0, scroll_y=0.0),
    )

    session = Session(
        id=SessionId("session_demo"),
        project_id=ProjectId("project_demo"),
        active_song_id=SongId("song_demo"),
        active_song_version_id=SongVersionId("song_version_demo"),
        active_timeline_id=timeline_id,
        transport_state=TransportState(is_playing=True, playhead=1.35, follow_mode=FollowMode.PAGE),
        mixer_state=MixerState(),
        playback_state=PlaybackState(status=PlaybackStatus.PLAYING, backend_name="demo"),
        sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref="show_manager"),
    )

    return timeline, session


def build_demo_app() -> DemoTimelineApp:
    timeline, session = build_demo_timeline()
    return DemoTimelineApp(timeline=timeline, session=session)


def main() -> int:
    app = QApplication(sys.argv)
    demo = build_demo_app()
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch)
    widget.resize(1400, 420)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
