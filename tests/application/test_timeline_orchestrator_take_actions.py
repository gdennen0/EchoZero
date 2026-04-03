from __future__ import annotations

from dataclasses import replace

from echozero.application.mixer.models import AudibilityState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
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
from echozero.application.timeline.intents import TriggerTakeAction
from echozero.application.timeline.models import Event, Layer, Take, Timeline
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService


class _SessionService(SessionService):
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


class _TransportService(TransportService):
    def __init__(self, state: TransportState):
        self._state = state

    def get_state(self) -> TransportState:
        return self._state

    def play(self) -> TransportState:
        self._state.is_playing = True
        return self._state

    def pause(self) -> TransportState:
        self._state.is_playing = False
        return self._state

    def stop(self) -> TransportState:
        self._state.is_playing = False
        self._state.playhead = 0.0
        return self._state

    def seek(self, position: float) -> TransportState:
        self._state.playhead = max(0.0, position)
        return self._state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._state.loop_region = loop_region
        self._state.loop_enabled = enabled
        return self._state


class _MixerService(MixerService):
    def __init__(self):
        self._state = MixerState()

    def get_state(self) -> MixerState:
        return self._state

    def set_layer_state(self, layer_id, state):
        self._state.layer_states[layer_id] = state
        return self._state

    def set_mute(self, layer_id, muted: bool):
        return self._state

    def set_solo(self, layer_id, soloed: bool):
        return self._state

    def set_gain(self, layer_id, gain_db: float):
        return self._state

    def set_pan(self, layer_id, pan: float):
        return self._state

    def resolve_audibility(self, layers: list[Layer]) -> list[AudibilityState]:
        return [AudibilityState(layer_id=layer.id, is_audible=True, reason="default") for layer in layers]


class _PlaybackService(PlaybackService):
    def __init__(self):
        self._state = PlaybackState()

    def get_state(self) -> PlaybackState:
        return self._state

    def prepare(self, timeline: Timeline) -> PlaybackState:
        return self._state

    def update_runtime(self, timeline, transport, audibility, sync) -> PlaybackState:
        return self._state

    def stop(self) -> PlaybackState:
        return self._state


class _SyncService(SyncService):
    def __init__(self):
        self._state = SyncState()

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode):
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        self._state.connected = True
        return self._state

    def disconnect(self) -> SyncState:
        self._state.connected = False
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _event(event_id: str, take_id: str, start: float) -> Event:
    return Event(
        id=EventId(event_id),
        take_id=TakeId(take_id),
        start=start,
        end=start + 0.2,
        label=event_id,
    )


def _build_orchestrator_and_timeline() -> tuple[TimelineOrchestrator, Timeline, Layer, Take, Take]:
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_kick"),
        name="Main",
        events=[_event("main_1", "take_main", 1.0), _event("main_2", "take_main", 2.0)],
    )
    alt_take = Take(
        id=TakeId("take_alt"),
        layer_id=LayerId("layer_kick"),
        name="Take 2",
        events=[_event("alt_1", "take_alt", 1.25), _event("alt_2", "take_alt", 2.25)],
    )
    layer = Layer(
        id=LayerId("layer_kick"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take, alt_take],
        active_take_id=TakeId("take_main"),
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=TimelineId("timeline_1"),
    )

    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(session.transport_state),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=_SyncService(),
        assembler=_Assembler(),
    )
    return orchestrator, timeline, layer, main_take, alt_take


def test_trigger_take_action_overwrite_main_replaces_events_from_source_take():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="overwrite_main"),
    )

    assert len(main_take.events) == len(alt_take.events)
    assert all(event.take_id == main_take.id for event in main_take.events)
    assert all(str(event.id).startswith("take_main:from:") for event in main_take.events)
    assert timeline.selection.selected_take_id == main_take.id


def test_trigger_take_action_merge_main_appends_sorted_events():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    before_count = len(main_take.events)
    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="merge_main"),
    )

    assert len(main_take.events) == before_count + len(alt_take.events)
    starts = [event.start for event in main_take.events]
    assert starts == sorted(starts)
    assert timeline.selection.selected_layer_id == layer.id


def test_trigger_take_action_unknown_action_is_noop():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    original = list(main_take.events)
    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="future_action"),
    )

    assert [event.id for event in main_take.events] == [event.id for event in original]
