from __future__ import annotations

from echozero.application.mixer.models import AudibilityState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.timeline.assembler import TimelineAssembler
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
from echozero.application.timeline.intents import (
    ClearSelection,
    SelectEvent,
    SelectAllEvents,
    SelectTake,
    Stop,
    ToggleLayerExpanded,
    ToggleMute,
    ToggleSolo,
    TriggerTakeAction,
)
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


def test_select_take_is_selection_only_and_does_not_change_main_truth():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    original_main_event_ids = [event.id for event in main_take.events]
    orchestrator.handle(
        timeline,
        SelectTake(layer_id=layer.id, take_id=alt_take.id),
    )

    assert timeline.selection.selected_take_id == alt_take.id
    assert [event.id for event in main_take.events] == original_main_event_ids


def test_toggle_layer_expanded_round_trips_through_assembled_presentation():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    orchestrator.assembler = TimelineAssembler()

    expanded = orchestrator.handle(
        timeline,
        ToggleLayerExpanded(layer_id=layer.id),
    )

    assert timeline.layers[0].presentation_hints.expanded is True
    assert expanded.layers[0].is_expanded is True

    collapsed = orchestrator.handle(
        timeline,
        ToggleLayerExpanded(layer_id=layer.id),
    )

    assert timeline.layers[0].presentation_hints.expanded is False
    assert collapsed.layers[0].is_expanded is False


def test_select_event_updates_selected_take_for_main_and_take_events():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=main_take.id, event_id=main_take.events[0].id),
    )
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [main_take.events[0].id]

    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id),
    )
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_event_additive_and_toggle_preserve_deterministic_take_context():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=main_take.id, event_id=main_take.events[0].id, mode="replace"),
    )
    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id, mode="additive"),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id, alt_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id

    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id, mode="toggle"),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id

    orchestrator.handle(
        timeline,
        SelectEvent(layer_id=layer.id, take_id=alt_take.id, event_id=main_take.events[0].id, mode="toggle"),
    )

    assert timeline.selection.selected_event_ids == []
    assert timeline.selection.selected_take_id is None


def test_clear_selection_clears_events_and_take_without_dropping_selected_layer():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [EventId("alt_1")]

    orchestrator.handle(timeline, ClearSelection())

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id is None
    assert timeline.selection.selected_event_ids == []


def test_select_all_events_uses_selected_layer_when_present_and_skips_locked_layers():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    layer.presentation_hints.visible = True
    layer.presentation_hints.locked = False
    other_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    other_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    other_layer.presentation_hints.locked = True
    timeline.layers.append(other_layer)
    timeline.selection.selected_layer_id = layer.id

    orchestrator.handle(timeline, SelectAllEvents())

    assert timeline.selection.selected_event_ids == [
        main_take.events[0].id,
        main_take.events[1].id,
        alt_take.events[0].id,
        alt_take.events[1].id,
    ]
    assert timeline.selection.selected_take_id == main_take.id


def test_select_all_events_without_selected_layer_uses_visible_unlocked_layers_only():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    layer.presentation_hints.visible = True
    layer.presentation_hints.locked = False

    visible_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    visible_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[visible_take],
    )
    hidden_take = Take(
        id=TakeId("take_hat_main"),
        layer_id=LayerId("layer_hat"),
        name="Main",
        events=[_event("hat_1", "take_hat_main", 4.0)],
    )
    hidden_layer = Layer(
        id=LayerId("layer_hat"),
        timeline_id=timeline.id,
        name="Hat",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[hidden_take],
    )
    hidden_layer.presentation_hints.visible = False
    timeline.layers.extend([visible_layer, hidden_layer])
    timeline.selection.selected_layer_id = None

    orchestrator.handle(timeline, SelectAllEvents())

    assert timeline.selection.selected_event_ids == [
        main_take.events[0].id,
        main_take.events[1].id,
        alt_take.events[0].id,
        alt_take.events[1].id,
        visible_take.events[0].id,
    ]
    assert timeline.selection.selected_take_id == main_take.id


def test_stop_resets_transport_playhead_and_playing_state():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    session = orchestrator.session_service.get_session()
    session.transport_state.is_playing = True
    session.transport_state.playhead = 3.5

    orchestrator.handle(timeline, Stop())

    assert session.transport_state.is_playing is False
    assert session.transport_state.playhead == 0.0


def test_toggle_mute_and_solo_update_layer_mixer_state():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(timeline, ToggleMute(layer.id))
    assert layer.mixer.mute is True

    orchestrator.handle(timeline, ToggleSolo(layer.id))
    assert layer.mixer.solo is True


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
