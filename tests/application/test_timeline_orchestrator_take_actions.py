from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import FollowMode, LayerKind
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    ProjectId,
    RegionId,
    SessionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.shared.ranges import TimeRange
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.event_batch_scope import EventBatchScope
from echozero.application.timeline.intents import (
    ClearSelection,
    CreateRegion,
    CreateEvent,
    DeleteRegion,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveSelectedEventsToAdjacentLayer,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    RenumberEventCueNumbers,
    SelectAllEvents,
    SelectAdjacentEventInSelectedLayer,
    SelectAdjacentLayer,
    SelectEveryOtherEvents,
    SelectEvent,
    SelectRegion,
    SelectTake,
    SetActivePlaybackTarget,
    SetFollowCursorEnabled,
    SetGain,
    SetLayerOutputBus,
    SetSelectedEvents,
    Stop,
    ToggleLayerExpanded,
    TriggerTakeAction,
    UpdateRegion,
)
from echozero.application.timeline.models import Event, EventRef, Layer, Take, Timeline
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
        return [
            AudibilityState(layer_id=layer.id, is_audible=True, reason="default")
            for layer in layers
        ]


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


def _event(
    event_id: str,
    take_id: str,
    start: float,
    *,
    metadata: dict[str, object] | None = None,
) -> Event:
    return Event(
        id=EventId(event_id),
        take_id=TakeId(take_id),
        start=start,
        end=start + 0.2,
        label=event_id,
        metadata=dict(metadata or {}),
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
    assert timeline.playback_target.layer_id is None
    assert timeline.playback_target.take_id is None
    assert [event.id for event in main_take.events] == original_main_event_ids


def test_set_active_playback_target_updates_playback_target_without_touching_selection():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()

    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id

    orchestrator.handle(
        timeline,
        SetActivePlaybackTarget(layer_id=layer.id, take_id=None),
    )

    assert timeline.playback_target.layer_id == layer.id
    assert timeline.playback_target.take_id is None
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id


def test_set_active_playback_target_uses_explicit_playback_target_path():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        SetActivePlaybackTarget(layer_id=layer.id, take_id=alt_take.id),
    )

    assert timeline.playback_target.layer_id == layer.id
    assert timeline.playback_target.take_id == alt_take.id
    assert timeline.selection.selected_layer_id is None
    assert timeline.selection.selected_take_id is None


def test_set_active_playback_target_can_clear_target_without_clearing_selection():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()

    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.playback_target.layer_id = layer.id
    timeline.playback_target.take_id = alt_take.id

    orchestrator.handle(
        timeline,
        SetActivePlaybackTarget(layer_id=None, take_id=None),
    )

    assert timeline.playback_target.layer_id is None
    assert timeline.playback_target.take_id is None
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id


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
        SelectEvent(
            layer_id=layer.id,
            take_id=main_take.id,
            event_id=main_take.events[0].id,
            mode="replace",
        ),
    )
    orchestrator.handle(
        timeline,
        SelectEvent(
            layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id, mode="additive"
        ),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id, alt_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id

    orchestrator.handle(
        timeline,
        SelectEvent(
            layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id, mode="toggle"
        ),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id

    orchestrator.handle(
        timeline,
        SelectEvent(
            layer_id=layer.id, take_id=alt_take.id, event_id=main_take.events[0].id, mode="toggle"
        ),
    )

    assert timeline.selection.selected_event_ids == [main_take.events[0].id]
    assert timeline.selection.selected_take_id == alt_take.id


def test_select_adjacent_layer_moves_selection_between_visible_layers():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    snare_take = Take(
        id=TakeId("take_snare"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare", 3.0)],
    )
    hat_take = Take(
        id=TakeId("take_hat"),
        layer_id=LayerId("layer_hat"),
        name="Main",
        events=[_event("hat_1", "take_hat", 4.0)],
    )
    snare_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[snare_take],
    )
    hat_layer = Layer(
        id=LayerId("layer_hat"),
        timeline_id=timeline.id,
        name="Hat",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[hat_take],
    )
    timeline.layers.extend([snare_layer, hat_layer])
    timeline.selection.selected_layer_id = snare_layer.id
    timeline.selection.selected_layer_ids = [snare_layer.id]
    timeline.selection.selected_take_id = snare_take.id
    timeline.selection.selected_event_ids = [snare_take.events[0].id]

    orchestrator.handle(timeline, SelectAdjacentLayer(direction=-1))

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_layer_ids == [layer.id]
    assert timeline.selection.selected_take_id is None
    assert timeline.selection.selected_event_ids == []

    orchestrator.handle(timeline, SelectAdjacentLayer(direction=1))

    assert timeline.selection.selected_layer_id == snare_layer.id
    assert timeline.selection.selected_layer_ids == [snare_layer.id]


def test_select_adjacent_event_in_selected_layer_uses_current_take_context():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=1))

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]

    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=-1))

    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_adjacent_event_without_selection_uses_playhead_position():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = []
    timeline.selection.selected_event_refs = []

    orchestrator.transport_service.seek(1.5)
    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=1))

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]

    timeline.selection.selected_event_ids = []
    timeline.selection.selected_event_refs = []
    orchestrator.transport_service.seek(2.1)
    orchestrator.handle(timeline, SelectAdjacentEventInSelectedLayer(direction=-1))

    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_adjacent_event_skips_demoted_when_demoted_navigation_disabled():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    alt_take.events = [
        _event("alt_1", "take_alt", 1.25),
        _event(
            "alt_2",
            "take_alt",
            2.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event("alt_3", "take_alt", 3.25),
    ]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=False),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[2].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=-1, include_demoted=False),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[0].id]


def test_select_adjacent_event_can_include_demoted_when_enabled():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    alt_take.events = [
        _event("alt_1", "take_alt", 1.25),
        _event(
            "alt_2",
            "take_alt",
            2.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event("alt_3", "take_alt", 3.25),
    ]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=True),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=True),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[2].id]


def test_select_adjacent_event_anchors_to_most_recent_selected_ref_order() -> None:
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    alt_take.events = [
        _event(
            "alt_1",
            "take_alt",
            1.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event(
            "alt_2",
            "take_alt",
            2.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
        _event(
            "alt_3",
            "take_alt",
            3.25,
            metadata={"review": {"promotion_state": "demoted"}},
        ),
    ]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_refs = [
        EventRef(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[1].id),
        EventRef(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id),
    ]
    timeline.selection.selected_event_ids = [
        alt_take.events[1].id,
        alt_take.events[0].id,
    ]

    orchestrator.handle(
        timeline,
        SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=True),
    )

    assert timeline.selection.selected_event_ids == [alt_take.events[1].id]


def test_clear_selection_clears_events_and_take_without_dropping_selected_layer():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [EventId("alt_1")]

    orchestrator.handle(timeline, ClearSelection())

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id is None
    assert timeline.selection.selected_event_ids == []


def test_region_intents_create_update_delete_and_select_round_trip():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        CreateRegion(
            time_range=TimeRange(start=1.0, end=2.0),
            label="Verse",
            color="#aabbcc",
            kind="song",
        ),
    )
    first_region_id = timeline.selection.selected_region_id
    assert first_region_id == RegionId("region_1")
    assert [region.order_index for region in timeline.regions] == [0]
    assert timeline.end == pytest.approx(2.0)

    orchestrator.handle(
        timeline,
        CreateRegion(
            time_range=TimeRange(start=0.0, end=0.5),
            label="Intro",
        ),
    )
    second_region_id = timeline.selection.selected_region_id
    assert second_region_id == RegionId("region_2")
    assert [region.id for region in timeline.regions] == [RegionId("region_2"), RegionId("region_1")]
    assert [region.order_index for region in timeline.regions] == [0, 1]

    orchestrator.handle(
        timeline,
        UpdateRegion(
            region_id=first_region_id,
            time_range=TimeRange(start=2.5, end=3.0),
            label="Chorus",
            color="#ddeeff",
            kind="structure",
        ),
    )
    assert timeline.selection.selected_region_id == first_region_id
    updated = next(region for region in timeline.regions if region.id == first_region_id)
    assert updated.start == pytest.approx(2.5)
    assert updated.end == pytest.approx(3.0)
    assert updated.label == "Chorus"
    assert updated.color == "#ddeeff"
    assert updated.kind == "structure"
    assert timeline.end == pytest.approx(3.0)

    orchestrator.handle(timeline, SelectRegion(region_id=second_region_id))
    assert timeline.selection.selected_region_id == second_region_id

    orchestrator.handle(timeline, DeleteRegion(region_id=second_region_id))
    assert [region.id for region in timeline.regions] == [first_region_id]
    assert timeline.selection.selected_region_id is None

    orchestrator.handle(timeline, SelectRegion(region_id=first_region_id))
    assert timeline.selection.selected_region_id == first_region_id
    orchestrator.handle(timeline, DeleteRegion(region_id=first_region_id))
    assert timeline.regions == []
    assert timeline.selection.selected_region_id is None


def test_clear_selection_clears_selected_region():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    orchestrator.handle(
        timeline,
        CreateRegion(time_range=TimeRange(start=0.5, end=1.5), label="Focus"),
    )
    assert timeline.selection.selected_region_id == RegionId("region_1")
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]

    orchestrator.handle(timeline, ClearSelection())

    assert timeline.selection.selected_region_id is None
    assert timeline.selection.selected_layer_id == layer.id


def test_select_every_other_events_region_scope_uses_visible_unlocked_main_events():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    snare_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[
            _event("snare_1", "take_snare_main", 1.1),
            _event("snare_2", "take_snare_main", 1.6),
        ],
    )
    snare_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[snare_take],
    )
    locked_take = Take(
        id=TakeId("take_locked_main"),
        layer_id=LayerId("layer_locked"),
        name="Main",
        events=[_event("locked_1", "take_locked_main", 1.2)],
    )
    locked_layer = Layer(
        id=LayerId("layer_locked"),
        timeline_id=timeline.id,
        name="Locked",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[locked_take],
    )
    locked_layer.presentation_hints.locked = True
    timeline.layers.extend([snare_layer, locked_layer])

    orchestrator.handle(
        timeline,
        CreateRegion(time_range=TimeRange(start=1.0, end=2.5), label="Batch"),
    )
    region_id = timeline.selection.selected_region_id
    assert region_id is not None

    orchestrator.handle(
        timeline,
        SelectEveryOtherEvents(scope=EventBatchScope(mode="region", region_id=region_id)),
    )

    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("snare_1")]
    assert EventId("main_2") not in timeline.selection.selected_event_ids
    assert alt_take.events[0].id not in timeline.selection.selected_event_ids
    assert EventId("locked_1") not in timeline.selection.selected_event_ids
    assert timeline.selection.selected_layer_ids == [layer.id, snare_layer.id]
    assert timeline.selection.selected_layer_id == snare_layer.id
    assert timeline.selection.selected_take_id == snare_take.id


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


def test_set_selected_events_preserves_cross_layer_batch_selection_context():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
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
    timeline.layers.append(other_layer)

    orchestrator.handle(
        timeline,
        SetSelectedEvents(
            event_ids=[main_take.events[0].id, other_take.events[0].id],
            anchor_layer_id=other_layer.id,
            anchor_take_id=other_take.id,
            selected_layer_ids=[layer.id, other_layer.id],
        ),
    )

    assert timeline.selection.selected_layer_id == other_layer.id
    assert timeline.selection.selected_layer_ids == [layer.id, other_layer.id]
    assert timeline.selection.selected_take_id == other_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("snare_1")]


def test_select_every_other_events_uses_current_selected_event_scope():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        SetSelectedEvents(
            event_ids=[main_take.events[0].id, alt_take.events[0].id, alt_take.events[1].id],
            event_refs=[],
            anchor_layer_id=layer.id,
            anchor_take_id=alt_take.id,
            selected_layer_ids=[layer.id],
        ),
    )

    orchestrator.handle(
        timeline,
        SelectEveryOtherEvents(scope=EventBatchScope(mode="selected_events")),
    )

    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_layer_ids == [layer.id]
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("alt_2")]


def test_select_every_other_events_restarts_per_selected_layer_main_scope():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    other_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[
            _event("snare_1", "take_snare_main", 3.0),
            _event("snare_2", "take_snare_main", 4.0),
        ],
    )
    other_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    timeline.layers.append(other_layer)
    timeline.selection.selected_layer_id = other_layer.id
    timeline.selection.selected_layer_ids = [layer.id, other_layer.id]
    timeline.selection.selected_take_id = other_take.id
    timeline.selection.selected_event_ids = []

    orchestrator.handle(
        timeline,
        SelectEveryOtherEvents(scope=EventBatchScope(mode="selected_layers_main")),
    )

    assert [event.id for event in alt_take.events] == [EventId("alt_1"), EventId("alt_2")]
    assert timeline.selection.selected_layer_id == other_layer.id
    assert timeline.selection.selected_layer_ids == [layer.id, other_layer.id]
    assert timeline.selection.selected_take_id == other_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("snare_1")]


def test_renumber_event_cue_numbers_restarts_per_selected_layer_main_scope():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    other_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[
            _event("snare_1", "take_snare_main", 3.0),
            _event("snare_2", "take_snare_main", 4.0),
        ],
    )
    other_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    timeline.layers.append(other_layer)
    timeline.selection.selected_layer_id = other_layer.id
    timeline.selection.selected_layer_ids = [layer.id, other_layer.id]
    timeline.selection.selected_take_id = other_take.id

    orchestrator.handle(
        timeline,
        RenumberEventCueNumbers(
            scope=EventBatchScope(mode="selected_layers_main"),
            start_at=1,
            step=1,
        ),
    )

    assert [event.cue_number for event in main_take.events] == [1, 2]
    assert [event.cue_number for event in other_take.events] == [1, 2]
    assert [event.cue_number for event in alt_take.events] == [1, 1]
    assert timeline.selection.selected_layer_id == other_layer.id
    assert timeline.selection.selected_layer_ids == [layer.id, other_layer.id]
    assert timeline.selection.selected_take_id == other_take.id
    assert timeline.selection.selected_event_ids == [
        EventId("main_1"),
        EventId("main_2"),
        EventId("snare_1"),
        EventId("snare_2"),
    ]


def test_create_event_appends_sorted_event_and_selects_it():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        CreateEvent(
            layer_id=layer.id,
            take_id=main_take.id,
            time_range=TimeRange(start=1.4, end=1.7),
            label="Inserted",
        ),
    )

    inserted = next(event for event in main_take.events if event.label == "Inserted")
    assert inserted.id == EventId("take_main:event:1")
    assert inserted.cue_number == 1
    assert [event.start for event in main_take.events] == [1.0, 1.4, 2.0]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [inserted.id]


def test_create_event_accepts_float_cue_numbers():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(
        timeline,
        CreateEvent(
            layer_id=layer.id,
            take_id=main_take.id,
            time_range=TimeRange(start=1.4, end=1.7),
            label="Inserted",
            cue_number=1.5,
        ),
    )

    inserted = next(event for event in main_take.events if event.label == "Inserted")
    assert inserted.cue_number == 1.5


def test_create_event_on_section_layer_creates_section_start_on_main_take():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    section_main_take = Take(
        id=TakeId("take_sections_main"),
        layer_id=LayerId("layer_sections"),
        name="Main",
        events=[
            Event(
                id=EventId("section_1"),
                take_id=TakeId("take_sections_main"),
                start=0.5,
                end=0.58,
                cue_number=7.5,
                label="Verse",
                cue_ref="Q7.5",
            )
        ],
    )
    section_alt_take = Take(
        id=TakeId("take_sections_alt"),
        layer_id=LayerId("layer_sections"),
        name="Take 2",
        events=[],
    )
    section_layer = Layer(
        id=LayerId("layer_sections"),
        timeline_id=timeline.id,
        name="Sections",
        kind=LayerKind.SECTION,
        order_index=1,
        takes=[section_main_take, section_alt_take],
    )
    timeline.layers.append(section_layer)

    orchestrator.handle(
        timeline,
        CreateEvent(
            layer_id=section_layer.id,
            take_id=section_alt_take.id,
            time_range=TimeRange(start=1.6, end=2.2),
        ),
    )

    created = next(
        event for event in section_main_take.events if event.id == EventId("take_sections_main:event:1")
    )
    assert created.start == pytest.approx(1.6)
    assert created.end == pytest.approx(1.68)
    assert created.cue_number == 8
    assert created.cue_ref == "Q8"
    assert created.label == "Section 8"
    assert section_alt_take.events == []
    assert timeline.selection.selected_layer_id == section_layer.id
    assert timeline.selection.selected_layer_ids == [section_layer.id]
    assert timeline.selection.selected_take_id == section_main_take.id
    assert timeline.selection.selected_event_ids == [created.id]


def test_delete_events_removes_records_and_clears_selected_take_when_selection_is_empty():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id, alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        DeleteEvents(event_ids=[main_take.events[0].id, alt_take.events[0].id]),
    )

    assert [event.id for event in main_take.events] == [EventId("main_2")]
    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id is None
    assert timeline.selection.selected_event_ids == []


def test_stop_resets_transport_playhead_and_playing_state():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    session = orchestrator.session_service.get_session()
    session.transport_state.is_playing = True
    session.transport_state.playhead = 3.5

    orchestrator.handle(timeline, Stop())

    assert session.transport_state.is_playing is False
    assert session.transport_state.playhead == 0.0


def test_set_gain_updates_layer_mixer_state():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(timeline, SetGain(layer.id, -6.0))
    assert layer.mixer.gain_db == -6.0


def test_set_layer_output_bus_updates_layer_and_mixer_session_state():
    orchestrator, timeline, layer, _main_take, _alt_take = _build_orchestrator_and_timeline()

    orchestrator.handle(timeline, SetLayerOutputBus(layer.id, "outputs_3_4"))

    assert layer.mixer.output_bus == "outputs_3_4"
    mixer_state = orchestrator.mixer_service.get_state()
    assert mixer_state.layer_states[layer.id].output_bus == "outputs_3_4"

    orchestrator.handle(timeline, SetLayerOutputBus(layer.id, None))

    assert layer.mixer.output_bus is None
    assert mixer_state.layer_states[layer.id].output_bus is None


def test_set_follow_cursor_enabled_updates_transport_follow_mode():
    orchestrator, timeline, _layer, _main_take, _alt_take = _build_orchestrator_and_timeline()
    session = orchestrator.session_service.get_session()
    session.transport_state.follow_mode = FollowMode.CENTER

    orchestrator.handle(timeline, SetFollowCursorEnabled(enabled=False))
    assert session.transport_state.follow_mode == FollowMode.OFF

    orchestrator.handle(timeline, SetFollowCursorEnabled(enabled=True))
    assert session.transport_state.follow_mode == FollowMode.CENTER


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


def test_trigger_take_action_add_selection_to_main_only_clones_selected_take_events():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    selected_event = alt_take.events[1]
    orchestrator.handle(
        timeline,
        SetSelectedEvents(
            event_ids=[selected_event.id],
            event_refs=[],
            anchor_layer_id=layer.id,
            anchor_take_id=alt_take.id,
            selected_layer_ids=[layer.id],
        ),
    )

    before_count = len(main_take.events)
    orchestrator.handle(
        timeline,
        TriggerTakeAction(
            layer_id=layer.id,
            take_id=alt_take.id,
            action_id="add_selection_to_main",
        ),
    )

    assert len(main_take.events) == before_count + 1
    cloned_event = next(event for event in main_take.events if event.start == selected_event.start)
    assert cloned_event.take_id == main_take.id
    assert cloned_event.parent_event_id == str(selected_event.id)
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [cloned_event.id]


def test_trigger_take_action_delete_take_removes_non_main_take_and_falls_back_to_main():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]
    timeline.playback_target.layer_id = layer.id
    timeline.playback_target.take_id = alt_take.id

    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="delete_take"),
    )

    assert [take.id for take in layer.takes] == [main_take.id]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == []
    assert timeline.playback_target.take_id == main_take.id


def test_trigger_take_action_unknown_action_is_noop():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()

    original = list(main_take.events)
    orchestrator.handle(
        timeline,
        TriggerTakeAction(layer_id=layer.id, take_id=alt_take.id, action_id="future_action"),
    )

    assert [event.id for event in main_take.events] == [event.id for event in original]


def test_move_selected_events_shifts_selected_events_and_preserves_deterministic_take_context():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id, alt_take.events[0].id]

    orchestrator.handle(timeline, MoveSelectedEvents(delta_seconds=0.5))

    assert main_take.events[0].start == 1.5
    assert main_take.events[0].end == 1.7
    assert alt_take.events[0].start == 1.75
    assert alt_take.events[0].end == 1.95
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert timeline.selection.selected_event_ids == [EventId("main_1"), EventId("alt_1")]


def test_move_selected_events_clamps_at_time_zero():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id]

    orchestrator.handle(timeline, MoveSelectedEvents(delta_seconds=-5.0))

    assert main_take.events[0].start == 0.0
    assert main_take.events[0].end == pytest.approx(0.2)
    assert timeline.selection.selected_take_id == main_take.id


def test_move_selected_events_transfers_to_target_main_take():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    target_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    target_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[target_take],
    )
    timeline.layers.append(target_layer)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.25, target_layer_id=target_layer.id),
    )

    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert [event.id for event in target_take.events] == [EventId("alt_1"), EventId("snare_1")]
    assert target_take.events[0].take_id == target_take.id
    assert target_take.events[0].start == 1.5
    assert target_take.events[0].end == 1.7
    assert timeline.selection.selected_layer_id == target_layer.id
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [EventId("alt_1")]


def test_move_selected_events_with_copy_selected_duplicates_in_place():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [main_take.events[0].id]

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.5, copy_selected=True),
    )

    assert [event.id for event in main_take.events] == [
        EventId("main_1"),
        EventId("take_main:dup:main_1:1"),
        EventId("main_2"),
    ]
    duplicate = next(
        event for event in main_take.events if event.id == EventId("take_main:dup:main_1:1")
    )
    original = next(event for event in main_take.events if event.id == EventId("main_1"))
    assert original.start == 1.0
    assert original.end == 1.2
    assert duplicate.start == 1.5
    assert duplicate.end == 1.7
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [EventId("take_main:dup:main_1:1")]


def test_move_selected_events_with_copy_selected_to_target_layer_keeps_originals():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    target_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[_event("snare_1", "take_snare_main", 3.0)],
    )
    target_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[target_take],
    )
    timeline.layers.append(target_layer)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(
        timeline,
        MoveSelectedEvents(
            delta_seconds=0.25,
            target_layer_id=target_layer.id,
            copy_selected=True,
        ),
    )

    assert [event.id for event in alt_take.events] == [EventId("alt_1"), EventId("alt_2")]
    assert [event.id for event in target_take.events] == [
        EventId("take_snare_main:dup:alt_1:1"),
        EventId("snare_1"),
    ]
    assert target_take.events[0].start == 1.5
    assert target_take.events[0].end == 1.7
    assert timeline.selection.selected_layer_id == target_layer.id
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [EventId("take_snare_main:dup:alt_1:1")]


def test_move_selected_events_rejects_locked_or_hidden_transfer_targets():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    target_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[],
    )
    target_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[target_take],
    )
    timeline.layers.append(target_layer)
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]
    original_start = alt_take.events[0].start
    original_take_ids = [event.id for event in alt_take.events]

    target_layer.presentation_hints.locked = True
    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.5, target_layer_id=target_layer.id),
    )
    assert [event.id for event in alt_take.events] == original_take_ids
    assert alt_take.events[0].start == original_start
    assert target_take.events == []

    target_layer.presentation_hints.locked = False
    target_layer.presentation_hints.visible = False
    orchestrator.handle(
        timeline,
        MoveSelectedEvents(delta_seconds=0.5, target_layer_id=target_layer.id),
    )
    assert [event.id for event in alt_take.events] == original_take_ids
    assert alt_take.events[0].start == original_start
    assert target_take.events == []


def test_move_selected_events_to_adjacent_layer_skips_locked_layers():
    orchestrator, timeline, layer, _main_take, alt_take = _build_orchestrator_and_timeline()
    locked_take = Take(
        id=TakeId("take_snare_main"),
        layer_id=LayerId("layer_snare"),
        name="Main",
        events=[],
    )
    target_take = Take(
        id=TakeId("take_hat_main"),
        layer_id=LayerId("layer_hat"),
        name="Main",
        events=[],
    )
    locked_layer = Layer(
        id=LayerId("layer_snare"),
        timeline_id=timeline.id,
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[locked_take],
    )
    locked_layer.presentation_hints.locked = True
    target_layer = Layer(
        id=LayerId("layer_hat"),
        timeline_id=timeline.id,
        name="Hat",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[target_take],
    )
    timeline.layers.extend([locked_layer, target_layer])
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    orchestrator.handle(timeline, MoveSelectedEventsToAdjacentLayer(direction=1))

    assert [event.id for event in alt_take.events] == [EventId("alt_2")]
    assert [event.id for event in target_take.events] == [EventId("alt_1")]
    assert timeline.selection.selected_layer_id == target_layer.id
    assert timeline.selection.selected_layer_ids == [target_layer.id]
    assert timeline.selection.selected_take_id == target_take.id
    assert timeline.selection.selected_event_ids == [EventId("alt_1")]


def test_nudge_selected_events_moves_selection_by_one_frame_and_preserves_identity():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    event = main_take.events[0]
    original_end = event.end
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [event.id]

    orchestrator.handle(timeline, NudgeSelectedEvents(direction=1))

    assert event.id == EventId("main_1")
    assert event.start == pytest.approx(1.0 + (1.0 / 30.0))
    assert event.end == pytest.approx(original_end + (1.0 / 30.0))
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == main_take.id
    assert timeline.selection.selected_event_ids == [event.id]


def test_nudge_selected_events_clamps_at_zero():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    event = main_take.events[0]
    event.start = 0.01
    event.end = 0.21
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [event.id]

    orchestrator.handle(timeline, NudgeSelectedEvents(direction=-1))

    assert event.start == pytest.approx(0.0)
    assert event.end == pytest.approx(0.2)


def test_duplicate_selected_events_creates_deterministic_ids_and_selects_new_copies():
    orchestrator, timeline, layer, main_take, alt_take = _build_orchestrator_and_timeline()
    selected_ids = [main_take.events[0].id, alt_take.events[0].id]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = selected_ids

    orchestrator.handle(timeline, DuplicateSelectedEvents())

    assert timeline.selection.selected_event_ids == [
        EventId("take_main:dup:main_1:1"),
        EventId("take_alt:dup:alt_1:1"),
    ]
    assert timeline.selection.selected_layer_id == layer.id
    assert timeline.selection.selected_take_id == alt_take.id
    assert any(event.id == EventId("take_main:dup:main_1:1") for event in main_take.events)
    assert any(event.id == EventId("take_alt:dup:alt_1:1") for event in alt_take.events)


def test_duplicate_selected_events_offsets_copies_and_is_repeatable():
    orchestrator, timeline, layer, main_take, _alt_take = _build_orchestrator_and_timeline()
    event = main_take.events[0]
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = main_take.id
    timeline.selection.selected_event_ids = [event.id]

    orchestrator.handle(timeline, DuplicateSelectedEvents())
    first_duplicate = next(
        candidate
        for candidate in main_take.events
        if str(candidate.id) == "take_main:dup:main_1:1"
    )

    timeline.selection.selected_event_ids = [event.id]
    orchestrator.handle(timeline, DuplicateSelectedEvents())
    second_duplicate = next(
        candidate
        for candidate in main_take.events
        if str(candidate.id) == "take_main:dup:main_1:2"
    )

    assert first_duplicate.start == pytest.approx(event.start + (1.0 / 30.0))
    assert second_duplicate.start == pytest.approx(event.start + (1.0 / 30.0))
