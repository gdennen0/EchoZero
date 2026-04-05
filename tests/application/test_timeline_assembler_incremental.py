from echozero.application.session.models import Session
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
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import Event, Layer, Take, Timeline


def _event(event_id: str, take_id: str, start: float) -> Event:
    return Event(
        id=EventId(event_id),
        take_id=TakeId(take_id),
        start=start,
        end=start + 0.2,
        label=event_id,
    )


def _timeline() -> Timeline:
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        events=[_event("e1", "take_main", 1.0), _event("e2", "take_main", 2.0)],
    )
    alt_take = Take(
        id=TakeId("take_alt"),
        layer_id=LayerId("layer_1"),
        name="Take 2",
        events=[_event("a1", "take_alt", 1.5)],
    )
    layer = Layer(
        id=LayerId("layer_1"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take, alt_take],
    )
    return Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )


def _session() -> Session:
    return Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=TimelineId("timeline_1"),
    )


def test_assembler_reuses_layer_presentations_when_structure_unchanged():
    assembler = TimelineAssembler()
    timeline = _timeline()
    session = _session()

    first = assembler.assemble(timeline, session)

    # transport-only update
    session.transport_state.playhead = 10.0
    second = assembler.assemble(timeline, session)

    assert first.layers is second.layers


def test_assembler_rebuilds_layers_when_take_events_change():
    assembler = TimelineAssembler()
    timeline = _timeline()
    session = _session()

    first = assembler.assemble(timeline, session)

    timeline.layers[0].takes[0].events.append(_event("e3", "take_main", 3.0))
    second = assembler.assemble(timeline, session)

    assert first.layers is not second.layers
    assert len(second.layers[0].events) == 3
