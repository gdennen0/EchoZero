"""Inspector lookup seam coverage for the canonical presentation layer.
Exists to freeze the routing assumptions behind the public inspector entrypoint split.
Connects lookup helpers to stable layer, take, and event resolution behavior in tests.
"""

from echozero.application.presentation.inspector_contract_lookup import (
    find_event,
    find_event_ref,
    find_layer,
    find_selected_event,
    find_take,
)
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.application.timeline.models import EventRef


def _lookup_test_presentation() -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_lookup"),
        title="Lookup",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_main"),
                title="Main",
                main_take_id=TakeId("take_main"),
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("shared_evt"),
                        start=1.0,
                        end=1.5,
                        label="Main Shared",
                    ),
                    EventPresentation(
                        event_id=EventId("main_evt"),
                        start=2.0,
                        end=2.5,
                        label="Main Only",
                    ),
                ],
                takes=[
                    TakeLanePresentation(
                        take_id=TakeId("take_alt"),
                        name="Take 2",
                        kind=LayerKind.EVENT,
                        events=[
                            EventPresentation(
                                event_id=EventId("shared_evt"),
                                start=3.0,
                                end=3.5,
                                label="Take Shared",
                            ),
                            EventPresentation(
                                event_id=EventId("take_evt"),
                                start=4.0,
                                end=4.5,
                                label="Take Only",
                            ),
                        ],
                    )
                ],
                status=LayerStatusPresentation(),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_other"),
                title="Other",
                main_take_id=TakeId("take_other_main"),
                kind=LayerKind.EVENT,
                events=[],
                takes=[
                    TakeLanePresentation(
                        take_id=TakeId("take_remote"),
                        name="Remote",
                        kind=LayerKind.EVENT,
                        events=[],
                    )
                ],
                status=LayerStatusPresentation(),
            ),
        ],
    )


def test_find_layer_returns_matching_layer() -> None:
    presentation = _lookup_test_presentation()

    layer = find_layer(presentation, LayerId("layer_main"))

    assert layer is not None
    assert layer.title == "Main"


def test_find_take_falls_back_to_global_search_when_hinted_layer_misses() -> None:
    presentation = _lookup_test_presentation()

    match = find_take(
        presentation,
        layer_id=LayerId("layer_main"),
        take_id=TakeId("take_remote"),
    )

    assert match is not None
    layer, take = match
    assert layer.layer_id == LayerId("layer_other")
    assert take.take_id == TakeId("take_remote")


def test_find_event_treats_main_take_id_as_main_layer_event_scope() -> None:
    presentation = _lookup_test_presentation()

    match = find_event(
        presentation,
        layer_id=LayerId("layer_main"),
        take_id=TakeId("take_main"),
        event_id=EventId("main_evt"),
    )

    assert match is not None
    layer, take, event = match
    assert layer.layer_id == LayerId("layer_main")
    assert take is None
    assert event.event_id == EventId("main_evt")


def test_find_event_ref_returns_none_for_missing_ref() -> None:
    presentation = _lookup_test_presentation()

    assert find_event_ref(presentation, None) is None


def test_find_selected_event_prefers_selected_take_context_for_shared_event_ids() -> None:
    presentation = _lookup_test_presentation()
    presentation.selected_layer_id = LayerId("layer_main")
    presentation.selected_take_id = TakeId("take_alt")
    presentation.selected_layer_ids = [LayerId("layer_other"), LayerId("layer_main")]

    match = find_selected_event(presentation, EventId("shared_evt"))

    assert match is not None
    layer, take, event = match
    assert layer.layer_id == LayerId("layer_main")
    assert take is not None
    assert take.take_id == TakeId("take_alt")
    assert event.label == "Take Shared"


def test_find_event_ref_uses_event_ref_coordinates() -> None:
    presentation = _lookup_test_presentation()
    event_ref = EventRef(
        layer_id=LayerId("layer_main"),
        take_id=TakeId("take_alt"),
        event_id=EventId("take_evt"),
    )

    match = find_event_ref(presentation, event_ref)

    assert match is not None
    _, take, event = match
    assert take is not None
    assert take.take_id == TakeId("take_alt")
    assert event.label == "Take Only"
