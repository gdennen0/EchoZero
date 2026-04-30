"""
TimelineAssemblyState: Typed selection and playback snapshot for presentation shaping.
Exists to keep TimelineAssembler argument lists short and explicit across helper modules.
Connects timeline truth to cached layer assembly without widget-owned state.
"""

from dataclasses import dataclass

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.models import EventRef, Timeline

SelectionEventRefKey = tuple[str, str, str]
AssemblerSignature = tuple[object, ...]

__all__ = [
    "AssemblerSignature",
    "SelectionEventRefKey",
    "TimelineAssemblyState",
    "build_timeline_assembly_state",
]


@dataclass(frozen=True, slots=True)
class TimelineAssemblyState:
    """Immutable selection and playback inputs for one timeline presentation build."""

    selected_layer_id: LayerId | None
    selected_layer_ids: tuple[LayerId, ...]
    selected_take_id: TakeId | None
    selected_event_refs: tuple[EventRef, ...]
    selected_event_ref_keys: frozenset[SelectionEventRefKey]
    selected_event_ids: tuple[EventId, ...]
    selected_event_ids_set: frozenset[EventId]


def build_timeline_assembly_state(timeline: Timeline) -> TimelineAssemblyState:
    """Capture the selection and playback state the assembler depends on."""

    selected_layer_ids = list(timeline.selection.selected_layer_ids)
    if not selected_layer_ids and timeline.selection.selected_layer_id is not None:
        selected_layer_ids = [timeline.selection.selected_layer_id]

    selected_event_refs = tuple(timeline.selection.selected_event_refs)
    selected_event_ids = tuple(timeline.selection.selected_event_ids)
    return TimelineAssemblyState(
        selected_layer_id=timeline.selection.selected_layer_id,
        selected_layer_ids=tuple(selected_layer_ids),
        selected_take_id=timeline.selection.selected_take_id,
        selected_event_refs=selected_event_refs,
        selected_event_ref_keys=_selected_event_ref_keys(selected_event_refs),
        selected_event_ids=selected_event_ids,
        selected_event_ids_set=frozenset(selected_event_ids),
    )


def _selected_event_ref_keys(
    selected_event_refs: tuple[EventRef, ...],
) -> frozenset[SelectionEventRefKey]:
    return frozenset(
        (
            str(event_ref.layer_id),
            str(event_ref.take_id),
            str(event_ref.event_id),
        )
        for event_ref in selected_event_refs
    )
