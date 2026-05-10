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

    selected_event_ids = tuple(dict.fromkeys(timeline.selection.selected_event_ids))
    selected_event_refs = _resolve_selection_event_refs(
        timeline,
        selected_event_ids=selected_event_ids,
        selected_layer_ids=tuple(selected_layer_ids),
        selected_take_id=timeline.selection.selected_take_id,
    )
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


def _resolve_selection_event_refs(
    timeline: Timeline,
    *,
    selected_event_ids: tuple[EventId, ...],
    selected_layer_ids: tuple[LayerId, ...],
    selected_take_id: TakeId | None,
) -> tuple[EventRef, ...]:
    if not selected_event_ids:
        return ()

    preferred_layers = set(selected_layer_ids)
    matches_by_event_id: dict[str, list[EventRef]] = {}
    for layer in timeline.layers:
        for take in layer.takes:
            for event in take.events:
                matches_by_event_id.setdefault(str(event.id), []).append(
                    EventRef(layer_id=layer.id, take_id=take.id, event_id=event.id)
                )

    selected_event_refs: list[EventRef] = []
    seen: set[SelectionEventRefKey] = set()
    for event_id in selected_event_ids:
        matches = list(matches_by_event_id.get(str(event_id), ()))
        if not matches:
            continue
        preferred_matches = [
            match
            for match in matches
            if (
                (selected_take_id is None or match.take_id == selected_take_id)
                and (not preferred_layers or match.layer_id in preferred_layers)
            )
        ]
        if preferred_matches:
            matches = preferred_matches[:1]
        elif selected_take_id is not None:
            take_matches = [match for match in matches if match.take_id == selected_take_id]
            if take_matches:
                matches = take_matches[:1]
        for match in matches:
            key = (str(match.layer_id), str(match.take_id), str(match.event_id))
            if key in seen:
                continue
            selected_event_refs.append(match)
            seen.add(key)
    return tuple(selected_event_refs)
