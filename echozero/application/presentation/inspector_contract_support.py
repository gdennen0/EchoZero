"""Inspector fact-row helpers for the canonical presentation layer.
Exists to keep selection, playback, and transfer summary rows separate from contract assembly.
Connects timeline presentation state to reusable inspector labels and fact rows.
"""

from __future__ import annotations

from echozero.application.presentation.inspector_contract_lookup import (
    find_layer,
    find_selected_event,
    find_take,
)
from echozero.application.presentation.inspector_contract_types import (
    InspectorFactRow,
)
from echozero.application.presentation.models import (
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)


def selection_playback_context_rows(
    presentation: TimelinePresentation,
) -> tuple[InspectorFactRow, ...]:
    selection_label = selected_identity_label(presentation)
    playback_label = playback_target_label(presentation)
    if selection_label == "none" and playback_label == "none":
        return ()
    return (
        InspectorFactRow("selected identity", selection_label),
        InspectorFactRow("playback target", playback_label),
    )


def selected_identity_label(presentation: TimelinePresentation) -> str:
    if presentation.selected_event_ids:
        event_match = find_selected_event(presentation, presentation.selected_event_ids[0])
        if event_match is not None:
            layer, take, event = event_match
            take_label = take_identity_label(layer, take)
            return f"Event {event.label} ({event.event_id}) on {layer.title} / {take_label}"
    if presentation.selected_take_id is not None and presentation.selected_layer_id is not None:
        take_match = find_take(
            presentation,
            layer_id=presentation.selected_layer_id,
            take_id=presentation.selected_take_id,
        )
        if take_match is not None:
            layer, take = take_match
            return f"Take {take.name} ({take.take_id}) on {layer.title}"
        selected_layer = find_layer(presentation, presentation.selected_layer_id)
        if selected_layer is not None and presentation.selected_take_id == selected_layer.main_take_id:
            return f"Take Main take ({selected_layer.main_take_id}) on {selected_layer.title}"
    if presentation.selected_layer_id is not None:
        selected_layer = find_layer(presentation, presentation.selected_layer_id)
        if selected_layer is not None:
            return f"Layer {selected_layer.title} ({selected_layer.layer_id})"
    return "none"


def playback_target_label(presentation: TimelinePresentation) -> str:
    layer_id = presentation.active_playback_layer_id
    if layer_id is None:
        return "none"
    layer = find_layer(presentation, layer_id)
    if layer is None:
        return f"Active layer {layer_id}"
    take_id = presentation.active_playback_take_id
    if take_id is None or take_id == layer.main_take_id:
        return f"Active {layer.title} / Main take ({layer.main_take_id or 'none'})"
    take_match = find_take(presentation, layer_id=layer.layer_id, take_id=take_id)
    if take_match is not None:
        _, take = take_match
        return f"Active {layer.title} / {take.name} ({take.take_id})"
    return f"Active {layer.title} / Take {take_id}"


def playback_state_label(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
) -> str:
    if presentation.active_playback_layer_id != layer.layer_id:
        return "Set Active"
    active_take_id = presentation.active_playback_take_id
    if take is None:
        return "Active" if active_take_id in (None, layer.main_take_id) else "Set Active"
    return "Active" if active_take_id == take.take_id else "Set Active"


def take_identity_label(layer: LayerPresentation, take: TakeLanePresentation | None) -> str:
    if take is None:
        return f"Main take ({layer.main_take_id or 'none'})"
    return f"{take.name} ({take.take_id})"


def sync_state_label(layer: LayerPresentation) -> str:
    state = layer.live_sync_state.value.replace("_", " ")
    return state.title()


def layer_transfer_rows(
    presentation: TimelinePresentation,
    layer: LayerPresentation,
) -> tuple[InspectorFactRow, ...]:
    rows = [
        InspectorFactRow("sync state", sync_state_label(layer)),
        InspectorFactRow("sync mapping", layer.sync_target_label or "none"),
    ]
    return tuple(rows)
