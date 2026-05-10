"""Inspector contract builders for the canonical timeline presentation layer.
Exists to turn selected timeline objects into stable inspector sections and actions.
Connects presentation models to the Qt inspector surface without widget-owned truth.
"""

from __future__ import annotations

from echozero.application.presentation.inspector_contract_builders import (
    build_empty_contract as _build_empty_contract,
    build_event_contract as _build_event_contract,
    build_layer_contract as _build_layer_contract,
    build_song_version_contract as _build_song_version_contract,
    build_take_contract as _build_take_contract,
)
from echozero.application.presentation.inspector_contract_lookup import (
    find_event as _find_event,
    find_event_ref as _find_event_ref,
    find_layer as _find_layer,
    find_selected_event as _find_selected_event,
    find_take as _find_take,
)
from echozero.application.presentation.inspector_contract_types import (
    InspectorAction,
    InspectorContextSection,
    InspectorContract,
    InspectorFactRow,
    InspectorObjectIdentity,
    InspectorSection,
    TimelineInspectorHitTarget,
)
from echozero.application.presentation.models import TimelinePresentation

__all__ = [
    "InspectorAction",
    "InspectorContextSection",
    "InspectorContract",
    "InspectorFactRow",
    "InspectorObjectIdentity",
    "InspectorSection",
    "TimelineInspectorHitTarget",
    "build_timeline_inspector_contract",
    "render_inspector_contract_text",
]


def build_timeline_inspector_contract(
    presentation: TimelinePresentation,
    *,
    hit_target: TimelineInspectorHitTarget | None = None,
) -> InspectorContract:
    if hit_target is not None:
        if hit_target.event_id is not None:
            event_match = _find_event(
                presentation,
                layer_id=hit_target.layer_id,
                take_id=hit_target.take_id,
                event_id=hit_target.event_id,
            )
            if event_match is not None:
                layer, take, event = event_match
                return _build_event_contract(
                    presentation,
                    layer=layer,
                    take=take,
                    event=event,
                    hit_target=hit_target,
                )
        if hit_target.take_id is not None:
            take_match = _find_take(
                presentation, layer_id=hit_target.layer_id, take_id=hit_target.take_id
            )
            if take_match is not None:
                layer, take = take_match
                return _build_take_contract(
                    presentation,
                    layer=layer,
                    take=take,
                    hit_target=hit_target,
                )
        if hit_target.layer_id is not None:
            selected_layer = _find_layer(presentation, hit_target.layer_id)
            if selected_layer is not None:
                return _build_layer_contract(
                    presentation,
                    layer=selected_layer,
                    hit_target=hit_target,
                )
        if presentation.active_song_version_id:
            return _build_song_version_contract(
                presentation,
                hit_target=hit_target,
                has_selected_events=bool(presentation.selected_event_ids),
            )
        return _build_empty_contract(
            presentation,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
        )

    selected_ref = presentation.primary_selected_event_ref()
    if selected_ref is not None:
        event_match = _find_event_ref(presentation, selected_ref)
        if event_match is not None:
            layer, take, event = event_match
            return _build_event_contract(
                presentation,
                layer=layer,
                take=take,
                event=event,
                hit_target=None,
            )

    if presentation.selected_event_ids and presentation.selected_layer_id is not None:
        selected_event_id = presentation.selected_event_ids[0]
        event_match = _find_selected_event(presentation, selected_event_id)
        if event_match is not None:
            layer, take, event = event_match
            return _build_event_contract(
                presentation,
                layer=layer,
                take=take,
                event=event,
                hit_target=None,
            )

    if presentation.selected_layer_id is not None and not presentation.selected_event_ids:
        selected_layer = _find_layer(presentation, presentation.selected_layer_id)
        if selected_layer is not None:
            return _build_layer_contract(
                presentation,
                layer=selected_layer,
                hit_target=None,
            )

    if presentation.active_song_version_id:
        return _build_song_version_contract(
            presentation,
            hit_target=None,
            has_selected_events=bool(presentation.selected_event_ids),
        )

    return _build_empty_contract(
        presentation,
        hit_target=None,
        has_selected_events=bool(presentation.selected_event_ids),
    )


def render_inspector_contract_text(contract: InspectorContract) -> str:
    """Render a compact text summary of an inspector contract for tests and logs."""

    if contract.identity is None and not contract.sections:
        return contract.empty_state

    lines: list[str] = [contract.title]
    for section in contract.sections:
        for row in section.rows:
            lines.append(f"{row.label}: {row.value}")
    return "\n".join(lines)
