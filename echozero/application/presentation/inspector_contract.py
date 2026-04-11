from __future__ import annotations

from dataclasses import dataclass, field

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)


@dataclass(slots=True, frozen=True)
class InspectorObjectIdentity:
    object_id: str
    object_type: str
    label: str


@dataclass(slots=True, frozen=True)
class InspectorFactRow:
    label: str
    value: str


@dataclass(slots=True, frozen=True)
class InspectorSection:
    section_id: str
    label: str
    rows: tuple[InspectorFactRow, ...] = ()


@dataclass(slots=True, frozen=True)
class InspectorAction:
    action_id: str
    label: str
    enabled: bool = True
    kind: str = "intent"
    group: str = "default"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class InspectorContextSection:
    section_id: str
    label: str
    actions: tuple[InspectorAction, ...] = ()


@dataclass(slots=True, frozen=True)
class TimelineInspectorHitTarget:
    kind: str
    layer_id: object | None = None
    take_id: object | None = None
    event_id: object | None = None
    time_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class InspectorContract:
    title: str
    identity: InspectorObjectIdentity | None = None
    sections: tuple[InspectorSection, ...] = ()
    context_sections: tuple[InspectorContextSection, ...] = ()
    empty_state: str = "No timeline object selected."


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
                return _event_contract(
                    presentation,
                    layer=layer,
                    take=take,
                    event=event,
                    hit_target=hit_target,
                )
        if hit_target.take_id is not None:
            take_match = _find_take(presentation, layer_id=hit_target.layer_id, take_id=hit_target.take_id)
            if take_match is not None:
                layer, take = take_match
                return _take_contract(
                    presentation,
                    layer=layer,
                    take=take,
                    hit_target=hit_target,
                )
        if hit_target.layer_id is not None:
            layer = _find_layer(presentation, hit_target.layer_id)
            if layer is not None:
                return _layer_contract(presentation, layer=layer, hit_target=hit_target)
        return _empty_contract(
            presentation,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
        )

    if presentation.selected_event_ids and presentation.selected_layer_id is not None:
        selected_event_id = presentation.selected_event_ids[0]
        event_match = _find_selected_event(presentation, selected_event_id)
        if event_match is not None:
            layer, take, event = event_match
            return _event_contract(
                presentation,
                layer=layer,
                take=take,
                event=event,
                hit_target=None,
            )

    if presentation.selected_layer_id is not None and not presentation.selected_event_ids:
        layer = _find_layer(presentation, presentation.selected_layer_id)
        if layer is not None:
            return _layer_contract(presentation, layer=layer, hit_target=None)

    return _empty_contract(
        presentation,
        hit_target=None,
        has_selected_events=bool(presentation.selected_event_ids),
    )


def render_inspector_contract_text(contract: InspectorContract) -> str:
    if contract.identity is None and not contract.sections:
        return contract.empty_state

    lines: list[str] = [contract.title]
    for section in contract.sections:
        for row in section.rows:
            lines.append(f"{row.label}: {row.value}")
    return "\n".join(lines)


def _empty_contract(
    presentation: TimelinePresentation,
    *,
    hit_target: TimelineInspectorHitTarget | None,
    has_selected_events: bool,
) -> InspectorContract:
    return InspectorContract(
        title="No timeline object selected.",
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=None,
            take=None,
            hit_target=hit_target,
            has_selected_events=has_selected_events,
        ),
    )


def _layer_contract(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    hit_target: TimelineInspectorHitTarget | None,
) -> InspectorContract:
    flags: list[str] = []
    if layer.muted:
        flags.append("muted")
    if layer.soloed:
        flags.append("soloed")
    if layer.status.stale:
        flags.append("stale")
    if layer.status.manually_modified:
        flags.append("edited")

    take_count = 0 if layer.main_take_id is None else 1 + len(layer.takes)
    rows = [
        InspectorFactRow("id", str(layer.layer_id)),
        InspectorFactRow("kind", layer.kind.name),
        InspectorFactRow("main take", str(layer.main_take_id or "none")),
        InspectorFactRow("takes", str(take_count) if take_count else "none"),
        InspectorFactRow("status flags", ", ".join(flags) if flags else "none"),
    ]
    if presentation.experimental_live_sync_enabled:
        rows.append(InspectorFactRow("live sync state", layer.live_sync_state.value))
        if layer.live_sync_pause_reason:
            rows.append(InspectorFactRow("live sync pause", layer.live_sync_pause_reason))
        if layer.live_sync_divergent:
            rows.append(InspectorFactRow("live sync divergence", "diverged"))

    return InspectorContract(
        title=f"Layer {layer.title}",
        identity=InspectorObjectIdentity(
            object_id=str(layer.layer_id),
            object_type="layer",
            label=layer.title,
        ),
        sections=(
            InspectorSection(
                section_id="layer-core",
                label="Layer",
                rows=tuple(rows),
            ),
        ),
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=layer,
            take=None,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
        ),
    )


def _take_contract(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation,
    hit_target: TimelineInspectorHitTarget | None,
) -> InspectorContract:
    return InspectorContract(
        title=f"Take {take.name}",
        identity=InspectorObjectIdentity(
            object_id=str(take.take_id),
            object_type="take",
            label=take.name,
        ),
        sections=(
            InspectorSection(
                section_id="take-core",
                label="Take",
                rows=(
                    InspectorFactRow("id", str(take.take_id)),
                    InspectorFactRow("layer", layer.title),
                    InspectorFactRow("kind", take.kind.name),
                    InspectorFactRow("main truth", "no"),
                    InspectorFactRow("events", str(len(take.events))),
                ),
            ),
        ),
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=layer,
            take=take,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
        ),
    )


def _event_contract(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
    event: EventPresentation,
    hit_target: TimelineInspectorHitTarget | None,
) -> InspectorContract:
    take_name = "Main take" if take is None else take.name
    take_id = layer.main_take_id if take is None else take.take_id
    sections = (
        InspectorSection(
            section_id="event-core",
            label="Event",
            rows=(
                InspectorFactRow("id", str(event.event_id)),
                InspectorFactRow("start", _format_seconds(event.start)),
                InspectorFactRow("end", _format_seconds(event.end)),
                InspectorFactRow("duration", _format_seconds(event.duration)),
                InspectorFactRow("layer", layer.title),
                InspectorFactRow("take", f"{take_name} ({take_id or 'none'})"),
            ),
        ),
    )
    return InspectorContract(
        title=f"Event {event.label}",
        identity=InspectorObjectIdentity(
            object_id=str(event.event_id),
            object_type="event",
            label=event.label,
        ),
        sections=sections,
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=layer,
            take=take,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
        ),
    )


def _shared_context_sections(
    *,
    presentation: TimelinePresentation,
    layer: LayerPresentation | None,
    take: TakeLanePresentation | None,
    hit_target: TimelineInspectorHitTarget | None,
    has_selected_events: bool,
) -> tuple[InspectorContextSection, ...]:
    sections: list[InspectorContextSection] = []

    if hit_target is not None and hit_target.time_seconds is not None:
        sections.append(
            InspectorContextSection(
                section_id="transport",
                label="Transport",
                actions=(
                    InspectorAction(
                        action_id="seek_here",
                        label=f"Seek to {_format_seconds(hit_target.time_seconds)}",
                        group="transport",
                        params={"time_seconds": hit_target.time_seconds},
                    ),
                ),
            )
        )

    if has_selected_events:
        sections.append(
            InspectorContextSection(
                section_id="selection",
                label="Selection",
                actions=(
                    InspectorAction(
                        action_id="nudge_left",
                        label="Nudge Left",
                        group="selection",
                        params={"direction": -1, "steps": 1},
                    ),
                    InspectorAction(
                        action_id="nudge_right",
                        label="Nudge Right",
                        group="selection",
                        params={"direction": 1, "steps": 1},
                    ),
                    InspectorAction(
                        action_id="duplicate",
                        label="Duplicate",
                        group="selection",
                        params={"steps": 1},
                    ),
                ),
            )
        )

    transfer_actions = [
        InspectorAction(
            action_id="pull_from_ma3",
            label="Pull from MA3",
            group="transfer",
        )
    ]
    if has_selected_events:
        transfer_actions.append(
            InspectorAction(
                action_id="push_to_ma3",
                label="Push Selection to MA3",
                group="transfer",
            )
        )
    sections.append(
        InspectorContextSection(
            section_id="transfer",
            label="Transfer",
            actions=tuple(transfer_actions),
        )
    )

    if layer is not None:
        sections.append(
            InspectorContextSection(
                section_id="layer-mix",
                label="Layer",
                actions=(
                    InspectorAction(
                        action_id="toggle_mute",
                        label="Unmute Layer" if layer.muted else "Mute Layer",
                        group="layer",
                        params={"layer_id": layer.layer_id},
                    ),
                    InspectorAction(
                        action_id="toggle_solo",
                        label="Unsolo Layer" if layer.soloed else "Solo Layer",
                        group="layer",
                        params={"layer_id": layer.layer_id},
                    ),
                    InspectorAction(
                        action_id="gain_down",
                        label="Set Gain -6 dB",
                        group="gain",
                        params={"layer_id": layer.layer_id, "gain_db": -6.0},
                    ),
                    InspectorAction(
                        action_id="gain_unity",
                        label="Set Gain 0 dB",
                        group="gain",
                        params={"layer_id": layer.layer_id, "gain_db": 0.0},
                    ),
                    InspectorAction(
                        action_id="gain_up",
                        label="Set Gain +6 dB",
                        group="gain",
                        params={"layer_id": layer.layer_id, "gain_db": 6.0},
                    ),
                ),
            )
        )

    if presentation.experimental_live_sync_enabled and layer is not None:
        actions = [
            InspectorAction(
                action_id="live_sync_set_off",
                label="Set Off",
                group="live_sync",
                params={"layer_id": layer.layer_id},
            ),
            InspectorAction(
                action_id="live_sync_set_observe",
                label="Set Observe",
                group="live_sync",
                params={"layer_id": layer.layer_id},
            ),
            InspectorAction(
                action_id="live_sync_set_armed_write",
                label="Set Armed Write",
                group="live_sync",
                params={"layer_id": layer.layer_id},
            ),
            InspectorAction(
                action_id="live_sync_set_pause_reason",
                label="Operator Pause",
                group="live_sync",
                params={"layer_id": layer.layer_id, "pause_reason": "operator pause"},
            ),
        ]
        if layer.live_sync_pause_reason:
            actions.append(
                InspectorAction(
                    action_id="live_sync_clear_pause_reason",
                    label="Clear Pause Reason",
                    group="live_sync",
                    params={"layer_id": layer.layer_id},
                )
            )
        sections.append(
            InspectorContextSection(
                section_id="live-sync",
                label="Live Sync",
                actions=tuple(actions),
            )
        )

    take_actions = _take_actions_for_contract(take) if take is not None else ()
    if take is not None and take_actions:
        sections.append(
            InspectorContextSection(
                section_id="take-actions",
                label="Take",
                actions=tuple(_map_take_action(layer, take, action) for action in take_actions),
            )
        )

    return tuple(section for section in sections if section.actions)


def _map_take_action(
    layer: LayerPresentation,
    take: TakeLanePresentation,
    action: TakeActionPresentation,
) -> InspectorAction:
    return InspectorAction(
        action_id=action.action_id,
        label=action.label,
        group="take",
        params={
            "layer_id": layer.layer_id,
            "take_id": take.take_id,
        },
    )


def _take_actions_for_contract(take: TakeLanePresentation) -> tuple[TakeActionPresentation, ...]:
    if take.actions:
        return tuple(take.actions)
    return (
        TakeActionPresentation(action_id="overwrite_main", label="Overwrite Main"),
        TakeActionPresentation(action_id="merge_main", label="Merge Main"),
    )


def _find_layer(presentation: TimelinePresentation, layer_id: object) -> LayerPresentation | None:
    for layer in presentation.layers:
        if layer.layer_id == layer_id:
            return layer
    return None


def _find_take(
    presentation: TimelinePresentation,
    *,
    layer_id: object | None,
    take_id: object,
) -> tuple[LayerPresentation, TakeLanePresentation] | None:
    layer = _find_layer(presentation, layer_id) if layer_id is not None else None
    if layer is not None:
        for take in layer.takes:
            if take.take_id == take_id:
                return layer, take
    for candidate_layer in presentation.layers:
        for take in candidate_layer.takes:
            if take.take_id == take_id:
                return candidate_layer, take
    return None


def _find_event(
    presentation: TimelinePresentation,
    *,
    layer_id: object | None,
    take_id: object | None,
    event_id: object,
) -> tuple[LayerPresentation, TakeLanePresentation | None, EventPresentation] | None:
    layer = _find_layer(presentation, layer_id) if layer_id is not None else None
    layers = [layer] if layer is not None else list(presentation.layers)
    for candidate_layer in layers:
        if candidate_layer is None:
            continue
        if take_id is None or take_id == candidate_layer.main_take_id:
            for event in candidate_layer.events:
                if event.event_id == event_id:
                    return candidate_layer, None, event
        for take in candidate_layer.takes:
            if take_id is not None and take.take_id != take_id:
                continue
            for event in take.events:
                if event.event_id == event_id:
                    return candidate_layer, take, event
    return None


def _find_selected_event(
    presentation: TimelinePresentation,
    event_id: object,
) -> tuple[LayerPresentation, TakeLanePresentation | None, EventPresentation] | None:
    return _find_event(
        presentation,
        layer_id=presentation.selected_layer_id,
        take_id=presentation.selected_take_id,
        event_id=event_id,
    ) or _find_event(
        presentation,
        layer_id=presentation.selected_layer_id,
        take_id=None,
        event_id=event_id,
    )


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"
