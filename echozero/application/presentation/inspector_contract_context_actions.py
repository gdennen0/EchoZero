"""Context-action helpers for the timeline inspector contract.
Exists to isolate inspector action-group assembly from contract builders and fact-row support helpers.
Connects timeline presentation state to reusable song, selection, transfer, layer, and take action sections.
"""

from __future__ import annotations

from echozero.application.presentation.inspector_contract_preview import (
    event_preview_params,
)
from echozero.application.presentation.inspector_contract_types import (
    InspectorAction,
    InspectorContextSection,
    TimelineInspectorHitTarget,
)
from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    EventPresentation,
    LayerPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
    default_take_actions,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.event_batch_scope import (
    EventBatchScope,
    event_batch_scope_params,
)
from echozero.application.timeline.object_actions import (
    SONG_ADD_DESCRIPTOR,
    pipeline_actions_for_audio_layer,
)


def event_context_sections(
    *,
    presentation: TimelinePresentation,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
    event: EventPresentation,
) -> tuple[InspectorContextSection, ...]:
    preview_params = event_preview_params(
        presentation,
        layer=layer,
        take=take,
        event=event,
    )
    if preview_params is None:
        return ()
    return (
        InspectorContextSection(
            section_id="event-preview",
            label="Clip",
            actions=(
                InspectorAction(
                    action_id="preview_event_clip",
                    label="Play Clip",
                    group="selection",
                    params=preview_params,
                ),
            ),
        ),
    )


def batch_context_actions(
    *,
    presentation: TimelinePresentation,
    layer: LayerPresentation | None,
    take: TakeLanePresentation | None,
    hit_target: TimelineInspectorHitTarget | None,
    has_selected_events: bool,
) -> tuple[InspectorAction, ...]:
    scope = resolve_batch_scope_for_contract(
        presentation=presentation,
        layer=layer,
        take=take,
        hit_target=hit_target,
        has_selected_events=has_selected_events,
    )
    if scope is None or not scope_has_events(presentation, scope):
        return ()

    suffix = batch_scope_label_suffix(scope)
    return (
        InspectorAction(
            action_id="selection.select_every_other",
            label=f"Select Every Other{suffix}",
            group="batch",
            params=event_batch_scope_params(scope),
        ),
        InspectorAction(
            action_id="selection.renumber_cues_from_one",
            label=f"Renumber Cues from 1{suffix}",
            group="batch",
            params=event_batch_scope_params(scope),
        ),
    )


def resolve_batch_scope_for_contract(
    *,
    presentation: TimelinePresentation,
    layer: LayerPresentation | None,
    take: TakeLanePresentation | None,
    hit_target: TimelineInspectorHitTarget | None,
    has_selected_events: bool,
) -> EventBatchScope | None:
    if has_selected_events and (hit_target is None or hit_target.kind == "event"):
        if hit_target is None:
            return EventBatchScope(mode="selected_events")
        if hit_target.event_id in set(presentation.selected_event_ids):
            return EventBatchScope(mode="selected_events")

    if take is not None and layer is not None:
        return EventBatchScope(mode="take", layer_id=layer.layer_id, take_id=take.take_id)

    if layer is None:
        if has_selected_events:
            return EventBatchScope(mode="selected_events")
        if presentation.selected_region_id is not None:
            return EventBatchScope(mode="region", region_id=presentation.selected_region_id)
        return None

    selected_layer_ids = list(dict.fromkeys(presentation.selected_layer_ids))
    if not selected_layer_ids and presentation.selected_layer_id is not None:
        selected_layer_ids = [presentation.selected_layer_id]
    if len(selected_layer_ids) > 1 and layer.layer_id in selected_layer_ids:
        return EventBatchScope(mode="selected_layers_main")
    return EventBatchScope(mode="layer_main", layer_id=layer.layer_id)


def batch_scope_label_suffix(scope: EventBatchScope) -> str:
    if scope.mode == "selected_events":
        return ""
    if scope.mode == "take":
        return " in Take"
    if scope.mode == "region":
        return " in Region"
    if scope.mode == "selected_layers_main":
        return " in Selected Layers"
    return " in Layer"


def scope_has_events(
    presentation: TimelinePresentation,
    scope: EventBatchScope,
) -> bool:
    if scope.mode == "selected_events":
        return bool(presentation.selected_event_ids)
    if scope.mode == "region":
        region = next(
            (candidate for candidate in presentation.regions if candidate.region_id == scope.region_id),
            None,
        )
        if region is None:
            return False
        return any(
            layer.visible
            and not layer.locked
            and any(
                float(event.start) < float(region.end) and float(event.end) > float(region.start)
                for event in layer.events
            )
            for layer in presentation.layers
        )
    if scope.mode == "selected_layers_main":
        selected_layer_ids = list(dict.fromkeys(presentation.selected_layer_ids))
        if not selected_layer_ids and presentation.selected_layer_id is not None:
            selected_layer_ids = [presentation.selected_layer_id]
        return any(
            layer.layer_id in selected_layer_ids and bool(layer.events)
            for layer in presentation.layers
        )
    if scope.mode == "layer_main":
        return any(
            layer.layer_id == scope.layer_id and bool(layer.events) for layer in presentation.layers
        )
    return any(
        layer.layer_id == scope.layer_id
        and any(take.take_id == scope.take_id and bool(take.events) for take in layer.takes)
        for layer in presentation.layers
    )


def shared_context_sections(
    *,
    presentation: TimelinePresentation,
    layer: LayerPresentation | None,
    take: TakeLanePresentation | None,
    hit_target: TimelineInspectorHitTarget | None,
    has_selected_events: bool,
    include_layer_transfer_controls: bool,
) -> tuple[InspectorContextSection, ...]:
    sections: list[InspectorContextSection] = []
    song_actions = song_context_actions(presentation)
    if song_actions:
        sections.append(
            InspectorContextSection(
                section_id="song",
                label="Song",
                actions=song_actions,
            )
        )
    if layer is None and take is None and not has_selected_events:
        sections.append(
            InspectorContextSection(
                section_id="tools",
                label="Tools",
                actions=(
                    InspectorAction(
                        action_id=SONG_ADD_DESCRIPTOR.action_id,
                        label=SONG_ADD_DESCRIPTOR.label,
                        group="tools",
                    ),
                    InspectorAction(
                        action_id="add_event_layer",
                        label="Add Event Layer",
                        group="tools",
                    ),
                    InspectorAction(
                        action_id="pull_from_ma3",
                        label="Import Event Layer from MA3",
                        group="tools",
                    ),
                ),
            )
        )

    if hit_target is not None and hit_target.time_seconds is not None:
        sections.append(
            InspectorContextSection(
                section_id="transport",
                label="Transport",
                actions=(
                    InspectorAction(
                        action_id="seek_here",
                        label=f"Seek to {format_seconds(hit_target.time_seconds)}",
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
                        action_id="timeline.duplicate_selection",
                        label="Duplicate",
                        group="selection",
                        params={"steps": 1},
                    ),
                ),
            )
        )

    batch_actions = batch_context_actions(
        presentation=presentation,
        layer=layer,
        take=take,
        hit_target=hit_target,
        has_selected_events=has_selected_events,
    )
    if batch_actions:
        sections.append(
            InspectorContextSection(
                section_id="event-batch",
                label="Batch Edit",
                actions=batch_actions,
            )
        )

        if layer is not None:
            transfer_actions: list[InspectorAction] = []
            if layer_supports_ma3_transfer(layer):
                route_label = (
                    "Change MA3 Route"
                    if layer.sync_target_label
                    else "Route Layer to MA3 Track"
                )
                if hit_target is not None and hit_target.kind == "event":
                    explicit_event_ids: list[str] = []
                    if (
                        hit_target.event_id is not None
                        and hit_target.take_id in {None, layer.main_take_id}
                        and not has_selected_events
                    ):
                        explicit_event_ids = [hit_target.event_id]
                    transfer_actions.append(
                        InspectorAction(
                            action_id="send_selected_events_to_ma3",
                            label=(
                                "Send Event to MA3"
                                if explicit_event_ids
                                else "Send Selected Events to MA3"
                            ),
                            group="transfer",
                            params={
                                "layer_id": layer.layer_id,
                                "event_ids": explicit_event_ids,
                            },
                            enabled=bool(explicit_event_ids) or (layer.is_selected and has_selected_events),
                        )
                    )
                else:
                    transfer_actions.extend(
                        [
                            InspectorAction(
                                action_id="pull_from_ma3",
                                label="Import Event Layer from MA3",
                                group="transfer",
                                params={"layer_id": layer.layer_id},
                            ),
                            InspectorAction(
                                action_id="route_layer_to_ma3_track",
                                label=route_label,
                                group="transfer",
                                params={"layer_id": layer.layer_id},
                            ),
                            InspectorAction(
                                action_id="send_layer_to_ma3",
                                label="Send Layer to MA3",
                                group="transfer",
                                params={"layer_id": layer.layer_id},
                                enabled=layer.main_take_id is not None,
                            ),
                            InspectorAction(
                                action_id="send_selected_events_to_ma3",
                                label="Send Selected Events to MA3",
                                group="transfer",
                                params={"layer_id": layer.layer_id},
                                enabled=layer.is_selected and has_selected_events,
                            ),
                            InspectorAction(
                                action_id="send_to_different_track_once",
                                label="Send to Different Track Once",
                                group="transfer",
                                params={"layer_id": layer.layer_id},
                                enabled=layer.main_take_id is not None,
                            ),
                        ]
                    )
            if transfer_actions:
                sections.append(
                    InspectorContextSection(
                        section_id="sync-transfer",
                        label="Sync & Transfer",
                        actions=tuple(transfer_actions),
                    )
                )
    else:
        pass

    if layer is not None:
        layer_actions = [
            InspectorAction(
                action_id="set_active_playback_target",
                label=(
                    "Audio Routed to Master"
                    if presentation.active_playback_layer_id == layer.layer_id
                    else "Route Audio to Master"
                ),
                group="layer",
                params={"layer_id": layer.layer_id},
                enabled=bool(layer.source_audio_path or layer.playback_source_ref),
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
            InspectorAction(
                action_id="delete_layer",
                label="Delete Layer",
                group="layer",
                params={"layer_id": layer.layer_id},
                enabled=layer.layer_id != "source_audio",
            ),
        ]
        layer_actions.extend(pipeline_actions_for_layer(layer))
        sections.append(
            InspectorContextSection(
                section_id="layer-mix",
                label="Layer",
                actions=tuple(layer_actions),
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

    take_actions = take_actions_for_contract(take) if take is not None else ()
    if take is not None and take_actions:
        assert layer is not None
        sections.append(
            InspectorContextSection(
                section_id="take-actions",
                label="Take",
                actions=tuple(map_take_action(layer, take, action) for action in take_actions),
            )
        )

    return tuple(section for section in sections if section.actions)


def song_context_actions(
    presentation: TimelinePresentation,
) -> tuple[InspectorAction, ...]:
    actions: list[InspectorAction] = []
    if presentation.available_songs:
        actions.append(
            InspectorAction(
                action_id="song.select",
                label="Select Song",
                group="song",
                enabled=len(presentation.available_songs) > 0,
            )
        )
    if presentation.active_song_id:
        actions.append(
            InspectorAction(
                action_id="song.version.switch",
                label="Switch Version",
                group="song",
                enabled=len(presentation.available_song_versions) > 1,
                params={"song_id": presentation.active_song_id},
            )
        )
        actions.append(
            InspectorAction(
                action_id="song.version.add",
                label="Add Version",
                group="song",
                params={"song_id": presentation.active_song_id},
            )
        )
        actions.append(
            InspectorAction(
                action_id="song.delete",
                label="Delete Song",
                group="song",
                params={"song_id": presentation.active_song_id},
            )
        )
    if presentation.active_song_version_id:
        label = (
            f"Set MA3 TC Pool (TC{presentation.active_song_version_ma3_timecode_pool_no})"
            if presentation.active_song_version_ma3_timecode_pool_no is not None
            else "Set MA3 TC Pool"
        )
        actions.append(
            InspectorAction(
                action_id="song.version.delete",
                label="Delete Version",
                group="song",
                params={"song_version_id": presentation.active_song_version_id},
            )
        )
        actions.append(
            InspectorAction(
                action_id="song.version.set_ma3_timecode_pool",
                label=label,
                group="song",
                params={"song_version_id": presentation.active_song_version_id},
            )
        )
    return tuple(actions)


def map_take_action(
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


def take_actions_for_contract(take: TakeLanePresentation) -> tuple[TakeActionPresentation, ...]:
    if take.actions:
        return tuple(take.actions)
    return tuple(default_take_actions())


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def pipeline_actions_for_layer(layer: LayerPresentation) -> tuple[InspectorAction, ...]:
    descriptors = pipeline_actions_for_audio_layer(
        is_stem_capable=is_stem_capable_layer(layer),
        is_drum_capable=is_drum_capable_layer(layer),
        is_song_drum_capable=is_song_drum_capable_layer(layer),
    )
    return tuple(
        InspectorAction(
            action_id=descriptor.action_id,
            label=descriptor.label,
            group="pipeline",
            params={"layer_id": layer.layer_id, **descriptor.static_params},
        )
        for descriptor in descriptors
    )


def is_stem_capable_layer(layer: LayerPresentation) -> bool:
    return layer.kind.name == "AUDIO"


def is_drum_capable_layer(layer: LayerPresentation) -> bool:
    if not is_stem_capable_layer(layer):
        return False
    title = layer.title.strip().lower()
    badges = {str(badge).strip().lower() for badge in layer.badges}
    source_label = (layer.status.source_label if layer.status is not None else "").strip().lower()
    return "drum" in title or "drums" in badges or "drum" in source_label


def is_song_drum_capable_layer(layer: LayerPresentation) -> bool:
    if not is_stem_capable_layer(layer) or is_drum_capable_layer(layer):
        return False
    if layer.status is None:
        return True
    source_layer_id = (layer.status.source_layer_id or "").strip()
    pipeline_id = (layer.status.pipeline_id or "").strip()
    return not source_layer_id and not pipeline_id


def layer_supports_ma3_transfer(layer: LayerPresentation) -> bool:
    return layer.kind is LayerKind.EVENT and layer.main_take_id is not None


def preview_transfer_plan_label(plan: BatchTransferPlanPresentation) -> str:
    return f"Preview Transfer Plan ({ready_count_label(plan.ready_count)})"


def apply_transfer_plan_label(plan: BatchTransferPlanPresentation) -> str:
    return f"Apply Transfer Plan ({ready_count_label(plan.ready_count)})"


def ready_count_label(count: int) -> str:
    noun = "ready row" if count == 1 else "ready rows"
    return f"{count} {noun}"
