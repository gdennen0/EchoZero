"""Demo-only presentation mutation helpers for the timeline fixture app.
Exists to keep support-only selection, event, and transfer mutations out of the demo app runtime shell.
Never use these helpers as canonical timeline truth mutation logic.
"""

from __future__ import annotations

from dataclasses import replace

from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    BatchTransferPlanRowPresentation,
    EventPresentation,
    LayerPresentation,
    ManualPullEventOptionPresentation,
    ManualPullFlowPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    ManualPushFlowPresentation,
    ManualPushTrackOptionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import EventId
from echozero.application.sync.service import SyncService


def format_demo_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def nudge_selected_events(
    presentation: TimelinePresentation,
    *,
    direction: int,
    steps: int,
) -> TimelinePresentation:
    delta = 0.01 * max(1, steps) * (-1 if direction < 0 else 1)
    selected_ids = set(presentation.selected_event_ids)
    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        updated_events = []
        for event in layer.events:
            if event.event_id in selected_ids:
                start = max(0.0, event.start + delta)
                duration = max(event.duration, 0.01)
                updated_events.append(replace(event, start=start, end=start + duration))
            else:
                updated_events.append(event)
        layers.append(replace(layer, events=updated_events))
    return replace(presentation, layers=layers)


def duplicate_selected_events(
    presentation: TimelinePresentation,
    *,
    steps: int,
) -> TimelinePresentation:
    selected_ids = set(presentation.selected_event_ids)
    if not selected_ids:
        return presentation
    delta = 0.05 * max(1, steps)
    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        next_events = list(layer.events)
        for index, event in enumerate(layer.events, start=1):
            if event.event_id not in selected_ids:
                continue
            duration = max(event.duration, 0.01)
            clone_start = event.start + delta
            next_events.append(
                replace(
                    event,
                    event_id=EventId(f"{event.event_id}_dup_{index}_{steps}"),
                    start=clone_start,
                    end=clone_start + duration,
                    is_selected=False,
                )
            )
        next_events.sort(
            key=lambda candidate: (candidate.start, candidate.end, str(candidate.event_id))
        )
        layers.append(replace(layer, events=next_events))
    return replace(presentation, layers=layers)


def move_selected_events(
    presentation: TimelinePresentation,
    *,
    delta_seconds: float,
    target_layer_id,
) -> TimelinePresentation:
    selected_ids = set(presentation.selected_event_ids)
    if not selected_ids:
        return presentation

    source_layer_id = presentation.selected_layer_id
    updated_layers: list[LayerPresentation] = []
    moved_events: list[EventPresentation] = []

    for layer in presentation.layers:
        next_events: list[EventPresentation] = []
        for event in layer.events:
            if event.event_id not in selected_ids:
                next_events.append(event)
                continue
            duration = max(event.duration, 0.01)
            moved_events.append(
                replace(
                    event,
                    start=max(0.0, event.start + delta_seconds),
                    end=max(0.0, event.start + delta_seconds) + duration,
                    is_selected=True,
                )
            )
        if target_layer_id is not None and layer.layer_id == target_layer_id:
            next_events.extend(moved_events)
            next_events.sort(
                key=lambda candidate: (candidate.start, candidate.end, str(candidate.event_id))
            )
        elif target_layer_id is None and layer.layer_id == source_layer_id:
            next_events.extend(moved_events)
            next_events.sort(
                key=lambda candidate: (candidate.start, candidate.end, str(candidate.event_id))
            )
        updated_layers.append(replace(layer, events=next_events))

    selected_layer_id = target_layer_id if target_layer_id is not None else source_layer_id
    return replace(
        presentation,
        layers=updated_layers,
        selected_layer_id=selected_layer_id,
        selected_layer_ids=[] if selected_layer_id is None else [selected_layer_id],
    )


def open_push_workspace(
    presentation: TimelinePresentation,
    sync_service: SyncService,
    *,
    selected_event_ids: list[EventId],
) -> TimelinePresentation:
    selected_layer_ids = list(presentation.selected_layer_ids) or (
        [presentation.selected_layer_id] if presentation.selected_layer_id is not None else []
    )
    track_options = [
        ManualPushTrackOptionPresentation(
            coord=str(option.coord),
            name=str(option.name),
            note=option.note,
            event_count=option.event_count,
        )
        for option in sync_list(sync_service, "list_push_track_options")
    ]
    target_layer_id = presentation.selected_layer_id
    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        if layer.layer_id == target_layer_id:
            layers.append(
                replace(
                    layer,
                    push_selection_count=len(selected_event_ids),
                    push_row_status="blocked",
                    push_row_issue="Select an MA3 target track",
                )
            )
        else:
            layers.append(layer)
    batch_plan = BatchTransferPlanPresentation(
        plan_id="push:timeline_selection",
        operation_type="push",
        rows=(
            [
                BatchTransferPlanRowPresentation(
                    row_id=f"push:{target_layer_id}",
                    direction="push",
                    source_label=next(
                        (layer.title for layer in layers if layer.layer_id == target_layer_id),
                        "Selection",
                    ),
                    target_label="Unmapped",
                    source_layer_id=target_layer_id,
                    selected_event_ids=list(selected_event_ids),
                    selected_count=len(selected_event_ids),
                    status="blocked",
                    issue="Select an MA3 target track",
                )
            ]
            if target_layer_id is not None
            else []
        ),
        blocked_count=1 if target_layer_id is not None else 0,
    )
    return replace(
        presentation,
        layers=layers,
        manual_push_flow=ManualPushFlowPresentation(
            dialog_open=False,
            push_mode_active=True,
            selected_layer_ids=selected_layer_ids,
            available_tracks=track_options,
            transfer_mode="merge",
        ),
        batch_transfer_plan=batch_plan,
    )


def open_pull_workspace(
    presentation: TimelinePresentation,
    sync_service: SyncService,
) -> TimelinePresentation:
    track_options = [
        ManualPullTrackOptionPresentation(
            coord=str(option.coord),
            name=str(option.name),
            note=option.note,
            event_count=option.event_count,
        )
        for option in sync_list(sync_service, "list_pull_track_options")
    ]
    target_options = [
        ManualPullTargetOptionPresentation(layer_id=layer.layer_id, name=layer.title)
        for layer in presentation.layers
        if is_event_like_layer_kind(layer.kind)
    ]
    return replace(
        presentation,
        manual_pull_flow=ManualPullFlowPresentation(
            dialog_open=False,
            workspace_active=True,
            available_tracks=track_options,
            available_events=(
                [
                    ManualPullEventOptionPresentation(
                        event_id=str(option.event_id),
                        label=str(option.label),
                        start=option.start,
                        end=option.end,
                    )
                    for option in sync_list(
                        sync_service, "list_pull_source_events", "tc1_tg2_tr3"
                    )
                ]
                if track_options
                else []
            ),
            available_target_layers=target_options,
        ),
        batch_transfer_plan=BatchTransferPlanPresentation(
            plan_id="pull:timeline_selection",
            operation_type="pull",
        ),
    )


def sync_list(sync_service: SyncService, method_name: str, *args):
    method = getattr(sync_service, method_name, None)
    if not callable(method):
        return []
    return list(method(*args))


def apply_take_action(
    presentation: TimelinePresentation,
    layer_id,
    take_id,
    action_id: str,
) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    selected_lookup = set(presentation.selected_event_ids)
    for layer in presentation.layers:
        if layer.layer_id != layer_id:
            updated.append(layer)
            continue

        take = next((candidate for candidate in layer.takes if candidate.take_id == take_id), None)
        if take is None:
            updated.append(layer)
            continue

        next_events = list(layer.events)
        if action_id in {"overwrite_main", "promote_take"}:
            next_events = clone_events_for_main(take.events, suffix="ow")
        elif action_id == "merge_main":
            merged = list(layer.events)
            merged.extend(clone_events_for_main(take.events, suffix="mg"))
            next_events = sorted(merged, key=lambda e: (e.start, e.end))
        elif action_id == "add_selection_to_main":
            if presentation.selected_take_id != take_id:
                updated.append(layer)
                continue
            selected_events = [
                event for event in take.events if event.event_id in selected_lookup
            ]
            if not selected_events:
                updated.append(layer)
                continue
            merged = list(layer.events)
            merged.extend(clone_events_for_main(selected_events, suffix="sel"))
            next_events = sorted(merged, key=lambda e: (e.start, e.end))
        elif action_id == "delete_take":
            updated.append(
                replace(
                    layer,
                    takes=[candidate for candidate in layer.takes if candidate.take_id != take_id],
                    status=replace(layer.status, stale=False, manually_modified=True),
                )
            )
            continue

        updated.append(
            replace(
                layer,
                events=next_events,
                status=replace(layer.status, stale=False, manually_modified=True),
            )
        )
    return updated


def clone_events_for_main(
    events: list[EventPresentation], *, suffix: str
) -> list[EventPresentation]:
    clones: list[EventPresentation] = []
    for idx, event in enumerate(events, start=1):
        clones.append(
            replace(
                event,
                event_id=EventId(f"{event.event_id}_{suffix}_{idx}"),
                is_selected=False,
            )
        )
    return clones


def clear_selection(layers: list[LayerPresentation]) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        updated.append(
            replace(
                layer,
                is_selected=False,
                events=[replace(event, is_selected=False) for event in layer.events],
                takes=[
                    replace(
                        take, events=[replace(event, is_selected=False) for event in take.events]
                    )
                    for take in layer.takes
                ],
            )
        )
    return updated


def select_event(
    layers: list[LayerPresentation],
    *,
    layer_id,
    take_id,
    event_id,
) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        is_layer_selected = layer.layer_id == layer_id
        events = [
            replace(
                event,
                is_selected=is_layer_selected and take_id is None and event.event_id == event_id,
            )
            for event in layer.events
        ]
        takes = []
        for take in layer.takes:
            takes.append(
                replace(
                    take,
                    events=[
                        replace(
                            event,
                            is_selected=is_layer_selected
                            and take.take_id == take_id
                            and event.event_id == event_id,
                        )
                        for event in take.events
                    ],
                )
            )
        updated.append(replace(layer, is_selected=is_layer_selected, events=events, takes=takes))
    return updated


def set_selected_events(
    layers: list[LayerPresentation],
    *,
    selected_event_ids: list[EventId],
) -> list[LayerPresentation]:
    selected_lookup = set(selected_event_ids)
    updated_layers: list[LayerPresentation] = []
    for layer in layers:
        updated_layers.append(
            replace(
                layer,
                is_selected=any(event.event_id in selected_lookup for event in layer.events)
                or any(
                    event.event_id in selected_lookup
                    for take in layer.takes
                    for event in take.events
                ),
                events=[
                    replace(event, is_selected=event.event_id in selected_lookup)
                    for event in layer.events
                ],
                takes=[
                    replace(
                        take,
                        events=[
                            replace(event, is_selected=event.event_id in selected_lookup)
                            for event in take.events
                        ],
                    )
                    for take in layer.takes
                ],
            )
        )
    return updated_layers


def create_demo_event(
    presentation: TimelinePresentation,
    *,
    layer_id,
    take_id,
    start: float,
    end: float,
    label: str,
) -> TimelinePresentation:
    target_event_id = None
    target_take_id = take_id
    updated_layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        if layer.layer_id != layer_id:
            updated_layers.append(layer)
            continue

        target_take_id = take_id if take_id is not None else layer.main_take_id
        if take_id in (None, layer.main_take_id):
            target_event_id = next_demo_event_id(
                layer.main_take_id or EventId(f"{layer.layer_id}:main"), layer.events
            )
            created = EventPresentation(
                event_id=target_event_id,
                start=start,
                end=end,
                label=label,
                is_selected=True,
            )
            updated_layers.append(
                replace(
                    layer,
                    is_selected=True,
                    events=sorted(
                        [replace(event, is_selected=False) for event in layer.events] + [created],
                        key=lambda event: (event.start, event.end, str(event.event_id)),
                    ),
                    takes=[
                        replace(
                            take,
                            events=[replace(event, is_selected=False) for event in take.events],
                        )
                        for take in layer.takes
                    ],
                )
            )
            continue

        updated_takes: list[TakeLanePresentation] = []
        for take in layer.takes:
            if take.take_id != take_id:
                updated_takes.append(
                    replace(
                        take,
                        events=[replace(event, is_selected=False) for event in take.events],
                    )
                )
                continue
            target_event_id = next_demo_event_id(take.take_id, take.events)
            created = EventPresentation(
                event_id=target_event_id,
                start=start,
                end=end,
                label=label,
                is_selected=True,
            )
            updated_takes.append(
                replace(
                    take,
                    events=sorted(
                        [replace(event, is_selected=False) for event in take.events] + [created],
                        key=lambda event: (event.start, event.end, str(event.event_id)),
                    ),
                )
            )
        updated_layers.append(
            replace(
                layer,
                is_selected=True,
                events=[replace(event, is_selected=False) for event in layer.events],
                takes=updated_takes,
            )
        )

    return replace(
        presentation,
        layers=updated_layers,
        selected_layer_id=layer_id,
        selected_layer_ids=[layer_id],
        selected_take_id=target_take_id,
        selected_event_ids=[] if target_event_id is None else [target_event_id],
    )


def delete_demo_events(
    presentation: TimelinePresentation,
    *,
    event_ids: list[EventId],
) -> TimelinePresentation:
    delete_lookup = set(event_ids)
    layers = [
        replace(
            layer,
            events=[
                replace(event, is_selected=False)
                for event in layer.events
                if event.event_id not in delete_lookup
            ],
            takes=[
                replace(
                    take,
                    events=[
                        replace(event, is_selected=False)
                        for event in take.events
                        if event.event_id not in delete_lookup
                    ],
                )
                for take in layer.takes
            ],
        )
        for layer in presentation.layers
    ]
    return replace(
        presentation,
        layers=layers,
        selected_take_id=None,
        selected_event_ids=[],
    )


def next_demo_event_id(take_id, events: list[EventPresentation]) -> EventId:
    existing = {str(event.event_id) for event in events}
    index = 1
    while True:
        candidate = EventId(f"{take_id}:event:{index}")
        if str(candidate) not in existing:
            return candidate
        index += 1


__all__ = [
    "apply_take_action",
    "clear_selection",
    "create_demo_event",
    "delete_demo_events",
    "duplicate_selected_events",
    "format_demo_time",
    "move_selected_events",
    "next_demo_event_id",
    "nudge_selected_events",
    "open_pull_workspace",
    "open_push_workspace",
    "select_event",
    "set_selected_events",
    "sync_list",
]
