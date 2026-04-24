"""Event and take edit helpers for the timeline orchestrator.
Exists to keep create/delete/move/duplicate and take-action mutations out of selection-state helpers.
Connects selected event context to atomic event-edit and take-edit operations on the canonical timeline.
"""

from __future__ import annotations

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.models import Event, EventRef, Layer, Take, Timeline
from echozero.application.timeline.orchestrator_selection_state_mixin import (
    TimelineOrchestratorSelectionStateMixin,
)


class TimelineOrchestratorEventEditMixin(TimelineOrchestratorSelectionStateMixin):
    """Applies event and take mutations against the current canonical selection context."""

    def _handle_create_event(
        self,
        timeline: Timeline,
        *,
        layer_id: LayerId,
        take_id: TakeId | None,
        time_range: TimeRange,
        label: str,
        cue_number: int,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        if layer.kind is not LayerKind.EVENT or layer.presentation_hints.locked:
            return

        target_take = (
            self._find_take(layer, take_id) if take_id is not None else self._main_take(layer)
        )
        if target_take is None:
            target_take = self._resolve_or_create_main_take(layer)

        new_event = Event(
            id=self._next_created_event_id(timeline, target_take),
            take_id=target_take.id,
            start=float(time_range.start),
            end=float(time_range.end),
            cue_number=cue_number,
            label=label,
        )
        target_take.events = self._sorted_events([*target_take.events, new_event])
        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_layer_ids = [layer.id]
        timeline.selection.selected_take_id = target_take.id
        self._set_selected_event_refs(
            timeline, [self._event_ref(layer.id, target_take.id, new_event.id)]
        )

    def _handle_delete_events(
        self,
        timeline: Timeline,
        *,
        event_ids: list[EventId],
        event_refs: list[EventRef],
    ) -> None:
        delete_refs = (
            list(event_refs)
            if event_refs
            else self._resolve_event_refs_by_ids(
                timeline,
                event_ids,
                preferred_layer_ids=list(timeline.selection.selected_layer_ids),
                preferred_take_id=timeline.selection.selected_take_id,
            )
        )
        if not delete_refs and not event_ids:
            return
        delete_ref_keys = {self._event_ref_key(event_ref) for event_ref in delete_refs}
        delete_ids = set() if delete_ref_keys else set(event_ids)

        affected_layers: list[LayerId] = []
        for layer in timeline.layers:
            layer_changed = False
            for take in layer.takes:
                original_count = len(take.events)
                if original_count:
                    take.events = [
                        event
                        for event in take.events
                        if self._event_ref_key(self._event_ref(layer.id, take.id, event.id))
                        not in delete_ref_keys
                        and event.id not in delete_ids
                    ]
                if len(take.events) != original_count:
                    layer_changed = True
            if layer_changed:
                affected_layers.append(layer.id)

        if not affected_layers:
            return

        remaining_selected_refs = [
            event_ref
            for event_ref in self._selected_event_refs(timeline)
            if self._event_ref_key(event_ref) not in delete_ref_keys
            and event_ref.event_id not in delete_ids
        ]
        self._set_selected_event_refs(timeline, remaining_selected_refs)
        if remaining_selected_refs:
            records = self._selected_event_records(timeline, remaining_selected_refs)
            if records:
                last_record = records[-1]
                timeline.selection.selected_layer_id = last_record.layer.id
                timeline.selection.selected_layer_ids = [last_record.layer.id]
                timeline.selection.selected_take_id = last_record.take.id
                return

        fallback_layer_id = timeline.selection.selected_layer_id
        if fallback_layer_id not in {layer.id for layer in timeline.layers}:
            fallback_layer_id = affected_layers[0] if affected_layers else None
        timeline.selection.selected_layer_id = fallback_layer_id
        timeline.selection.selected_layer_ids = (
            [] if fallback_layer_id is None else [fallback_layer_id]
        )
        timeline.selection.selected_take_id = None

    def _handle_trigger_take_action(
        self,
        timeline: Timeline,
        layer_id: LayerId,
        take_id: TakeId,
        action_id: str,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        source_take = self._find_take(layer, take_id)
        main_take = self._main_take(layer)
        if source_take is None or main_take is None:
            return

        normalized = (action_id or "").strip().lower()
        if normalized in {"overwrite_main", "promote_take"}:
            if source_take.id == main_take.id:
                return
            main_take.events = self._clone_events_for_target(source_take.events, main_take)
        elif normalized == "merge_main":
            if source_take.id == main_take.id:
                return
            merged = list(main_take.events)
            merged.extend(self._clone_events_for_target(source_take.events, main_take))
            main_take.events = sorted(
                merged, key=lambda event: (event.start, event.end, str(event.id))
            )
        elif normalized == "add_selection_to_main":
            if source_take.id == main_take.id:
                return
            selected_source_events = [
                record.event
                for record in self._selected_event_records(
                    timeline, self._selected_event_refs(timeline)
                )
                if record.layer.id == layer.id and record.take.id == source_take.id
            ]
            if not selected_source_events:
                return
            cloned_events = self._clone_events_for_target(selected_source_events, main_take)
            main_take.events = self._sorted_events([*main_take.events, *cloned_events])
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_layer_ids = [layer.id]
            timeline.selection.selected_take_id = main_take.id
            self._set_selected_event_refs(
                timeline,
                [self._event_ref(layer.id, main_take.id, event.id) for event in cloned_events],
            )
            return
        elif normalized == "delete_take":
            if source_take.id == main_take.id:
                return
            layer.takes = [candidate for candidate in layer.takes if candidate.id != source_take.id]
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_layer_ids = [layer.id]
            timeline.selection.selected_take_id = main_take.id
            self._set_selected_event_refs(timeline, [])
            if (
                timeline.playback_target.layer_id == layer.id
                and timeline.playback_target.take_id == source_take.id
            ):
                timeline.playback_target.take_id = main_take.id
            return
        else:
            return

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_layer_ids = [layer.id]
        timeline.selection.selected_take_id = main_take.id
        self._set_selected_event_refs(timeline, [])

    def _handle_move_selected_events(
        self,
        timeline: Timeline,
        delta_seconds: float,
        target_layer_id: LayerId | None,
    ) -> None:
        selected_refs = self._selected_event_refs(timeline)
        if not selected_refs:
            return

        records = self._selected_event_records(timeline, selected_refs)
        if not records:
            self._set_selected_event_refs(timeline, [])
            timeline.selection.selected_take_id = None
            return

        applied_delta = max(float(delta_seconds), -min(record.event.start for record in records))
        source_layer_ids = {record.layer.id for record in records}
        source_layer_id = timeline.selection.selected_layer_id
        transfer_target: Layer | None = None

        if target_layer_id is not None and target_layer_id not in source_layer_ids:
            transfer_target = self._find_layer(timeline, target_layer_id)
            if (
                transfer_target.kind != records[0].layer.kind
                or transfer_target.kind.value != "event"
                or transfer_target.presentation_hints.locked
                or not transfer_target.presentation_hints.visible
            ):
                return

        affected_takes: dict[TakeId, Take] = {}
        if transfer_target is None:
            for record in records:
                record.event.start += applied_delta
                record.event.end += applied_delta
                affected_takes[record.take.id] = record.take
            self._sort_take_events(*affected_takes.values())
            if source_layer_id is None and records:
                source_layer_id = records[0].layer.id
            timeline.selection.selected_layer_id = source_layer_id
            timeline.selection.selected_layer_ids = (
                [] if source_layer_id is None else [source_layer_id]
            )
            timeline.selection.selected_take_id = self._resolve_selected_take_id(
                (
                    self._find_layer(timeline, source_layer_id)
                    if source_layer_id is not None
                    else records[0].layer
                ),
                [self._event_ref(record.layer.id, record.take.id, record.event.id) for record in records],
                fallback_take_id=timeline.selection.selected_take_id,
            )
            self._set_selected_event_refs(
                timeline,
                [self._event_ref(record.layer.id, record.take.id, record.event.id) for record in records],
            )
            return

        target_take = self._main_take(transfer_target)
        if target_take is None:
            return

        for record in records:
            record.take.events = [
                candidate for candidate in record.take.events if candidate.id != record.event.id
            ]
            affected_takes[record.take.id] = record.take
            record.event.start += applied_delta
            record.event.end += applied_delta
            record.event.take_id = target_take.id
            target_take.events.append(record.event)

        affected_takes[target_take.id] = target_take
        self._sort_take_events(*affected_takes.values())
        timeline.selection.selected_layer_id = transfer_target.id
        timeline.selection.selected_layer_ids = [transfer_target.id]
        timeline.selection.selected_take_id = target_take.id
        self._set_selected_event_refs(
            timeline,
            [
                self._event_ref(transfer_target.id, target_take.id, record.event.id)
                for record in records
            ],
        )

    def _handle_move_selected_events_to_adjacent_layer(
        self,
        timeline: Timeline,
        *,
        direction: int,
    ) -> None:
        step = 1 if direction > 0 else -1 if direction < 0 else 0
        if step == 0:
            return

        records = self._selected_event_records(timeline, self._selected_event_refs(timeline))
        if not records:
            return

        source_layer_ids = list(dict.fromkeys(record.layer.id for record in records))
        if len(source_layer_ids) != 1:
            return

        source_layer = self._find_layer(timeline, source_layer_ids[0])
        movable_layers = [
            layer
            for layer in sorted(timeline.layers, key=lambda value: value.order_index)
            if (
                layer.kind == source_layer.kind
                and layer.presentation_hints.visible
                and not layer.presentation_hints.locked
            )
        ]
        movable_ids = [layer.id for layer in movable_layers]
        if source_layer.id not in movable_ids:
            return

        source_index = movable_ids.index(source_layer.id)
        target_index = source_index + step
        if target_index < 0 or target_index >= len(movable_ids):
            return

        self._handle_move_selected_events(
            timeline,
            delta_seconds=0.0,
            target_layer_id=movable_ids[target_index],
        )

    @staticmethod
    def _sort_take_events(*takes: Take) -> None:
        for take in takes:
            take.events = sorted(
                take.events, key=lambda event: (event.start, event.end, str(event.id))
            )

    def _handle_nudge_selected_events(
        self,
        timeline: Timeline,
        direction: int,
        steps: int,
    ) -> None:
        if direction == 0 or steps <= 0:
            return

        selected = self._selected_events(timeline)
        if not selected:
            return

        delta = float(direction) * float(steps) * (1.0 / 30.0)
        for _layer, take, event in selected:
            duration = event.duration
            next_start = max(0.0, event.start + delta)
            event.start = next_start
            event.end = next_start + duration
            take.events = self._sorted_events(take.events)

    def _handle_duplicate_selected_events(self, timeline: Timeline, steps: int) -> None:
        if steps <= 0:
            return

        selected = self._selected_events(timeline)
        if not selected:
            return

        delta = float(steps) * (1.0 / 30.0)
        existing_ids = self._all_event_ids(timeline)
        duplicated_ids: list[EventId] = []
        selected_layer_id = timeline.selection.selected_layer_id
        selected_take_id = timeline.selection.selected_take_id

        for layer, take, event in selected:
            duplicate_id = self._next_duplicate_event_id(take, event, existing_ids)
            duplicate = Event(
                id=duplicate_id,
                take_id=take.id,
                start=event.start + delta,
                end=event.end + delta,
                cue_number=event.cue_number,
                source_event_id=event.source_event_id,
                parent_event_id=str(event.id),
                payload_ref=event.payload_ref,
                label=event.label,
                color=event.color,
                muted=event.muted,
            )
            take.events = self._sorted_events([*take.events, duplicate])
            existing_ids.add(str(duplicate.id))
            duplicated_ids.append(duplicate.id)
            selected_layer_id = layer.id
            selected_take_id = take.id

        timeline.selection.selected_layer_id = selected_layer_id
        timeline.selection.selected_layer_ids = (
            [] if selected_layer_id is None else [selected_layer_id]
        )
        timeline.selection.selected_take_id = selected_take_id
        self._set_selected_event_refs(
            timeline,
            []
            if selected_layer_id is None or selected_take_id is None
            else [
                self._event_ref(selected_layer_id, selected_take_id, duplicated_id)
                for duplicated_id in duplicated_ids
            ],
        )

    @staticmethod
    def _clone_events_for_target(events: list[Event], target_take: Take) -> list[Event]:
        clones: list[Event] = []
        for idx, event in enumerate(events, start=1):
            clones.append(
                Event(
                    id=EventId(f"{target_take.id}:from:{event.id}:{idx}"),
                    take_id=target_take.id,
                    start=event.start,
                    end=event.end,
                    cue_number=event.cue_number,
                    source_event_id=event.source_event_id,
                    parent_event_id=str(event.id),
                    payload_ref=event.payload_ref,
                    label=event.label,
                    color=event.color,
                    muted=event.muted,
                )
            )
        return clones

    @staticmethod
    def _sorted_events(events: list[Event]) -> list[Event]:
        return sorted(events, key=lambda event: (event.start, event.end, str(event.id)))

    @staticmethod
    def _all_event_ids(timeline: Timeline) -> set[str]:
        return {
            str(event.id)
            for layer in timeline.layers
            for take in layer.takes
            for event in take.events
        }

    @staticmethod
    def _next_duplicate_event_id(take: Take, event: Event, existing_ids: set[str]) -> EventId:
        index = 1
        while True:
            candidate = f"{take.id}:dup:{event.id}:{index}"
            if candidate not in existing_ids:
                return EventId(candidate)
            index += 1

    @staticmethod
    def _next_created_event_id(timeline: Timeline, take: Take) -> EventId:
        existing_ids = TimelineOrchestratorEventEditMixin._all_event_ids(timeline)
        index = 1
        while True:
            candidate = f"{take.id}:event:{index}"
            if candidate not in existing_ids:
                return EventId(candidate)
            index += 1
