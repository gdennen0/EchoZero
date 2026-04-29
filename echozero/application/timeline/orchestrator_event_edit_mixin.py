"""Event and take edit helpers for the timeline orchestrator.
Exists to keep create/delete/move/duplicate and take-action mutations out of selection-state helpers.
Connects selected event context to atomic event-edit and take-edit operations on the canonical timeline.
"""

from __future__ import annotations

from copy import deepcopy

from echozero.application.shared.cue_numbers import CueNumber, cue_number_text
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.models import (
    Event,
    EventRef,
    Layer,
    Take,
    Timeline,
    cue_number_from_ref,
)
from echozero.application.timeline.orchestrator_selection_state_mixin import (
    TimelineOrchestratorSelectionStateMixin,
)
from echozero.application.timeline.intents import ReplaceSectionCues, SectionCueEdit, TrimEvent, UpdateEventLabel


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
        cue_number: CueNumber,
        source_event_id: str | None = None,
        payload_ref: str | None = None,
        color: str | None = None,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        if not is_event_like_layer_kind(layer.kind) or layer.presentation_hints.locked:
            return

        if layer.kind is LayerKind.SECTION:
            target_take = self._resolve_or_create_main_take(layer)
            section_start = float(time_range.start)
            section_cue_number = self._next_section_cue_number(target_take)
            cue_suffix = cue_number_text(section_cue_number) or str(section_cue_number)
            normalized_label = str(label or "").strip()
            section_label = (
                f"Section {cue_suffix}"
                if not normalized_label or normalized_label.casefold() == "event"
                else normalized_label
            )
            new_event = Event(
                id=self._next_created_event_id(timeline, target_take),
                take_id=target_take.id,
                start=section_start,
                end=section_start + 0.08,
                origin="manual_added",
                classifications={"label": section_label} if section_label else {},
                metadata={},
                cue_number=section_cue_number,
                label=section_label,
                cue_ref=f"Q{cue_suffix}",
                source_event_id=source_event_id,
                payload_ref=payload_ref,
                color=color,
            )
            target_take.events = self._sorted_events([*target_take.events, new_event])
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_layer_ids = [layer.id]
            timeline.selection.selected_take_id = target_take.id
            self._set_selected_event_refs(
                timeline, [self._event_ref(layer.id, target_take.id, new_event.id)]
            )
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
            origin="manual_added",
            classifications={"label": label} if label else {},
            metadata={},
            cue_number=cue_number,
            label=label,
            source_event_id=source_event_id,
            payload_ref=payload_ref,
            color=color,
        )
        target_take.events = self._sorted_events([*target_take.events, new_event])
        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_layer_ids = [layer.id]
        timeline.selection.selected_take_id = target_take.id
        self._set_selected_event_refs(
            timeline, [self._event_ref(layer.id, target_take.id, new_event.id)]
        )

    @staticmethod
    def _next_section_cue_number(take: Take) -> CueNumber:
        if not take.events:
            return 1
        highest_existing = max(float(event.cue_number) for event in take.events)
        return max(1, int(highest_existing) + 1)

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

    def _handle_trim_event(
        self,
        timeline: Timeline,
        *,
        event_id: EventId,
        new_range: TimeRange,
    ) -> None:
        records = self._selected_event_records(
            timeline,
            self._resolve_event_refs_by_ids(
                timeline,
                [event_id],
                preferred_layer_ids=list(timeline.selection.selected_layer_ids),
                preferred_take_id=timeline.selection.selected_take_id,
            ),
        )
        if not records:
            return

        for record in records:
            if record.event.id != event_id:
                continue
            record.event.start = float(new_range.start)
            record.event.end = float(new_range.end)
            record.take.events = self._sorted_events(record.take.events)
            timeline.selection.selected_layer_id = record.layer.id
            timeline.selection.selected_layer_ids = [record.layer.id]
            timeline.selection.selected_take_id = record.take.id
            self._set_selected_event_refs(
                timeline,
                [self._event_ref(record.layer.id, record.take.id, record.event.id)],
            )
            return

    def _handle_update_event_label(
        self,
        timeline: Timeline,
        *,
        event_id: EventId,
        label: str,
        layer_id: LayerId | None = None,
        take_id: TakeId | None = None,
    ) -> None:
        preferred_layer_ids = [layer_id] if layer_id is not None else list(
            timeline.selection.selected_layer_ids
        )
        preferred_take_id = take_id if take_id is not None else timeline.selection.selected_take_id
        records = self._selected_event_records(
            timeline,
            self._resolve_event_refs_by_ids(
                timeline,
                [event_id],
                preferred_layer_ids=preferred_layer_ids,
                preferred_take_id=preferred_take_id,
            ),
        )
        if not records:
            return

        for record in records:
            if record.event.id != event_id:
                continue
            record.event.label = label
            timeline.selection.selected_layer_id = record.layer.id
            timeline.selection.selected_layer_ids = [record.layer.id]
            timeline.selection.selected_take_id = record.take.id
            self._set_selected_event_refs(
                timeline,
                [self._event_ref(record.layer.id, record.take.id, record.event.id)],
            )
            return

    def _handle_replace_section_cues(
        self,
        timeline: Timeline,
        *,
        cues: list[SectionCueEdit],
    ) -> None:
        section_layer = self._section_layer(timeline)
        if section_layer is None and not cues:
            return
        if section_layer is None:
            section_layer = self._create_section_layer(timeline)
        if section_layer.presentation_hints.locked:
            return

        main_take = self._resolve_or_create_main_take(section_layer)
        existing_by_id = {str(event.id): event for event in main_take.events}
        replacement_events: list[Event] = []
        for cue in sorted(cues, key=lambda value: (float(value.start), str(value.cue_id or ""))):
            existing = (
                existing_by_id.get(str(cue.cue_id))
                if cue.cue_id is not None
                else None
            )
            event_id = existing.id if existing is not None else self._next_created_event_id(
                timeline,
                main_take,
            )
            payload_ref = cue.payload_ref or (existing.payload_ref if existing is not None else None)
            replacement_events.append(
                Event(
                    id=event_id,
                    take_id=main_take.id,
                    start=float(cue.start),
                    end=float(cue.start) + 0.08,
                    origin=existing.origin if existing is not None else "manual_added",
                    classifications={"label": cue.name} if cue.name else {},
                    metadata=deepcopy(existing.metadata) if existing is not None else {},
                    cue_number=cue_number_from_ref(
                        cue.cue_ref,
                        fallback=existing.cue_number if existing is not None else 1,
                    ),
                    source_event_id=existing.source_event_id if existing is not None else None,
                    parent_event_id=existing.parent_event_id if existing is not None else None,
                    payload_ref=payload_ref,
                    label=cue.name,
                    cue_ref=cue.cue_ref,
                    color=cue.color,
                    notes=cue.notes,
                    muted=existing.muted if existing is not None else False,
                )
            )

        main_take.events = self._sorted_events(replacement_events)
        timeline.selection.selected_layer_id = section_layer.id
        timeline.selection.selected_layer_ids = [section_layer.id]
        timeline.selection.selected_take_id = main_take.id
        self._set_selected_event_refs(timeline, [])

    @staticmethod
    def _section_layer(timeline: Timeline) -> Layer | None:
        section_layers = sorted(
            (layer for layer in timeline.layers if layer.kind is LayerKind.SECTION),
            key=lambda layer: (int(layer.order_index), str(layer.id)),
        )
        return section_layers[0] if section_layers else None

    @staticmethod
    def _next_created_section_layer_id(timeline: Timeline) -> LayerId:
        existing_ids = {str(layer.id) for layer in timeline.layers}
        if "layer_sections" not in existing_ids:
            return LayerId("layer_sections")
        index = 2
        while True:
            candidate = f"layer_sections_{index}"
            if candidate not in existing_ids:
                return LayerId(candidate)
            index += 1

    def _create_section_layer(self, timeline: Timeline) -> Layer:
        source_present = any(layer.id == LayerId("source_audio") for layer in timeline.layers)
        insert_order = 1 if source_present else 0
        for layer in timeline.layers:
            if int(layer.order_index) >= insert_order:
                layer.order_index += 1
        section_layer = Layer(
            id=self._next_created_section_layer_id(timeline),
            timeline_id=timeline.id,
            name="Sections",
            kind=LayerKind.SECTION,
            order_index=insert_order,
        )
        timeline.layers.append(section_layer)
        timeline.layers = sorted(timeline.layers, key=lambda layer: (int(layer.order_index), str(layer.id)))
        return section_layer

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
        copy_selected: bool = False,
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
                or not is_event_like_layer_kind(transfer_target.kind)
                or transfer_target.presentation_hints.locked
                or not transfer_target.presentation_hints.visible
            ):
                return

        if copy_selected:
            existing_ids = self._all_event_ids(timeline)
            copied_refs: list[EventRef] = []
            if transfer_target is None:
                affected_takes: dict[TakeId, Take] = {}
                for record in records:
                    duplicate_id = self._next_duplicate_event_id(
                        record.take,
                        record.event,
                        existing_ids,
                    )
                    duplicate = self._duplicate_event(
                        record.event,
                        duplicate_id=duplicate_id,
                        target_take_id=record.take.id,
                        start=record.event.start + applied_delta,
                        end=record.event.end + applied_delta,
                    )
                    record.take.events.append(duplicate)
                    affected_takes[record.take.id] = record.take
                    existing_ids.add(str(duplicate.id))
                    copied_refs.append(
                        self._event_ref(record.layer.id, record.take.id, duplicate.id)
                    )

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
                    copied_refs,
                    fallback_take_id=timeline.selection.selected_take_id,
                )
                self._set_selected_event_refs(timeline, copied_refs)
                return

            target_take = self._main_take(transfer_target)
            if target_take is None:
                return

            for record in records:
                duplicate_id = self._next_duplicate_event_id(
                    target_take,
                    record.event,
                    existing_ids,
                )
                duplicate = self._duplicate_event(
                    record.event,
                    duplicate_id=duplicate_id,
                    target_take_id=target_take.id,
                    start=record.event.start + applied_delta,
                    end=record.event.end + applied_delta,
                )
                target_take.events.append(duplicate)
                existing_ids.add(str(duplicate.id))
                copied_refs.append(self._event_ref(transfer_target.id, target_take.id, duplicate.id))

            self._sort_take_events(target_take)
            timeline.selection.selected_layer_id = transfer_target.id
            timeline.selection.selected_layer_ids = [transfer_target.id]
            timeline.selection.selected_take_id = target_take.id
            self._set_selected_event_refs(timeline, copied_refs)
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

    def _handle_reorder_layer(
        self,
        timeline: Timeline,
        *,
        source_layer_id: LayerId,
        target_after_layer_id: LayerId | None,
        insert_at_start: bool,
    ) -> None:
        if str(source_layer_id) == "source_audio":
            return

        ordered_layers = sorted(timeline.layers, key=lambda layer: layer.order_index)
        source_index = next(
            (index for index, layer in enumerate(ordered_layers) if layer.id == source_layer_id),
            None,
        )
        if source_index is None:
            return

        source_layer = ordered_layers[source_index]
        remaining_layers = [
            layer for layer in ordered_layers if layer.id != source_layer_id
        ]

        normalized_insert_at_start = bool(insert_at_start)
        target_after = target_after_layer_id
        if normalized_insert_at_start:
            target_after = None

        if target_after is None:
            if normalized_insert_at_start:
                next_layers = [source_layer, *remaining_layers]
            else:
                next_layers = [*remaining_layers, source_layer]
        else:
            if target_after == source_layer.id:
                return
            target_index = next(
                (index for index, layer in enumerate(remaining_layers) if layer.id == target_after),
                None,
            )
            if target_index is None:
                return
            next_layers = list(remaining_layers)
            next_layers.insert(target_index + 1, source_layer)

        source_audio_layer = next(
            (layer for layer in next_layers if str(layer.id) == "source_audio"),
            None,
        )
        if source_audio_layer is not None and next_layers:
            if next_layers[0].id != source_audio_layer.id:
                next_layers = [
                    source_audio_layer,
                    *(
                        layer
                        for layer in next_layers
                        if layer.id != source_audio_layer.id
                    ),
                ]

        if tuple(layer.id for layer in ordered_layers) == tuple(layer.id for layer in next_layers):
            return

        for index, layer in enumerate(next_layers):
            layer.order_index = index
        timeline.layers = next_layers

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
            duplicate = self._duplicate_event(
                event,
                duplicate_id=duplicate_id,
                target_take_id=take.id,
                start=event.start + delta,
                end=event.end + delta,
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
                TimelineOrchestratorEventEditMixin._duplicate_event(
                    event,
                    duplicate_id=EventId(f"{target_take.id}:from:{event.id}:{idx}"),
                    target_take_id=target_take.id,
                    start=event.start,
                    end=event.end,
                )
            )
        return clones

    @staticmethod
    def _duplicate_event(
        event: Event,
        *,
        duplicate_id: EventId,
        target_take_id: TakeId,
        start: float,
        end: float,
    ) -> Event:
        return Event(
            id=duplicate_id,
            take_id=target_take_id,
            start=start,
            end=end,
            origin=event.origin,
            classifications=dict(event.classifications),
            metadata=dict(event.metadata),
            cue_number=event.cue_number,
            source_event_id=event.source_event_id,
            parent_event_id=str(event.id),
            payload_ref=event.payload_ref,
            label=event.label,
            cue_ref=event.cue_ref,
            color=event.color,
            notes=event.notes,
            muted=event.muted,
        )

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
