"""Selection-state helpers for the timeline orchestrator.
Exists to keep layer/take/event selection and ref resolution out of event-edit mutation flows.
Connects timeline truth to deterministic selected-layer, selected-take, and event-ref state.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.models import Event, EventRef, Layer, Take, Timeline


@dataclass(slots=True)
class SelectedEventRecord:
    """Resolved selection record spanning the owning layer, take, and event."""

    layer: Layer
    take: Take
    event: Event


class TimelineOrchestratorSelectionStateMixin:
    """Owns canonical layer/take/event selection state and ref-resolution helpers."""

    def _handle_select_layer(
        self,
        timeline: Timeline,
        layer_id: LayerId | None,
        *,
        mode: str,
    ) -> None:
        if layer_id is None:
            timeline.selection.selected_layer_id = None
            timeline.selection.selected_layer_ids = []
            timeline.selection.selected_take_id = None
            self._set_selected_event_refs(timeline, [])
            return

        self._find_layer(timeline, layer_id)
        mode_normalized = (mode or "replace").strip().lower()
        current_ids = list(dict.fromkeys(timeline.selection.selected_layer_ids))
        if not current_ids and timeline.selection.selected_layer_id is not None:
            current_ids = [timeline.selection.selected_layer_id]

        if mode_normalized == "replace":
            next_ids = [layer_id]
        elif mode_normalized == "toggle":
            if layer_id in current_ids:
                next_ids = [candidate for candidate in current_ids if candidate != layer_id]
            else:
                next_ids = [*current_ids, layer_id]
        elif mode_normalized == "range":
            ordered_ids = [
                layer.id for layer in sorted(timeline.layers, key=lambda value: value.order_index)
            ]
            anchor_id = current_ids[0] if current_ids else timeline.selection.selected_layer_id
            if anchor_id is None or anchor_id not in ordered_ids:
                anchor_id = layer_id
            start = ordered_ids.index(anchor_id)
            end = ordered_ids.index(layer_id)
            low, high = sorted((start, end))
            next_ids = ordered_ids[low : high + 1]
        else:
            raise ValueError(f"Unsupported layer selection mode: {mode}")

        timeline.selection.selected_layer_id = layer_id if next_ids else None
        timeline.selection.selected_layer_ids = next_ids
        timeline.selection.selected_take_id = None
        self._set_selected_event_refs(timeline, [])

    def _handle_select_adjacent_layer(self, timeline: Timeline, *, direction: int) -> None:
        step = 1 if direction > 0 else -1 if direction < 0 else 0
        if step == 0:
            return

        ordered_layers = self._ordered_visible_layers(timeline)
        if not ordered_layers:
            return

        ordered_ids = [layer.id for layer in ordered_layers]
        current_layer_id = self._navigation_layer_id(timeline)
        if current_layer_id not in ordered_ids:
            target_id = ordered_ids[0] if step > 0 else ordered_ids[-1]
            self._handle_select_layer(timeline, target_id, mode="replace")
            return

        current_index = ordered_ids.index(current_layer_id)
        target_index = current_index + step
        if target_index < 0 or target_index >= len(ordered_ids):
            return
        self._handle_select_layer(timeline, ordered_ids[target_index], mode="replace")

    def _handle_select_take(
        self,
        timeline: Timeline,
        layer_id: LayerId,
        take_id: TakeId | None,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        if take_id is not None:
            self._find_take(layer, take_id)
        timeline.selection.selected_layer_id = layer_id
        timeline.selection.selected_layer_ids = [layer_id]
        timeline.selection.selected_take_id = take_id
        self._set_selected_event_refs(timeline, [])

    def _handle_set_active_playback_target(
        self,
        timeline: Timeline,
        layer_id: LayerId | None,
        take_id: TakeId | None,
    ) -> None:
        if layer_id is None:
            timeline.playback_target.layer_id = None
            timeline.playback_target.take_id = None
            return

        layer = self._find_layer(timeline, layer_id)
        if take_id is not None:
            self._find_take(layer, take_id)
        timeline.playback_target.layer_id = layer.id
        timeline.playback_target.take_id = take_id

    @staticmethod
    def _event_ref(layer_id: LayerId, take_id: TakeId, event_id: EventId) -> EventRef:
        return EventRef(layer_id=layer_id, take_id=take_id, event_id=event_id)

    @staticmethod
    def _event_ref_key(event_ref: EventRef) -> tuple[str, str, str]:
        return (
            str(event_ref.layer_id),
            str(event_ref.take_id),
            str(event_ref.event_id),
        )

    def _set_selected_event_refs(self, timeline: Timeline, event_refs: list[EventRef]) -> None:
        normalized_refs: list[EventRef] = []
        seen: set[tuple[str, str, str]] = set()
        for event_ref in event_refs:
            key = self._event_ref_key(event_ref)
            if key in seen:
                continue
            normalized_refs.append(event_ref)
            seen.add(key)
        timeline.selection.selected_event_refs = normalized_refs
        timeline.selection.selected_event_ids = [event_ref.event_id for event_ref in normalized_refs]

    def _resolve_event_refs_by_ids(
        self,
        timeline: Timeline,
        event_ids: list[EventId],
        *,
        preferred_layer_ids: list[LayerId] | None = None,
        preferred_take_id: TakeId | None = None,
    ) -> list[EventRef]:
        if not event_ids:
            return []

        preferred_layers = {layer_id for layer_id in (preferred_layer_ids or []) if layer_id is not None}
        event_records: dict[str, list[EventRef]] = {}
        for layer in timeline.layers:
            for take in layer.takes:
                for event in take.events:
                    event_records.setdefault(str(event.id), []).append(
                        self._event_ref(layer.id, take.id, event.id)
                    )

        resolved: list[EventRef] = []
        seen: set[tuple[str, str, str]] = set()
        for event_id in event_ids:
            matches = list(event_records.get(str(event_id), []))
            if not matches:
                continue

            preferred_matches = [
                match
                for match in matches
                if (
                    (preferred_take_id is None or match.take_id == preferred_take_id)
                    and (not preferred_layers or match.layer_id in preferred_layers)
                )
            ]
            if preferred_matches:
                matches = preferred_matches[:1]
            elif len(matches) > 1:
                take_matches = [
                    match
                    for match in matches
                    if preferred_take_id is not None and match.take_id == preferred_take_id
                ]
                if take_matches:
                    matches = take_matches[:1]

            for match in matches:
                key = self._event_ref_key(match)
                if key in seen:
                    continue
                resolved.append(match)
                seen.add(key)
        return resolved

    def _selected_event_refs(self, timeline: Timeline) -> list[EventRef]:
        if (
            timeline.selection.selected_event_refs
            and [event_ref.event_id for event_ref in timeline.selection.selected_event_refs]
            == list(timeline.selection.selected_event_ids)
        ):
            return list(timeline.selection.selected_event_refs)
        return self._resolve_event_refs_by_ids(
            timeline,
            list(timeline.selection.selected_event_ids),
            preferred_layer_ids=list(timeline.selection.selected_layer_ids),
            preferred_take_id=timeline.selection.selected_take_id,
        )

    def _handle_select_event(
        self,
        timeline: Timeline,
        layer_id: LayerId,
        take_id: TakeId | None,
        event_id: EventId | None,
        mode: str,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        target_take = self._find_take(layer, take_id) if take_id is not None else self._main_take(layer)
        target_take_id = target_take.id if target_take is not None else take_id
        if event_id is None:
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_layer_ids = [layer.id]
            timeline.selection.selected_take_id = target_take_id
            self._set_selected_event_refs(timeline, [])
            return
        if target_take is None or all(candidate.id != event_id for candidate in target_take.events):
            return

        mode_normalized = (mode or "replace").strip().lower()
        selected_refs = list(self._selected_event_refs(timeline))
        assert target_take_id is not None
        target_ref = self._event_ref(layer.id, target_take_id, event_id)

        if mode_normalized == "replace":
            selected_refs = [target_ref]
        elif mode_normalized == "additive":
            if target_ref not in selected_refs:
                selected_refs.append(target_ref)
        elif mode_normalized == "toggle":
            if target_ref in selected_refs:
                selected_refs = [
                    selected_ref for selected_ref in selected_refs if selected_ref != target_ref
                ]
            else:
                selected_refs.append(target_ref)
        else:
            raise ValueError(f"Unsupported selection mode: {mode}")

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_layer_ids = [layer.id]
        self._set_selected_event_refs(timeline, selected_refs)
        timeline.selection.selected_take_id = target_take_id if selected_refs else None

    def _handle_select_adjacent_event_in_selected_layer(
        self,
        timeline: Timeline,
        *,
        direction: int,
        include_demoted: bool = False,
    ) -> None:
        step = 1 if direction > 0 else -1 if direction < 0 else 0
        if step == 0:
            return

        layer_id = self._navigation_layer_id(timeline)
        if layer_id is None:
            return

        layer = self._find_layer(timeline, layer_id)
        take = self._navigation_take_for_layer(timeline, layer)
        if take is None or not take.events:
            return

        selected_refs = [
            event_ref
            for event_ref in self._selected_event_refs(timeline)
            if event_ref.layer_id == layer.id and event_ref.take_id == take.id
        ]
        if not selected_refs and timeline.selection.selected_event_ids:
            take_event_ids = {event.id for event in take.events}
            selected_refs = [
                self._event_ref(layer.id, take.id, event_id)
                for event_id in timeline.selection.selected_event_ids
                if event_id in take_event_ids
            ]
        playhead_seconds = None
        if not selected_refs:
            playhead_seconds = float(self.transport_service.get_state().playhead)
        target_event = self._adjacent_event_for_selection(
            self._ordered_events(take),
            selected_refs,
            direction=step,
            playhead_seconds=playhead_seconds,
            include_demoted=include_demoted,
        )
        if target_event is None:
            return

        self._handle_select_event(
            timeline,
            layer_id=layer.id,
            take_id=take.id,
            event_id=target_event.id,
            mode="replace",
        )

    def _handle_select_all_events(self, timeline: Timeline) -> None:
        selected_layer_ids = self._selected_layer_scope(timeline)
        target_layers: list[Layer]
        if selected_layer_ids:
            target_layers = [
                self._find_layer(timeline, layer_id) for layer_id in selected_layer_ids
            ]
        else:
            target_layers = [
                layer
                for layer in timeline.layers
                if layer.presentation_hints.visible and not layer.presentation_hints.locked
            ]

        selected_event_refs: list[EventRef] = []
        selected_take_id: TakeId | None = None
        for layer in target_layers:
            if not layer.presentation_hints.visible or layer.presentation_hints.locked:
                continue
            for take in layer.takes:
                if take.events and selected_take_id is None:
                    selected_take_id = take.id
                selected_event_refs.extend(
                    self._event_ref(layer.id, take.id, event.id) for event in take.events
                )

        self._set_selected_event_refs(timeline, selected_event_refs)
        timeline.selection.selected_take_id = selected_take_id

    def _handle_set_selected_events(
        self,
        timeline: Timeline,
        *,
        event_ids: list[EventId],
        event_refs: list[EventRef],
        anchor_layer_id: LayerId | None,
        anchor_take_id: TakeId | None,
        selected_layer_ids: list[LayerId],
    ) -> None:
        normalized_event_ids = list(dict.fromkeys(event_ids))
        normalized_event_refs = (
            list(dict.fromkeys(event_refs))
            if event_refs
            else self._resolve_event_refs_by_ids(
                timeline,
                normalized_event_ids,
                preferred_layer_ids=selected_layer_ids,
                preferred_take_id=anchor_take_id,
            )
        )
        normalized_layer_ids = list(dict.fromkeys(selected_layer_ids))

        if not normalized_event_refs and not normalized_event_ids:
            self._set_selected_event_refs(timeline, [])
            timeline.selection.selected_take_id = None
            timeline.selection.selected_layer_ids = (
                normalized_layer_ids
                if normalized_layer_ids
                else ([anchor_layer_id] if anchor_layer_id is not None else [])
            )
            timeline.selection.selected_layer_id = (
                anchor_layer_id if timeline.selection.selected_layer_ids else None
            )
            return

        if anchor_layer_id is None or anchor_take_id is None:
            selected_items = normalized_event_refs if normalized_event_refs else normalized_event_ids
            records = self._selected_event_records(timeline, selected_items)
            if records:
                last_record = records[-1]
                anchor_layer_id = last_record.layer.id
                anchor_take_id = last_record.take.id
                if not normalized_layer_ids:
                    normalized_layer_ids = list(
                        dict.fromkeys(record.layer.id for record in records)
                    )

        timeline.selection.selected_layer_id = anchor_layer_id
        timeline.selection.selected_layer_ids = (
            normalized_layer_ids
            if normalized_layer_ids
            else ([anchor_layer_id] if anchor_layer_id is not None else [])
        )
        timeline.selection.selected_take_id = anchor_take_id
        self._set_selected_event_refs(
            timeline,
            normalized_event_refs
            if normalized_event_refs
            else self._resolve_event_refs_by_ids(
                timeline,
                normalized_event_ids,
                preferred_layer_ids=timeline.selection.selected_layer_ids,
                preferred_take_id=anchor_take_id,
            ),
        )

    def _selected_event_records(
        self,
        timeline: Timeline,
        selected_items: list[EventRef] | list[EventId],
    ) -> list[SelectedEventRecord]:
        if selected_items and isinstance(selected_items[0], EventRef):
            selected_refs = list(selected_items)
        else:
            selected_refs = self._resolve_event_refs_by_ids(
                timeline,
                list(selected_items),
                preferred_layer_ids=list(timeline.selection.selected_layer_ids),
                preferred_take_id=timeline.selection.selected_take_id,
            )
        selected_lookup = {self._event_ref_key(event_ref) for event_ref in selected_refs}
        order = {
            self._event_ref_key(event_ref): idx for idx, event_ref in enumerate(selected_refs)
        }
        records: list[SelectedEventRecord] = []
        for layer in timeline.layers:
            for take in layer.takes:
                for event in take.events:
                    event_ref = self._event_ref(layer.id, take.id, event.id)
                    key = self._event_ref_key(event_ref)
                    if key in selected_lookup:
                        records.append(SelectedEventRecord(layer=layer, take=take, event=event))
        records.sort(
            key=lambda record: order.get(
                self._event_ref_key(
                    self._event_ref(record.layer.id, record.take.id, record.event.id)
                ),
                len(order),
            )
        )
        return records

    @staticmethod
    def _resolve_selected_take_id(
        layer: Layer,
        selected_refs: list[EventRef],
        fallback_take_id: TakeId | None = None,
    ) -> TakeId | None:
        selected_lookup = {
            (str(event_ref.take_id), str(event_ref.event_id)) for event_ref in selected_refs
        }
        if fallback_take_id is not None:
            fallback = TimelineOrchestratorSelectionStateMixin._find_take(layer, fallback_take_id)
            if fallback is not None and any(
                (str(fallback.id), str(event.id)) in selected_lookup for event in fallback.events
            ):
                return fallback_take_id

        for take in layer.takes:
            if any((str(take.id), str(event.id)) in selected_lookup for event in take.events):
                return take.id
        return None

    @staticmethod
    def _main_take(layer: Layer) -> Take | None:
        if layer.takes:
            return layer.takes[0]
        return None

    @staticmethod
    def _resolve_or_create_main_take(layer: Layer) -> Take:
        main_take = TimelineOrchestratorSelectionStateMixin._main_take(layer)
        if main_take is not None:
            return main_take
        main_take = Take(
            id=TakeId(f"{layer.id}:main"),
            layer_id=layer.id,
            name="Main",
        )
        layer.takes.insert(0, main_take)
        return main_take

    @staticmethod
    def _find_take(layer: Layer, take_id: TakeId) -> Take | None:
        for take in layer.takes:
            if take.id == take_id:
                return take
        return None

    def _selected_events(self, timeline: Timeline) -> list[tuple[Layer, Take, Event]]:
        selected_refs = self._selected_event_refs(timeline)
        if not selected_refs:
            return []

        records = self._selected_event_records(timeline, selected_refs)
        return [(record.layer, record.take, record.event) for record in records]

    @staticmethod
    def _selected_events_by_ids(timeline: Timeline, selected_ids: list[EventId]) -> list[Event]:
        selected_lookup = set(selected_ids)
        selected_order = {str(event_id): index for index, event_id in enumerate(selected_ids)}
        selected: list[Event] = []
        for layer in timeline.layers:
            main_take = TimelineOrchestratorSelectionStateMixin._main_take(layer)
            if main_take is None:
                continue
            for event in main_take.events:
                if event.id in selected_lookup:
                    selected.append(event)

        selected.sort(key=lambda event: selected_order[str(event.id)])
        return selected

    def _selected_event_records_by_layer(
        self, timeline: Timeline
    ) -> list[tuple[Layer, list[SelectedEventRecord]]]:
        records = self._selected_event_records(timeline, self._selected_event_refs(timeline))
        explicit_selected_layer_scope = set(dict.fromkeys(timeline.selection.selected_layer_ids))
        if explicit_selected_layer_scope:
            records = [
                record for record in records if record.layer.id in explicit_selected_layer_scope
            ]
        grouped: dict[LayerId, list[SelectedEventRecord]] = {}
        layer_order: dict[LayerId, int] = {}
        for index, layer in enumerate(
            sorted(timeline.layers, key=lambda value: value.order_index)
        ):
            layer_order[layer.id] = index
        layer_lookup: dict[LayerId, Layer] = {}
        for record in records:
            layer_lookup[record.layer.id] = record.layer
            grouped.setdefault(record.layer.id, []).append(record)
        ordered_layer_ids = sorted(
            grouped.keys(), key=lambda layer_id: layer_order.get(layer_id, 0)
        )
        return [(layer_lookup[layer_id], grouped[layer_id]) for layer_id in ordered_layer_ids]

    @staticmethod
    def _selected_layer_scope(timeline: Timeline) -> list[LayerId]:
        selected_layer_ids = list(dict.fromkeys(timeline.selection.selected_layer_ids))
        if selected_layer_ids:
            return selected_layer_ids
        if timeline.selection.selected_layer_id is not None:
            return [timeline.selection.selected_layer_id]
        return []

    @staticmethod
    def _ordered_visible_layers(timeline: Timeline) -> list[Layer]:
        return [
            layer
            for layer in sorted(timeline.layers, key=lambda value: value.order_index)
            if layer.presentation_hints.visible
        ]

    @staticmethod
    def _ordered_events(take: Take) -> list[Event]:
        return sorted(take.events, key=lambda event: (event.start, event.end, str(event.id)))

    def _navigation_layer_id(self, timeline: Timeline) -> LayerId | None:
        if timeline.selection.selected_layer_id is not None:
            return timeline.selection.selected_layer_id
        selected_layer_ids = list(dict.fromkeys(timeline.selection.selected_layer_ids))
        if selected_layer_ids:
            return selected_layer_ids[-1]
        return None

    def _navigation_take_for_layer(self, timeline: Timeline, layer: Layer) -> Take | None:
        selected_take_id = timeline.selection.selected_take_id
        if selected_take_id is not None:
            selected_take = self._find_take(layer, selected_take_id)
            if selected_take is not None:
                return selected_take

        selected_refs = [
            event_ref
            for event_ref in self._selected_event_refs(timeline)
            if event_ref.layer_id == layer.id
        ]
        for event_ref in reversed(selected_refs):
            selected_take = self._find_take(layer, event_ref.take_id)
            if selected_take is not None:
                return selected_take
        return self._main_take(layer)

    @staticmethod
    def _adjacent_event_for_selection(
        ordered_events: list[Event],
        selected_refs: list[EventRef],
        *,
        direction: int,
        playhead_seconds: float | None = None,
        include_demoted: bool = False,
    ) -> Event | None:
        if not ordered_events:
            return None

        step = 1 if direction > 0 else -1 if direction < 0 else 0
        if step == 0:
            return None

        if not selected_refs:
            return TimelineOrchestratorSelectionStateMixin._adjacent_event_for_playhead(
                ordered_events,
                direction=step,
                playhead_seconds=playhead_seconds,
                include_demoted=include_demoted,
            )

        selected_indices: list[int] = []
        for selected_ref in selected_refs:
            match_index = next(
                (
                    index
                    for index, event in enumerate(ordered_events)
                    if event.id == selected_ref.event_id
                ),
                None,
            )
            if match_index is not None:
                selected_indices.append(match_index)
        if not selected_indices:
            return TimelineOrchestratorSelectionStateMixin._adjacent_event_for_playhead(
                ordered_events,
                direction=step,
                playhead_seconds=playhead_seconds,
                include_demoted=include_demoted,
            )

        anchor_index = selected_indices[-1]
        target_index = anchor_index + step
        while 0 <= target_index < len(ordered_events):
            target_event = ordered_events[target_index]
            if include_demoted or not TimelineOrchestratorSelectionStateMixin._event_is_demoted(
                target_event
            ):
                return target_event
            target_index += step
        return None

    @staticmethod
    def _adjacent_event_for_playhead(
        ordered_events: list[Event],
        *,
        direction: int,
        playhead_seconds: float | None,
        include_demoted: bool = False,
    ) -> Event | None:
        if not ordered_events:
            return None
        if playhead_seconds is None:
            iterable = ordered_events if direction > 0 else reversed(ordered_events)
            for event in iterable:
                if include_demoted or not TimelineOrchestratorSelectionStateMixin._event_is_demoted(
                    event
                ):
                    return event
            return None

        playhead = float(playhead_seconds)
        if direction > 0:
            for event in ordered_events:
                center = 0.5 * (float(event.start) + float(event.end))
                if center >= playhead and (
                    include_demoted
                    or not TimelineOrchestratorSelectionStateMixin._event_is_demoted(event)
                ):
                    return event
            return None

        for event in reversed(ordered_events):
            center = 0.5 * (float(event.start) + float(event.end))
            if center <= playhead and (
                include_demoted
                or not TimelineOrchestratorSelectionStateMixin._event_is_demoted(event)
            ):
                return event
        return None

    @staticmethod
    def _event_is_demoted(event: Event) -> bool:
        return event.promotion_state == "demoted"

    def _find_layer(self, timeline: Timeline, layer_id: LayerId) -> Layer:
        for layer in timeline.layers:
            if layer.id == layer_id:
                return layer
        raise ValueError(f"Layer not found: {layer_id}")
