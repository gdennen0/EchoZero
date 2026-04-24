"""Scoped event batch helpers for the timeline orchestrator.
Exists to keep batch-selection scopes and multi-event edit semantics out of the lower-level selection and event-edit mixins.
Connects canonical timeline intents to shared selected-events, take, and layer-main event targeting rules.
"""

from __future__ import annotations

from echozero.application.shared.ids import LayerId, TakeId
from echozero.application.timeline.event_batch_scope import (
    EventBatchScope,
    ResolvedEventBatchScope,
)
from echozero.application.timeline.models import EventRef, Layer, Take, Timeline
from echozero.application.timeline.orchestrator_event_edit_mixin import (
    TimelineOrchestratorEventEditMixin,
)


class TimelineOrchestratorEventBatchMixin(TimelineOrchestratorEventEditMixin):
    """Applies scoped event batch operations through the canonical selection/edit path."""

    def _handle_select_every_other_events(
        self,
        timeline: Timeline,
        *,
        scope: EventBatchScope,
    ) -> None:
        resolved = self._resolve_event_batch_scope(timeline, scope)
        if resolved.is_empty:
            return

        next_refs = [
            event_ref
            for event_ref_group in resolved.event_ref_groups
            for index, event_ref in enumerate(event_ref_group)
            if index % 2 == 0
        ]
        self._apply_event_batch_scope_selection(timeline, resolved, next_refs)

    def _handle_renumber_event_cue_numbers(
        self,
        timeline: Timeline,
        *,
        scope: EventBatchScope,
        start_at: int,
        step: int,
    ) -> None:
        if start_at < 1 or step < 1:
            return

        resolved = self._resolve_event_batch_scope(timeline, scope)
        if resolved.is_empty:
            return

        for event_ref_group in resolved.event_ref_groups:
            next_cue_number = start_at
            for record in self._selected_event_records(timeline, list(event_ref_group)):
                record.event.cue_number = next_cue_number
                next_cue_number += step

        self._apply_event_batch_scope_selection(timeline, resolved, list(resolved.event_refs))

    def _resolve_event_batch_scope(
        self,
        timeline: Timeline,
        scope: EventBatchScope,
    ) -> ResolvedEventBatchScope:
        if scope.mode == "selected_events":
            event_refs = tuple(self._selected_event_refs(timeline))
            selected_layer_ids = tuple(self._selected_layer_scope(timeline))
            anchor_layer_id = (
                timeline.selection.selected_layer_id
                if timeline.selection.selected_layer_id is not None
                else (event_refs[-1].layer_id if event_refs else None)
            )
            anchor_take_id = (
                timeline.selection.selected_take_id
                if timeline.selection.selected_take_id is not None
                else (event_refs[-1].take_id if event_refs else None)
            )
            return ResolvedEventBatchScope(
                scope=scope,
                event_refs=event_refs,
                event_ref_groups=((event_refs,) if event_refs else ()),
                anchor_layer_id=anchor_layer_id,
                anchor_take_id=anchor_take_id,
                selected_layer_ids=selected_layer_ids,
                label="selection",
            )

        if scope.mode == "take":
            assert scope.layer_id is not None
            assert scope.take_id is not None
            layer = self._find_layer(timeline, scope.layer_id)
            take = self._find_take(layer, scope.take_id)
            if take is None:
                return ResolvedEventBatchScope(
                    scope=scope,
                    event_refs=(),
                    event_ref_groups=(),
                    anchor_layer_id=layer.id,
                    anchor_take_id=scope.take_id,
                    selected_layer_ids=(layer.id,),
                    label="take",
                )
            event_refs = self._ordered_event_refs_for_take(layer, take)
            return ResolvedEventBatchScope(
                scope=scope,
                event_refs=event_refs,
                event_ref_groups=((event_refs,) if event_refs else ()),
                anchor_layer_id=layer.id,
                anchor_take_id=take.id,
                selected_layer_ids=(layer.id,),
                label="take",
            )

        if scope.mode == "layer_main":
            assert scope.layer_id is not None
            layer = self._find_layer(timeline, scope.layer_id)
            main_take = self._main_take(layer)
            event_refs = (
                self._ordered_event_refs_for_take(layer, main_take) if main_take is not None else ()
            )
            return ResolvedEventBatchScope(
                scope=scope,
                event_refs=event_refs,
                event_ref_groups=((event_refs,) if event_refs else ()),
                anchor_layer_id=layer.id,
                anchor_take_id=main_take.id if main_take is not None else None,
                selected_layer_ids=(layer.id,),
                label="layer",
            )

        if scope.mode == "region":
            assert scope.region_id is not None
            region = next(
                (candidate for candidate in timeline.regions if candidate.id == scope.region_id),
                None,
            )
            if region is None:
                return ResolvedEventBatchScope(
                    scope=scope,
                    event_refs=(),
                    event_ref_groups=(),
                    anchor_layer_id=None,
                    anchor_take_id=None,
                    selected_layer_ids=(),
                    label="region",
                )
            event_ref_groups = self._region_main_groups(
                timeline,
                start_seconds=float(region.start),
                end_seconds=float(region.end),
            )
            event_refs = self._flatten_event_ref_groups(event_ref_groups)
            anchor_layer_id = event_refs[-1].layer_id if event_refs else None
            anchor_take_id = event_refs[-1].take_id if event_refs else None
            selected_layer_ids = tuple(
                dict.fromkeys(event_ref.layer_id for event_ref in event_refs)
            )
            return ResolvedEventBatchScope(
                scope=scope,
                event_refs=event_refs,
                event_ref_groups=event_ref_groups,
                anchor_layer_id=anchor_layer_id,
                anchor_take_id=anchor_take_id,
                selected_layer_ids=selected_layer_ids,
                label="region",
            )

        selected_layer_ids = tuple(self._selected_layer_scope(timeline))
        anchor_layer_id = self._navigation_layer_id(timeline)
        if anchor_layer_id is None and selected_layer_ids:
            anchor_layer_id = selected_layer_ids[-1]
        event_ref_groups = self._selected_layers_main_groups(timeline, selected_layer_ids)
        event_refs = self._flatten_event_ref_groups(event_ref_groups)
        anchor_take_id = self._anchor_take_id_for_selected_layers(timeline, anchor_layer_id)
        return ResolvedEventBatchScope(
            scope=scope,
            event_refs=event_refs,
            event_ref_groups=event_ref_groups,
            anchor_layer_id=anchor_layer_id,
            anchor_take_id=anchor_take_id,
            selected_layer_ids=selected_layer_ids,
            label="selected layers",
        )

    def _apply_event_batch_scope_selection(
        self,
        timeline: Timeline,
        resolved: ResolvedEventBatchScope,
        event_refs: list[EventRef],
    ) -> None:
        timeline.selection.selected_layer_id = (
            resolved.anchor_layer_id if resolved.selected_layer_ids else None
        )
        timeline.selection.selected_layer_ids = list(resolved.selected_layer_ids)
        timeline.selection.selected_take_id = (
            resolved.anchor_take_id if event_refs else None
        )
        self._set_selected_event_refs(timeline, event_refs)

    def _selected_layers_main_groups(
        self,
        timeline: Timeline,
        selected_layer_ids: tuple[LayerId, ...],
    ) -> tuple[tuple[EventRef, ...], ...]:
        groups: list[tuple[EventRef, ...]] = []
        for layer_id in selected_layer_ids:
            layer = self._find_layer(timeline, layer_id)
            main_take = self._main_take(layer)
            if main_take is None:
                continue
            event_refs = self._ordered_event_refs_for_take(layer, main_take)
            if event_refs:
                groups.append(event_refs)
        return tuple(groups)

    def _anchor_take_id_for_selected_layers(
        self,
        timeline: Timeline,
        anchor_layer_id: LayerId | None,
    ) -> TakeId | None:
        if anchor_layer_id is None:
            return None
        layer = self._find_layer(timeline, anchor_layer_id)
        main_take = self._main_take(layer)
        return main_take.id if main_take is not None else None

    def _ordered_event_refs_for_take(
        self,
        layer: Layer,
        take: Take,
    ) -> tuple[EventRef, ...]:
        return tuple(
            self._event_ref(layer.id, take.id, event.id) for event in self._ordered_events(take)
        )

    def _region_main_groups(
        self,
        timeline: Timeline,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> tuple[tuple[EventRef, ...], ...]:
        groups: list[tuple[EventRef, ...]] = []
        for layer in self._ordered_visible_layers(timeline):
            if layer.presentation_hints.locked:
                continue
            main_take = self._main_take(layer)
            if main_take is None:
                continue
            refs = tuple(
                self._event_ref(layer.id, main_take.id, event.id)
                for event in self._ordered_events(main_take)
                if float(event.start) < end_seconds and float(event.end) > start_seconds
            )
            if refs:
                groups.append(refs)
        return tuple(groups)

    @staticmethod
    def _flatten_event_ref_groups(
        event_ref_groups: tuple[tuple[EventRef, ...], ...],
    ) -> tuple[EventRef, ...]:
        return tuple(
            event_ref
            for event_ref_group in event_ref_groups
            for event_ref in event_ref_group
        )
