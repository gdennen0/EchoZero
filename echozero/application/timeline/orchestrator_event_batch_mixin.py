"""Scoped event batch helpers for the timeline orchestrator.
Exists to keep batch-selection scopes and multi-event edit semantics out of the lower-level selection and event-edit mixins.
Connects canonical timeline intents to shared selected-events, take, and layer-main event targeting rules.
"""

from __future__ import annotations

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_comparison_service import (
    EventComparisonCandidateRecord,
    EventComparisonRequest,
    EventComparisonService,
)
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

    def _handle_select_similar_events(
        self,
        timeline: Timeline,
        *,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        scope_mode: str,
        comparison_mode: str,
        match_strength: str,
        similarity_threshold_override: float | None,
        comparison_options: dict[str, object],
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        take = self._find_take(layer, take_id)
        if take is None:
            return
        anchor_event = next(
            (candidate for candidate in take.events if candidate.id == event_id),
            None,
        )
        if anchor_event is None:
            return

        if scope_mode == "take":
            candidate_ref_groups = (
                tuple(
                    self._event_ref(layer.id, take.id, event.id)
                    for event in self._ordered_events(take)
                ),
            )
            selected_layer_ids = (layer.id,)
            anchor_take_id = take.id
        elif scope_mode == "layer":
            candidate_ref_groups = tuple(
                tuple(
                    self._event_ref(layer.id, candidate_take.id, event.id)
                    for event in self._ordered_events(candidate_take)
                )
                for candidate_take in layer.takes
                if candidate_take.events
            )
            selected_layer_ids = (layer.id,)
            anchor_take_id = take.id
        else:
            selected_scope_layer_ids = tuple(self._selected_layer_scope(timeline))
            if not selected_scope_layer_ids:
                selected_scope_layer_ids = (layer.id,)
            candidate_ref_groups = self._selected_layers_main_groups(
                timeline,
                selected_scope_layer_ids,
            )
            selected_layer_ids = selected_scope_layer_ids
            anchor_take_id = self._anchor_take_id_for_selected_layers(timeline, layer.id)

        selected_records = [
            record
            for candidate_ref_group in candidate_ref_groups
            for record in self._selected_event_records(timeline, list(candidate_ref_group))
        ]
        candidate_records = [
            EventComparisonCandidateRecord(
                layer_id=record.layer.id,
                take_id=record.take.id,
                event=record.event,
                layer=record.layer,
                take=record.take,
            )
            for record in selected_records
        ]
        similar_event_refs = EventComparisonService().select_matching_event_refs(
            anchor_layer=layer,
            anchor_take=take,
            candidate_records=candidate_records,
            request=EventComparisonRequest(
                anchor_event_id=event_id,
                comparison_mode=comparison_mode,
                similarity_threshold=_similarity_threshold(
                    match_strength,
                    similarity_threshold_override=similarity_threshold_override,
                ),
                comparison_settings=comparison_options.get("comparison_settings"),
            ),
        )

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_layer_ids = list(selected_layer_ids)
        timeline.selection.selected_take_id = anchor_take_id
        self._set_selected_event_refs(timeline, similar_event_refs)

    def _handle_select_similar_sounding_events(
        self,
        timeline: Timeline,
        *,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        scope_mode: str,
        match_strength: str,
        similarity_threshold_override: float | None,
    ) -> None:
        self._handle_select_similar_events(
            timeline,
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
            scope_mode=scope_mode,
            comparison_mode="shape_envelope",
            match_strength=match_strength,
            similarity_threshold_override=similarity_threshold_override,
            comparison_options={},
        )

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
                self._ordered_event_refs_for_take(layer, main_take)
                if main_take is not None
                else ()
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
        timeline.selection.selected_take_id = resolved.anchor_take_id if event_refs else None
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
            self._event_ref(layer.id, take.id, event.id)
            for event in self._ordered_events(take)
        )

    @staticmethod
    def _flatten_event_ref_groups(
        event_ref_groups: tuple[tuple[EventRef, ...], ...],
    ) -> tuple[EventRef, ...]:
        return tuple(
            event_ref
            for event_ref_group in event_ref_groups
            for event_ref in event_ref_group
        )


def _similarity_threshold(
    match_strength: str,
    *,
    similarity_threshold_override: float | None = None,
) -> float:
    if similarity_threshold_override is not None:
        return max(0.0, min(1.0, float(similarity_threshold_override)))
    return {
        "very_strict": 0.95,
        "strict": 0.90,
        "balanced": 0.78,
        "loose": 0.65,
    }.get(match_strength, 0.78)
