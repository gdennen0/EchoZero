"""Scoped event batch helpers for the timeline orchestrator.
Exists to keep batch-selection scopes and multi-event edit semantics out of the lower-level selection and event-edit mixins.
Connects canonical timeline intents to shared selected-events, take, and layer-main event targeting rules.
"""

from __future__ import annotations

from uuid import uuid4

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.timeline.event_sequence_similarity import (
    EventSequenceMatchRequest,
    EventSequenceRecord,
    EventSequenceSimilarityService,
)
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
                comparison_settings=(
                    comparison_options.get("comparison_settings")
                    or comparison_options.get("artifact_path")
                    or comparison_options.get("mini_model_path")
                    or comparison_options
                ),
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

    def _handle_select_similar_event_sequences(
        self,
        timeline: Timeline,
        *,
        scope_mode: str,
        strictness: str,
        min_events: int,
        allow_missing_events: int,
        timing_weight: float,
        label_weight: float,
        length_weight: float,
        minimum_score: float | None,
        use_label_similarity: bool,
    ) -> None:
        selected_refs = self._selected_event_refs(timeline)
        selected_records = self._selected_event_records(timeline, selected_refs)
        if len(selected_records) < max(2, int(min_events)):
            return

        anchor_records = [
            EventSequenceRecord(
                event_ref=self._event_ref(record.layer.id, record.take.id, record.event.id),
                event=record.event,
            )
            for record in selected_records
        ]
        candidate_groups = self._sequence_candidate_groups(
            timeline,
            anchor_records=anchor_records,
            scope_mode=scope_mode,
        )
        matched_refs = EventSequenceSimilarityService().select_matching_event_refs(
            anchor_records=anchor_records,
            candidate_groups=candidate_groups,
            request=EventSequenceMatchRequest(
                min_events=min_events,
                strictness=strictness,
                allow_missing_events=allow_missing_events,
                timing_weight=timing_weight,
                label_weight=label_weight,
                length_weight=length_weight,
                minimum_score=minimum_score,
                use_label_similarity=use_label_similarity,
            ),
        )
        if not matched_refs:
            return

        timeline.selection.selected_layer_id = selected_records[-1].layer.id
        timeline.selection.selected_layer_ids = list(
            dict.fromkeys(event_ref.layer_id for event_ref in matched_refs)
        )
        timeline.selection.selected_take_id = selected_records[-1].take.id
        self._set_selected_event_refs(timeline, list(matched_refs))

    def _handle_create_event_sequence_from_selection(
        self,
        timeline: Timeline,
        *,
        name: str | None,
    ) -> None:
        selected_refs = self._selected_event_refs(timeline)
        records = self._selected_event_records(timeline, selected_refs)
        if len(records) < 2:
            return
        ordered_records = sorted(
            records,
            key=lambda record: (
                float(record.event.start),
                float(record.event.end),
                str(record.event.id),
            ),
        )
        sequence_id = f"seq_{uuid4().hex[:10]}"
        sequence_name = name or f"Sequence {len(ordered_records)} events"
        sequence_size = len(ordered_records)
        for index, record in enumerate(ordered_records, start=1):
            metadata = dict(record.event.metadata or {})
            metadata["sequence"] = {
                "id": sequence_id,
                "name": sequence_name,
                "order": index,
                "size": sequence_size,
                "created_from": "user",
            }
            record.event.metadata = metadata
        self._set_selected_event_refs(
            timeline,
            [
                self._event_ref(record.layer.id, record.take.id, record.event.id)
                for record in ordered_records
            ],
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

    def _handle_snap_events_to_beat_grid(
        self,
        timeline: Timeline,
        *,
        scope: EventBatchScope,
        grid_denominator: int,
        bpm: float,
        beat_anchor_seconds: float | None,
    ) -> None:
        if bpm <= 0.0 or grid_denominator not in {4, 8, 16, 32, 64}:
            return

        resolved = self._resolve_event_batch_scope(timeline, scope)
        if resolved.is_empty:
            return

        step_seconds = (60.0 / float(bpm)) * (4.0 / float(grid_denominator))
        if step_seconds <= 0.0:
            return

        anchor_seconds = max(0.0, float(beat_anchor_seconds or 0.0))
        for event_ref_group in resolved.event_ref_groups:
            affected_takes: dict[TakeId, Take] = {}
            for record in self._selected_event_records(timeline, list(event_ref_group)):
                duration = max(0.0, float(record.event.end) - float(record.event.start))
                snapped_start = _snap_time_to_grid(
                    float(record.event.start),
                    step_seconds=step_seconds,
                    anchor_seconds=anchor_seconds,
                )
                record.event.start = snapped_start
                record.event.end = snapped_start + duration
                affected_takes[record.take.id] = record.take
            for take in affected_takes.values():
                take.events = self._sorted_events(take.events)

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

    def _sequence_candidate_groups(
        self,
        timeline: Timeline,
        *,
        anchor_records: list[EventSequenceRecord],
        scope_mode: str,
    ) -> list[list[EventSequenceRecord]]:
        anchor_layer_ids = tuple(
            dict.fromkeys(record.event_ref.layer_id for record in anchor_records)
        )
        if scope_mode == "current_layer":
            layer_ids = anchor_layer_ids
        elif scope_mode == "selected_layers_main":
            layer_ids = tuple(self._selected_layer_scope(timeline)) or anchor_layer_ids
        else:
            layer_ids = tuple(
                layer.id for layer in timeline.layers if is_event_like_layer_kind(layer.kind)
            )

        groups: list[list[EventSequenceRecord]] = []
        for layer_id in layer_ids:
            layer = self._find_layer(timeline, layer_id)
            if not is_event_like_layer_kind(layer.kind):
                continue
            anchor_take_id = next(
                (
                    record.event_ref.take_id
                    for record in anchor_records
                    if record.event_ref.layer_id == layer_id
                ),
                None,
            )
            take = self._find_take(layer, anchor_take_id) if anchor_take_id is not None else None
            if take is None:
                take = self._main_take(layer)
            if take is None:
                continue
            groups.append(
                [
                    EventSequenceRecord(
                        event_ref=self._event_ref(layer.id, take.id, event.id),
                        event=event,
                    )
                    for event in self._ordered_events(take)
                ]
            )
        return groups


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


def _snap_time_to_grid(
    time_seconds: float,
    *,
    step_seconds: float,
    anchor_seconds: float,
) -> float:
    grid_index = round((float(time_seconds) - anchor_seconds) / step_seconds)
    return max(0.0, anchor_seconds + (float(grid_index) * step_seconds))
