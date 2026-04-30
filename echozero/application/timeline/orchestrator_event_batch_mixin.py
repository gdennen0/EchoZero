"""Scoped event batch helpers for the timeline orchestrator.
Exists to keep batch-selection scopes and multi-event edit semantics out of the lower-level selection and event-edit mixins.
Connects canonical timeline intents to shared selected-events, take, and layer-main event targeting rules.
"""

from __future__ import annotations

from echozero.application.shared.ids import EventId, LayerId, TakeId
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

    def _handle_select_similar_sounding_events(
        self,
        timeline: Timeline,
        *,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        scope_mode: str,
        match_strength: str,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        take = self._find_take(layer, take_id)
        if take is None:
            return
        anchor_event = next((candidate for candidate in take.events if candidate.id == event_id), None)
        if anchor_event is None:
            return

        if scope_mode == "take":
            candidate_ref_groups = (
                tuple(self._event_ref(layer.id, take.id, event.id) for event in self._ordered_events(take)),
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

        anchor_profile = _event_sound_profile(anchor_event)
        similar_event_refs: list[EventRef] = []
        for candidate_ref_group in candidate_ref_groups:
            records = self._selected_event_records(timeline, list(candidate_ref_group))
            for record in records:
                if _events_sound_similar(
                    anchor=anchor_event,
                    candidate=record.event,
                    anchor_profile=anchor_profile,
                    match_strength=match_strength,
                ):
                    similar_event_refs.append(
                        self._event_ref(record.layer.id, record.take.id, record.event.id)
                    )
        if not similar_event_refs:
            similar_event_refs = [self._event_ref(layer.id, take.id, anchor_event.id)]

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_layer_ids = list(selected_layer_ids)
        timeline.selection.selected_take_id = anchor_take_id
        self._set_selected_event_refs(timeline, similar_event_refs)

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


def _events_sound_similar(
    *,
    anchor: Event,
    candidate: Event,
    anchor_profile: tuple[str | None, float | None],
    match_strength: str,
) -> bool:
    anchor_token, anchor_confidence = anchor_profile
    candidate_token = _event_similarity_token(candidate)
    if anchor_token is not None:
        if candidate_token != anchor_token:
            return False
        return _duration_is_similar(
            anchor=anchor,
            candidate=candidate,
            match_strength=match_strength,
        )

    candidate_confidence = _event_confidence(candidate)
    confidence_tolerance = _confidence_tolerance(match_strength)
    if (
        anchor_confidence is not None
        and candidate_confidence is not None
        and abs(candidate_confidence - anchor_confidence) > confidence_tolerance
    ):
        return False
    return _duration_is_similar(
        anchor=anchor,
        candidate=candidate,
        match_strength=match_strength,
    )


def _event_sound_profile(event: Event) -> tuple[str | None, float | None]:
    return (_event_similarity_token(event), _event_confidence(event))


def _event_similarity_token(event: Event) -> str | None:
    for key in ("class", "label", "type", "note", "instrument"):
        normalized = _normalize_similarity_token(event.classifications.get(key))
        if normalized is not None:
            return normalized
    for value in event.classifications.values():
        normalized = _normalize_similarity_token(value)
        if normalized is not None:
            return normalized
    detection = event.detection_metadata
    if isinstance(detection, dict):
        for key in ("class", "label", "type"):
            normalized = _normalize_similarity_token(detection.get(key))
            if normalized is not None:
                return normalized
    origin = event.origin
    if ":" in origin:
        normalized = _normalize_similarity_token(origin.rsplit(":", maxsplit=1)[-1])
        if normalized is not None:
            return normalized
    return _normalize_similarity_token(event.label)


def _normalize_similarity_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = " ".join(value.strip().replace("_", " ").replace("-", " ").split())
    if not token:
        return None
    lowered = token.casefold()
    if lowered in {"event", "events", "main", "take", "clip", "cue"}:
        return None
    return lowered


def _event_confidence(event: Event) -> float | None:
    for key in ("confidence", "classifier_score", "score", "probability"):
        value = event.classifications.get(key)
        parsed = _coerce_numeric(value)
        if parsed is not None:
            return parsed
    detection = event.detection_metadata
    if isinstance(detection, dict):
        for key in ("confidence", "classifier_score", "score", "probability"):
            parsed = _coerce_numeric(detection.get(key))
            if parsed is not None:
                return parsed
    return None


def _coerce_numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _confidence_tolerance(match_strength: str) -> float:
    return {
        "strict": 0.15,
        "balanced": 0.25,
        "loose": 0.45,
    }.get(match_strength, 0.25)


def _duration_is_similar(
    *,
    anchor: Event,
    candidate: Event,
    match_strength: str,
) -> bool:
    anchor_duration = max(0.0, float(anchor.duration))
    candidate_duration = max(0.0, float(candidate.duration))
    if match_strength == "strict":
        tolerance = max(0.02, min(0.12, anchor_duration * 0.25))
    elif match_strength == "loose":
        tolerance = max(0.05, min(0.35, anchor_duration * 0.8))
    else:
        tolerance = max(0.03, min(0.2, anchor_duration * 0.5))
    return abs(candidate_duration - anchor_duration) <= tolerance
