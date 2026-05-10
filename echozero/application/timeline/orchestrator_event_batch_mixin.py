"""Scoped event batch helpers for the timeline orchestrator.
Exists to keep batch-selection scopes and multi-event edit semantics out of the lower-level selection and event-edit mixins.
Connects canonical timeline intents to shared selected-events, take, and layer-main event targeting rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_batch_scope import (
    EventBatchScope,
    ResolvedEventBatchScope,
)
from echozero.application.timeline.models import Event, EventRef, Layer, Take, Timeline
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
        similarity_threshold_override: float | None,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        take = self._find_take(layer, take_id)
        if take is None:
            return
        anchor_event = next(
            (candidate for candidate in take.events if candidate.id == event_id), None
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

        candidate_records = [
            record
            for candidate_ref_group in candidate_ref_groups
            for record in self._selected_event_records(timeline, list(candidate_ref_group))
        ]
        anchor_features = _event_similarity_features(anchor_event)
        if anchor_features is None or not anchor_features.has_signal:
            similar_event_refs = [self._event_ref(layer.id, take.id, anchor_event.id)]
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_layer_ids = list(selected_layer_ids)
            timeline.selection.selected_take_id = anchor_take_id
            self._set_selected_event_refs(timeline, similar_event_refs)
            return

        scope_context = _build_similarity_scope_context(
            anchor_features=anchor_features,
            candidate_events=[record.event for record in candidate_records],
        )
        score_threshold = _similarity_threshold(
            match_strength,
            similarity_threshold_override=similarity_threshold_override,
        )
        similar_event_refs: list[EventRef] = []
        for record in candidate_records:
            candidate_features = _event_similarity_features(record.event)
            score = _event_similarity_score(
                anchor_features=anchor_features,
                candidate_features=candidate_features,
                scope_context=scope_context,
            )
            if score is None or score < score_threshold:
                continue
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
            self._event_ref(layer.id, take.id, event.id) for event in self._ordered_events(take)
        )

    @staticmethod
    def _flatten_event_ref_groups(
        event_ref_groups: tuple[tuple[EventRef, ...], ...],
    ) -> tuple[EventRef, ...]:
        return tuple(
            event_ref for event_ref_group in event_ref_groups for event_ref in event_ref_group
        )


@dataclass(slots=True)
class _EventSimilarityFeatures:
    label_weights: dict[str, float]
    confidence: float | None
    duration_log: float

    @property
    def has_signal(self) -> bool:
        return bool(self.label_weights) or self.confidence is not None


@dataclass(slots=True)
class _SimilarityScopeContext:
    label_vocabulary: tuple[str, ...]
    duration_mean: float
    duration_std: float


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


def _build_similarity_scope_context(
    *,
    anchor_features: _EventSimilarityFeatures,
    candidate_events: list[Event],
) -> _SimilarityScopeContext:
    label_tokens = set(anchor_features.label_weights)
    duration_values = [anchor_features.duration_log]
    for candidate_event in candidate_events:
        candidate_features = _event_similarity_features(candidate_event)
        if candidate_features is None:
            continue
        label_tokens.update(candidate_features.label_weights)
        duration_values.append(candidate_features.duration_log)
    duration_mean = sum(duration_values) / len(duration_values)
    duration_variance = sum(
        (duration_value - duration_mean) ** 2 for duration_value in duration_values
    ) / len(duration_values)
    duration_std = math.sqrt(duration_variance)
    if duration_std < 1e-6:
        duration_std = 1.0
    return _SimilarityScopeContext(
        label_vocabulary=tuple(sorted(label_tokens)),
        duration_mean=duration_mean,
        duration_std=duration_std,
    )


def _event_similarity_score(
    *,
    anchor_features: _EventSimilarityFeatures,
    candidate_features: _EventSimilarityFeatures | None,
    scope_context: _SimilarityScopeContext,
) -> float | None:
    if not anchor_features.has_signal:
        return None
    if candidate_features is None or not candidate_features.has_signal:
        return None
    anchor_vector = _feature_vector(anchor_features, scope_context)
    candidate_vector = _feature_vector(candidate_features, scope_context)
    return _cosine_similarity(anchor_vector, candidate_vector)


def _feature_vector(
    features: _EventSimilarityFeatures,
    scope_context: _SimilarityScopeContext,
) -> tuple[float, ...]:
    has_label_axis = bool(scope_context.label_vocabulary)
    vector = [features.label_weights.get(label, 0.0) for label in scope_context.label_vocabulary]
    confidence = features.confidence if features.confidence is not None else 0.0
    vector.append(confidence)
    vector.append(1.0 - confidence)
    duration_z_score = 0.0
    if has_label_axis:
        duration_z_score = (
            features.duration_log - scope_context.duration_mean
        ) / scope_context.duration_std
    vector.append(duration_z_score)
    return tuple(vector)


def _cosine_similarity(
    anchor_vector: tuple[float, ...], candidate_vector: tuple[float, ...]
) -> float | None:
    if len(anchor_vector) != len(candidate_vector):
        return None
    dot_product = sum(
        anchor_value * candidate_value
        for anchor_value, candidate_value in zip(anchor_vector, candidate_vector)
    )
    anchor_norm = math.sqrt(sum(value * value for value in anchor_vector))
    candidate_norm = math.sqrt(sum(value * value for value in candidate_vector))
    if anchor_norm <= 0.0 or candidate_norm <= 0.0:
        return None
    return dot_product / (anchor_norm * candidate_norm)


def _event_similarity_features(event: Event) -> _EventSimilarityFeatures | None:
    label_weights = _event_label_weights(event)
    confidence = _normalized_confidence(_event_confidence(event))
    duration_log = math.log1p(max(0.0, float(event.duration)))
    features = _EventSimilarityFeatures(
        label_weights=label_weights,
        confidence=confidence,
        duration_log=duration_log,
    )
    if not features.has_signal:
        return None
    return features


def _event_label_weights(event: Event) -> dict[str, float]:
    label_weights: dict[str, float] = {}
    _merge_label_weights(label_weights, event.classifications)
    detection = event.detection_metadata
    if isinstance(detection, dict):
        _merge_label_weights(label_weights, detection)
    if not label_weights:
        fallback_token = _normalize_similarity_token(event.label)
        if fallback_token is not None:
            _add_label_weight(label_weights, fallback_token, 1.0)
    return label_weights


def _merge_label_weights(label_weights: dict[str, float], mapping: dict[str, object]) -> None:
    default_weight = _normalized_confidence(_confidence_from_mapping(mapping)) or 1.0
    for key in ("class", "label", "type", "note", "instrument"):
        token = _normalize_similarity_token(mapping.get(key))
        if token is not None:
            _add_label_weight(label_weights, token, default_weight)

    for key, value in mapping.items():
        if key in {"confidence", "classifier_score", "score", "probability"}:
            continue
        if isinstance(value, dict):
            nested_token = _token_from_nested_classification(value)
            nested_weight = (
                _normalized_confidence(_confidence_from_mapping(value)) or default_weight
            )
            if nested_token is not None:
                _add_label_weight(label_weights, nested_token, nested_weight)
                continue
            if _mapping_looks_like_classifier_signal(value):
                key_token = _normalize_classifier_label_key(key)
                if key_token is not None:
                    _add_label_weight(label_weights, key_token, nested_weight)
                    continue
        token = _normalize_similarity_token(value)
        if token is not None:
            _add_label_weight(label_weights, token, default_weight)


def _add_label_weight(label_weights: dict[str, float], token: str, raw_weight: float) -> None:
    normalized_weight = _normalized_confidence(raw_weight) or 0.0
    current_weight = label_weights.get(token)
    if current_weight is None or normalized_weight > current_weight:
        label_weights[token] = normalized_weight


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
    nested_confidences = [
        confidence
        for value in event.classifications.values()
        if isinstance(value, dict)
        for confidence in [_confidence_from_mapping(value)]
        if confidence is not None
    ]
    if nested_confidences:
        return max(nested_confidences)
    detection = event.detection_metadata
    if isinstance(detection, dict):
        for key in ("confidence", "classifier_score", "score", "probability"):
            parsed = _coerce_numeric(detection.get(key))
            if parsed is not None:
                return parsed
    return None


def _normalized_confidence(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(0.0, min(1.0, float(value)))


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


def _token_from_nested_classification(value: dict[str, object]) -> str | None:
    for key in ("class", "label", "type", "note", "instrument"):
        normalized = _normalize_similarity_token(value.get(key))
        if normalized is not None:
            return normalized
    return None


def _mapping_looks_like_classifier_signal(value: dict[str, object]) -> bool:
    if _token_from_nested_classification(value) is not None:
        return True
    return _confidence_from_mapping(value) is not None


def _normalize_classifier_label_key(key: str) -> str | None:
    normalized = _normalize_similarity_token(key)
    if normalized is None:
        return None
    if normalized in {
        "classifier",
        "classifiers",
        "classification",
        "classifications",
        "model",
        "models",
        "prediction",
        "predictions",
        "result",
        "results",
        "score",
        "scores",
        "probability",
        "probabilities",
        "confidence",
    }:
        return None
    return normalized


def _confidence_from_mapping(value: dict[str, object]) -> float | None:
    for key in ("confidence", "classifier_score", "score", "probability"):
        parsed = _coerce_numeric(value.get(key))
        if parsed is not None:
            return parsed
    return None
