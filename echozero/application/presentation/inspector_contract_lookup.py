"""Lookup helpers for the timeline inspector contract.
Exists to isolate presentation-layer object lookup from preview and context-action support logic.
Connects inspector contract builders to stable layer, take, and event resolution on timeline presentations.
"""

from __future__ import annotations

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)


def find_layer(presentation: TimelinePresentation, layer_id: object) -> LayerPresentation | None:
    for layer in presentation.layers:
        if layer.layer_id == layer_id:
            return layer
    return None


def find_take(
    presentation: TimelinePresentation,
    *,
    layer_id: object | None,
    take_id: object,
) -> tuple[LayerPresentation, TakeLanePresentation] | None:
    layer = find_layer(presentation, layer_id) if layer_id is not None else None
    if layer is not None:
        for take in layer.takes:
            if take.take_id == take_id:
                return layer, take
    for candidate_layer in presentation.layers:
        for take in candidate_layer.takes:
            if take.take_id == take_id:
                return candidate_layer, take
    return None


def find_event(
    presentation: TimelinePresentation,
    *,
    layer_id: object | None,
    take_id: object | None,
    event_id: object,
) -> tuple[LayerPresentation, TakeLanePresentation | None, EventPresentation] | None:
    layer = find_layer(presentation, layer_id) if layer_id is not None else None
    layers = [layer] if layer is not None else list(presentation.layers)
    for candidate_layer in layers:
        if candidate_layer is None:
            continue
        if take_id is None or take_id == candidate_layer.main_take_id:
            for event in candidate_layer.events:
                if event.event_id == event_id:
                    return candidate_layer, None, event
        for take in candidate_layer.takes:
            if take_id is not None and take.take_id != take_id:
                continue
            for event in take.events:
                if event.event_id == event_id:
                    return candidate_layer, take, event
    return None


def find_selected_event(
    presentation: TimelinePresentation,
    event_id: object,
) -> tuple[LayerPresentation, TakeLanePresentation | None, EventPresentation] | None:
    return find_event(
        presentation,
        layer_id=presentation.selected_layer_id,
        take_id=presentation.selected_take_id,
        event_id=event_id,
    ) or find_event(
        presentation,
        layer_id=presentation.selected_layer_id,
        take_id=None,
        event_id=event_id,
    )
