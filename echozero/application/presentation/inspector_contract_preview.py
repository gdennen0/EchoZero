"""Preview helpers for the timeline inspector contract.
Exists to isolate preview source resolution and event clip params from general inspector support logic.
Connects timeline presentation objects to reusable preview metadata for the inspector surface.
"""

from __future__ import annotations

from echozero.application.presentation.inspector_contract_lookup import find_layer
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)


def event_preview_source_ref(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
) -> str | None:
    direct_source_ref = preview_audio_source_ref(layer=layer, take=take)
    if direct_source_ref:
        return direct_source_ref
    source_layer = preview_source_layer(presentation, layer)
    if source_layer is None:
        return None
    return preview_audio_source_ref(layer=source_layer, take=None)


def event_preview_params(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
    event: EventPresentation,
) -> dict[str, object] | None:
    source_ref = event_preview_source_ref(
        presentation,
        layer=layer,
        take=take,
    )
    if not source_ref or event.duration <= 0.0:
        return None
    waveform_key = event_preview_waveform_key(
        presentation,
        layer=layer,
        take=take,
    )
    source_audio_path = event_preview_source_audio_path(
        presentation,
        layer=layer,
        take=take,
    )
    return {
        "layer_id": layer.layer_id,
        "take_id": take.take_id if take is not None else layer.main_take_id,
        "event_id": event.event_id,
        "source_ref": source_ref,
        "source_audio_path": source_audio_path,
        "waveform_key": waveform_key,
        "start_seconds": float(event.start),
        "end_seconds": float(event.end),
        "duration_seconds": float(event.duration),
    }


def event_preview_source_audio_path(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
) -> str | None:
    if take is not None:
        if take.source_audio_path:
            return take.source_audio_path
        if layer.source_audio_path:
            return layer.source_audio_path
    elif layer.source_audio_path:
        return layer.source_audio_path
    source_layer = preview_source_layer(presentation, layer)
    if source_layer is None:
        return None
    return source_layer.source_audio_path


def event_preview_waveform_key(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
) -> str | None:
    if take is not None and take.waveform_key:
        return take.waveform_key
    if layer.waveform_key:
        return layer.waveform_key
    source_layer = preview_source_layer(presentation, layer)
    if source_layer is None:
        return None
    return source_layer.waveform_key


def preview_source_layer(
    presentation: TimelinePresentation,
    layer: LayerPresentation,
) -> LayerPresentation | None:
    source_layer_id = (layer.status.source_layer_id or "").strip()
    if not source_layer_id:
        return None
    return find_layer(presentation, source_layer_id)


def preview_audio_source_ref(
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
) -> str | None:
    if take is not None:
        return (
            take.playback_source_ref
            or take.source_audio_path
            or layer.playback_source_ref
            or layer.source_audio_path
        )
    return layer.playback_source_ref or layer.source_audio_path
