"""Timeline state helpers for the Qt app shell.
Exists to isolate selection/playback preservation and event preview lookup.
Connects app-shell refresh flows to stable timeline target restoration behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.models import Timeline


@dataclass(frozen=True, slots=True)
class EventClipPreview:
    source_ref: str
    start_seconds: float
    end_seconds: float
    gain_db: float


def clear_selected_events(timeline: Timeline) -> None:
    timeline.selection.selected_event_refs = []
    timeline.selection.selected_event_ids = []


def restore_timeline_targets(
    *,
    timeline: Timeline,
    prior_presentation: TimelinePresentation,
    current_presentation: TimelinePresentation,
) -> None:
    selected_layer_id = _resolve_preserved_selected_layer_id(
        prior_presentation,
        current_presentation,
    )
    selected_take_id = _resolve_take_id(
        current_presentation,
        layer_id=selected_layer_id,
        take_id=prior_presentation.selected_take_id,
    )
    active_playback_layer_id = _resolve_preserved_active_playback_layer_id(
        prior_presentation,
        current_presentation,
    )
    active_playback_take_id = _resolve_take_id(
        current_presentation,
        layer_id=active_playback_layer_id,
        take_id=prior_presentation.active_playback_take_id,
    )
    timeline.selection.selected_layer_id = selected_layer_id
    timeline.selection.selected_layer_ids = [selected_layer_id] if selected_layer_id is not None else []
    timeline.selection.selected_take_id = selected_take_id
    clear_selected_events(timeline)
    timeline.playback_target.layer_id = active_playback_layer_id
    timeline.playback_target.take_id = active_playback_take_id


def surface_new_take_rows(
    *,
    timeline: Timeline,
    prior_presentation: TimelinePresentation,
    current_presentation: TimelinePresentation,
) -> None:
    target = _new_take_target(prior_presentation, current_presentation)
    if target is None:
        return
    timeline.selection.selected_layer_id = target[0]
    timeline.selection.selected_layer_ids = [target[0]]
    timeline.selection.selected_take_id = target[1]
    clear_selected_events(timeline)


def resolve_event_clip_preview(
    presentation: TimelinePresentation,
    *,
    layer_id: LayerId | str,
    take_id: TakeId | str | None,
    event_id: EventId | str,
) -> EventClipPreview:
    for layer in presentation.layers:
        if layer.layer_id != layer_id:
            continue
        if take_id in (None, layer.main_take_id):
            for event in layer.events:
                if event.event_id == event_id:
                    return _build_event_clip_preview(
                        presentation,
                        layer=layer,
                        take=None,
                        event=event,
                    )
        for take in layer.takes:
            if take.take_id != take_id:
                continue
            for event in take.events:
                if event.event_id == event_id:
                    return _build_event_clip_preview(
                        presentation,
                        layer=layer,
                        take=take,
                        event=event,
                    )
    raise ValueError(f"Unknown event clip preview target: {event_id}")


def _build_event_clip_preview(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
    event: EventPresentation,
) -> EventClipPreview:
    source_ref = _event_preview_source_ref(
        presentation,
        layer=layer,
        take=take,
    )
    if not source_ref:
        raise RuntimeError("Selected event does not have a source audio clip.")
    return EventClipPreview(
        source_ref=source_ref,
        start_seconds=float(event.start),
        end_seconds=float(event.end),
        gain_db=float(layer.gain_db),
    )


def _event_preview_source_ref(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
) -> str | None:
    direct_source_ref = (
        (
            take.playback_source_ref
            or take.source_audio_path
            or layer.playback_source_ref
            or layer.source_audio_path
        )
        if take is not None
        else (layer.playback_source_ref or layer.source_audio_path)
    )
    if direct_source_ref:
        return direct_source_ref
    source_layer_id = getattr(layer.status, "source_layer_id", "")
    if not source_layer_id:
        return None
    for candidate in presentation.layers:
        if str(candidate.layer_id) != str(source_layer_id):
            continue
        return candidate.playback_source_ref or candidate.source_audio_path
    return None


def _new_take_target(
    prior_presentation: TimelinePresentation,
    current_presentation: TimelinePresentation,
) -> tuple[LayerId, TakeId] | None:
    prior_take_ids_by_layer = {
        layer.layer_id: {take.take_id for take in layer.takes}
        for layer in prior_presentation.layers
    }
    candidates: list[tuple[LayerId, TakeId, str]] = []
    for layer in current_presentation.layers:
        prior_take_ids = prior_take_ids_by_layer.get(layer.layer_id, set())
        new_take_rows = [take for take in layer.takes if take.take_id not in prior_take_ids]
        if not new_take_rows:
            continue
        candidates.append(
            (
                layer.layer_id,
                new_take_rows[-1].take_id,
                str(layer.status.source_layer_id or ""),
            )
        )
    if not candidates:
        return None

    selected_layer_id = (
        str(prior_presentation.selected_layer_id)
        if prior_presentation.selected_layer_id is not None
        else ""
    )
    for layer_id, take_id, _source_layer_id in candidates:
        if str(layer_id) == selected_layer_id:
            return layer_id, take_id
    for layer_id, take_id, source_layer_id in candidates:
        if source_layer_id and source_layer_id == selected_layer_id:
            return layer_id, take_id

    layer_id, take_id, _source_layer_id = candidates[0]
    return layer_id, take_id


def _resolve_preserved_selected_layer_id(
    prior_presentation: TimelinePresentation,
    current_presentation: TimelinePresentation,
) -> LayerId | None:
    if prior_presentation.selected_layer_id is not None and _has_layer(
        current_presentation,
        prior_presentation.selected_layer_id,
    ):
        return prior_presentation.selected_layer_id
    return current_presentation.selected_layer_id


def _resolve_preserved_active_playback_layer_id(
    prior_presentation: TimelinePresentation,
    current_presentation: TimelinePresentation,
) -> LayerId | None:
    if prior_presentation.active_playback_layer_id is not None and _has_layer(
        current_presentation,
        prior_presentation.active_playback_layer_id,
    ):
        return prior_presentation.active_playback_layer_id
    if current_presentation.active_playback_layer_id is not None:
        return current_presentation.active_playback_layer_id
    return current_presentation.selected_layer_id


def _resolve_take_id(
    presentation: TimelinePresentation,
    *,
    layer_id: LayerId | None,
    take_id: TakeId | None,
) -> TakeId | None:
    if layer_id is None or take_id is None:
        return None
    for layer in presentation.layers:
        if layer.layer_id != layer_id:
            continue
        if layer.main_take_id == take_id:
            return take_id
        if any(take.take_id == take_id for take in layer.takes):
            return take_id
        return None
    return None


def _has_layer(presentation: TimelinePresentation, layer_id: LayerId) -> bool:
    return any(layer.layer_id == layer_id for layer in presentation.layers)
