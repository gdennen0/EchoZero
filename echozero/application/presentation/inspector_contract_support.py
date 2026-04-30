"""Inspector fact-row helpers for the canonical presentation layer.
Exists to keep selection, playback, and transfer summary rows separate from contract assembly.
Connects timeline presentation state to reusable inspector labels and fact rows.
"""

from __future__ import annotations

from echozero.application.presentation.inspector_contract_lookup import (
    find_layer,
    find_selected_event,
    find_take,
)
from echozero.application.presentation.inspector_contract_types import (
    InspectorFactRow,
)
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)

_CLASSIFICATION_CONFIDENCE_KEYS = ("confidence", "classifier_score", "score", "probability")


def selection_playback_context_rows(
    presentation: TimelinePresentation,
) -> tuple[InspectorFactRow, ...]:
    selection_label = selected_identity_label(presentation)
    if selection_label == "none":
        return ()
    return (InspectorFactRow("selected identity", selection_label),)


def selected_identity_label(presentation: TimelinePresentation) -> str:
    if presentation.selected_event_ids:
        event_match = find_selected_event(presentation, presentation.selected_event_ids[0])
        if event_match is not None:
            layer, take, event = event_match
            take_label = take_identity_label(layer, take)
            return f"Event {event.label} ({event.event_id}) on {layer.title} / {take_label}"
    if presentation.selected_take_id is not None and presentation.selected_layer_id is not None:
        take_match = find_take(
            presentation,
            layer_id=presentation.selected_layer_id,
            take_id=presentation.selected_take_id,
        )
        if take_match is not None:
            layer, take = take_match
            return f"Take {take.name} ({take.take_id}) on {layer.title}"
        selected_layer = find_layer(presentation, presentation.selected_layer_id)
        if selected_layer is not None and presentation.selected_take_id == selected_layer.main_take_id:
            return f"Take Main take ({selected_layer.main_take_id}) on {selected_layer.title}"
    if presentation.selected_layer_id is not None:
        selected_layer = find_layer(presentation, presentation.selected_layer_id)
        if selected_layer is not None:
            return f"Layer {selected_layer.title} ({selected_layer.layer_id})"
    return "none"


def playback_state_label(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
) -> str:
    if presentation.selected_layer_id != layer.layer_id:
        return "Main Mix"
    selected_take_id = presentation.selected_take_id
    if take is None:
        return "Selected" if selected_take_id in (None, layer.main_take_id) else "Main Mix"
    return "Selected" if selected_take_id == take.take_id else "Main Mix"


def take_identity_label(layer: LayerPresentation, take: TakeLanePresentation | None) -> str:
    if take is None:
        return f"Main take ({layer.main_take_id or 'none'})"
    return f"{take.name} ({take.take_id})"


def sync_state_label(layer: LayerPresentation) -> str:
    state = layer.live_sync_state.value.replace("_", " ")
    return state.title()


def layer_transfer_rows(
    presentation: TimelinePresentation,
    layer: LayerPresentation,
) -> tuple[InspectorFactRow, ...]:
    rows = [
        InspectorFactRow("sync state", sync_state_label(layer)),
        InspectorFactRow("sync mapping", layer.sync_target_label or "none"),
    ]
    return tuple(rows)


def event_classification_rows(event: EventPresentation) -> tuple[InspectorFactRow, ...]:
    return (
        InspectorFactRow("classification", event_classification_label(event)),
        InspectorFactRow("confidence score", event_confidence_score_label(event)),
    )


def event_classification_label(event: EventPresentation) -> str:
    if not event.classifications:
        return "none"
    for key in ("label", "class", "type"):
        label = _normalized_classification_value(event.classifications.get(key))
        if label is not None:
            return label
    for key, value in event.classifications.items():
        if key in _CLASSIFICATION_CONFIDENCE_KEYS:
            continue
        label = _normalized_classification_value(value)
        if label is not None:
            return label
        if isinstance(key, str) and key.strip():
            return _normalize_classification_key(key)
    return "none"


def event_confidence_score_label(event: EventPresentation) -> str:
    confidence = _first_confidence_value(event.classifications)
    if confidence is None:
        confidence = _first_confidence_value(event.detection_metadata)
    if confidence is None:
        return "n/a"
    numeric = _coerce_confidence_float(confidence)
    if numeric is not None:
        return _format_confidence_score(numeric)
    text = _normalized_freeform_text(confidence)
    return text if text is not None else "n/a"


def _first_confidence_value(values: dict[str, object]) -> object | None:
    for key in _CLASSIFICATION_CONFIDENCE_KEYS:
        candidate = values.get(key)
        if candidate not in (None, ""):
            return candidate
    for candidate in values.values():
        if isinstance(candidate, dict):
            nested = _first_confidence_value(candidate)
            if nested is not None:
                return nested
    return None


def _normalized_classification_value(value: object) -> str | None:
    text = _normalized_freeform_text(value)
    if text is None:
        return None
    return text.replace("_", " ").title()


def _normalize_classification_key(value: str) -> str:
    text = value.strip().replace("_", " ")
    if ":" in text:
        text = text.split(":", maxsplit=1)[1]
    return text.title() if text else "none"


def _normalized_freeform_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _coerce_confidence_float(value: object) -> float | None:
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


def _format_confidence_score(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
