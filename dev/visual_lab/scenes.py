"""Visual Lab scene compatibility API.
Exists so older lab imports resolve to current-state assembled presentation data.
The implementation delegates to current production timeline models and assembler.
"""

from __future__ import annotations

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation

from dev.visual_lab.current_state import (
    build_current_visual_lab_presentation,
    current_layer_by_id,
)


def visual_lab_layer_by_id(layer_id: str) -> LayerPresentation:
    """Return one current Visual Lab layer by string id."""

    return current_layer_by_id(layer_id)


def build_visual_lab_presentation() -> TimelinePresentation:
    """Build the Visual Lab presentation through the current app timeline contract."""

    return build_current_visual_lab_presentation()
