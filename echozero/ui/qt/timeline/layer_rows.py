"""Timeline row ordering helpers for layer hierarchy presentation.
Exists to keep parent/child display grouping explicit and reusable across canvas sizing,
painting, hit testing, and UI tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from echozero.application.presentation.models import LayerPresentation
from echozero.application.shared.ids import LayerId


@dataclass(frozen=True, slots=True)
class TimelineLayerRow:
    layer: LayerPresentation
    depth: int = 0
    has_child_layers: bool = False


def build_timeline_layer_rows(layers: Iterable[LayerPresentation]) -> list[TimelineLayerRow]:
    """Return visible layer rows with children nested under their parent.

    The canonical presentation remains flat. This function is only the display-order
    projection used by timeline surfaces that render rows.
    """

    ordered_layers = list(layers)
    layer_ids = {layer.layer_id for layer in ordered_layers}
    children_by_parent: dict[LayerId, list[LayerPresentation]] = {}
    for layer in ordered_layers:
        parent_layer_id = layer.parent_layer_id
        if parent_layer_id is None or parent_layer_id not in layer_ids:
            continue
        children_by_parent.setdefault(parent_layer_id, []).append(layer)

    rows: list[TimelineLayerRow] = []
    emitted: set[LayerId] = set()

    def append_layer(layer: LayerPresentation, *, depth: int) -> None:
        if layer.layer_id in emitted:
            return
        emitted.add(layer.layer_id)
        children = children_by_parent.get(layer.layer_id, [])
        rows.append(
            TimelineLayerRow(
                layer=layer,
                depth=depth,
                has_child_layers=bool(children),
            )
        )
        if not layer.is_expanded or layer.is_fully_collapsed:
            return
        for child in children:
            append_layer(child, depth=depth + 1)

    for layer in ordered_layers:
        parent_layer_id = layer.parent_layer_id
        if parent_layer_id is not None and parent_layer_id in layer_ids:
            continue
        append_layer(layer, depth=0)

    for layer in ordered_layers:
        parent_layer_id = layer.parent_layer_id
        if parent_layer_id is not None and parent_layer_id in layer_ids:
            continue
        append_layer(layer, depth=0)

    return rows
