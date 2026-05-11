from __future__ import annotations

from echozero.application.presentation.models import LayerPresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId
from echozero.ui.qt.timeline.layer_rows import build_timeline_layer_rows


def test_timeline_layer_rows_hide_child_layers_until_parent_expands() -> None:
    parent = _layer("source", "Imported Song", expanded=False)
    child = _layer("drums", "Drums", parent_layer_id=parent.layer_id)
    sibling = _layer("markers", "Markers", kind=LayerKind.EVENT)

    rows = build_timeline_layer_rows([parent, child, sibling])

    assert [(row.layer.layer_id, row.depth, row.has_child_layers) for row in rows] == [
        (parent.layer_id, 0, True),
        (sibling.layer_id, 0, False),
    ]


def test_timeline_layer_rows_nest_child_layers_under_expanded_parent() -> None:
    parent = _layer("source", "Imported Song", expanded=True)
    drums = _layer("drums", "Drums", parent_layer_id=parent.layer_id)
    bass = _layer("bass", "Bass", parent_layer_id=parent.layer_id)
    sibling = _layer("markers", "Markers", kind=LayerKind.EVENT)

    rows = build_timeline_layer_rows([parent, drums, sibling, bass])

    assert [(row.layer.layer_id, row.depth, row.has_child_layers) for row in rows] == [
        (parent.layer_id, 0, True),
        (drums.layer_id, 1, False),
        (bass.layer_id, 1, False),
        (sibling.layer_id, 0, False),
    ]


def test_timeline_layer_rows_preserve_orphaned_child_rows() -> None:
    orphan = _layer("drums", "Drums", parent_layer_id=LayerId("missing"))

    rows = build_timeline_layer_rows([orphan])

    assert [(row.layer.layer_id, row.depth, row.has_child_layers) for row in rows] == [
        (orphan.layer_id, 0, False),
    ]


def _layer(
    layer_id: str,
    title: str,
    *,
    kind: LayerKind = LayerKind.AUDIO,
    expanded: bool = False,
    parent_layer_id: LayerId | None = None,
) -> LayerPresentation:
    return LayerPresentation(
        layer_id=LayerId(layer_id),
        title=title,
        kind=kind,
        is_expanded=expanded,
        parent_layer_id=parent_layer_id,
    )
