from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import (
    LayerPresentation,
    LayerStatusPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    default_take_actions,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId, TakeId
from echozero.ui.qt.timeline.blocks.layouts import TakeRowLayout
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock


def _take(actions=None) -> TakeLanePresentation:
    return TakeLanePresentation(
        take_id=TakeId("take_test"),
        name="Take 2",
        kind=LayerKind.EVENT,
        actions=actions or [],
    )


def test_take_row_default_actions_include_overwrite_and_merge():
    block = TakeRowBlock()
    actions = block._actions_for_take(_take())
    action_ids = {action.action_id for action in actions}

    assert {"overwrite_main", "merge_main", "delete_take"} <= action_ids


def test_take_row_uses_explicit_take_actions_when_provided():
    block = TakeRowBlock()
    custom = [TakeActionPresentation(action_id="custom_action", label="Custom")]
    actions = block._actions_for_take(_take(actions=custom))

    assert len(actions) == 1
    assert actions[0].action_id == "custom_action"


def test_take_row_layout_reserves_options_area():
    layout = TakeRowLayout.create(top=10, width=900, header_width=320, row_height=44)

    assert layout.options_button_rect.width() > 0
    assert layout.options_area_rect.width() > 0
    assert layout.options_area_rect.top() > layout.label_rect.top()


def test_take_row_renders_selected_take_actions_without_truncating_fourth_chip():
    app = QApplication.instance() or QApplication([])
    image = QImage(320, 44, QImage.Format.Format_ARGB32)
    block = TakeRowBlock()
    layout = TakeRowLayout.create(top=0, width=320, header_width=320, row_height=44)
    painter = QPainter(image)
    try:
        hit_targets = block.paint_header(
            painter,
            layout,
            LayerPresentation(
                layer_id=LayerId("layer_test"),
                title="Kick",
                main_take_id=TakeId("take_main"),
                status=LayerStatusPresentation(),
            ),
            _take(actions=default_take_actions(has_selection=True)),
            options_open=True,
        )
    finally:
        painter.end()
        app.processEvents()

    assert len(hit_targets.action_rects) == 4
