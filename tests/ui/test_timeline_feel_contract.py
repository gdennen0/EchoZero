from PyQt6.QtWidgets import QApplication

from echozero.ui.FEEL import (
    EVENT_BAR_HEIGHT_PX,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
    LAYER_ROW_HEIGHT_PX,
    TAKE_ROW_HEIGHT_PX,
)
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.widget import TimelineCanvas


def test_timeline_canvas_dimensions_are_sourced_from_feel():
    app = QApplication.instance() or QApplication([])
    canvas = TimelineCanvas(build_demo_app().presentation())
    try:
        assert canvas._header_width == LAYER_HEADER_WIDTH_PX
        assert canvas._top_padding == LAYER_HEADER_TOP_PADDING_PX
        assert canvas._main_row_height == LAYER_ROW_HEIGHT_PX
        assert canvas._take_row_height == TAKE_ROW_HEIGHT_PX
        assert canvas._event_height == EVENT_BAR_HEIGHT_PX
    finally:
        canvas.close()
