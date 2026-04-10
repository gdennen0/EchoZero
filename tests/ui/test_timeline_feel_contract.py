from dataclasses import replace

from PyQt6.QtWidgets import QApplication

from echozero.ui.FEEL import (
    EVENT_BAR_HEIGHT_PX,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
    LAYER_ROW_HEIGHT_PX,
    RULER_HEIGHT_PX,
    TAKE_ROW_HEIGHT_PX,
    TIMELINE_RIGHT_PADDING_PX,
    TIMELINE_ZOOM_MAX_PPS,
    TIMELINE_ZOOM_MIN_PPS,
    TIMELINE_ZOOM_STEP_FACTOR,
)
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.widget import (
    TimelineCanvas,
    TimelineRuler,
    TimelineWidget,
    compute_scroll_bounds,
    estimate_timeline_span_seconds,
)


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


def test_timeline_ruler_fixed_height_is_sourced_from_feel():
    app = QApplication.instance() or QApplication([])
    ruler = TimelineRuler(build_demo_app().presentation())
    try:
        assert ruler.minimumHeight() == RULER_HEIGHT_PX
        assert ruler.maximumHeight() == RULER_HEIGHT_PX
    finally:
        ruler.close()


def test_compute_scroll_bounds_default_right_padding_is_sourced_from_feel():
    presentation = build_demo_app().presentation()
    viewport_width = 900

    content_width, max_scroll = compute_scroll_bounds(presentation, viewport_width=viewport_width)
    span = estimate_timeline_span_seconds(presentation)
    expected_content_width = max(
        viewport_width,
        int(
            LAYER_HEADER_WIDTH_PX
            + (presentation.pixels_per_second * span)
            + TIMELINE_RIGHT_PADDING_PX
        ),
    )

    assert content_width == expected_content_width
    assert max_scroll == expected_content_width - viewport_width


def test_timeline_zoom_in_clamps_to_feel_max_pps():
    app = QApplication.instance() or QApplication([])
    presentation = build_demo_app().presentation()
    widget = TimelineWidget(
        presentation,
    )
    try:
        widget.set_presentation(
            replace(
                widget.presentation,
                pixels_per_second=TIMELINE_ZOOM_MAX_PPS / TIMELINE_ZOOM_STEP_FACTOR,
            )
        )
        widget._zoom_from_input(120, anchor_x=widget._canvas._header_width + 120.0)

        assert widget.presentation.pixels_per_second == TIMELINE_ZOOM_MAX_PPS
    finally:
        widget.close()


def test_timeline_zoom_out_clamps_to_feel_min_pps():
    app = QApplication.instance() or QApplication([])
    presentation = build_demo_app().presentation()
    widget = TimelineWidget(
        presentation,
    )
    try:
        widget.set_presentation(
            replace(
                widget.presentation,
                pixels_per_second=TIMELINE_ZOOM_MIN_PPS * TIMELINE_ZOOM_STEP_FACTOR,
            )
        )
        widget._zoom_from_input(-120, anchor_x=widget._canvas._header_width + 120.0)

        assert widget.presentation.pixels_per_second == TIMELINE_ZOOM_MIN_PPS
    finally:
        widget.close()
