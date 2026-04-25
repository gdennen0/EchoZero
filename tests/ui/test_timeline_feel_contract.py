from dataclasses import replace

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from echozero.application.shared.enums import LayerKind
from echozero.ui.FEEL import (
    EVENT_BAR_HEIGHT_PX,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
    LAYER_ROW_HEIGHT_PX,
    RULER_HEIGHT_PX,
    TIMELINE_RIGHT_PADDING_PX,
    TIMELINE_ZOOM_MAX_PPS,
    TIMELINE_ZOOM_MIN_PPS,
    TIMELINE_ZOOM_STEP_FACTOR,
)
from echozero.ui.qt.timeline.blocks.layouts import MainRowLayout
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.layer_height_config import timeline_layer_height_config
from echozero.ui.qt.timeline.widget import (
    TimelineCanvas,
    TimelineRuler,
    TimelineWidget,
    compute_scroll_bounds,
    estimate_timeline_span_seconds,
)


def _mouse_drag(target, points: list[QPoint]) -> None:
    first = points[0]
    QApplication.sendEvent(
        target,
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(first),
            QPointF(first),
            QPointF(first),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    for point in points[1:]:
        QApplication.sendEvent(
            target,
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(point),
                QPointF(point),
                QPointF(point),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    last = points[-1]
    QApplication.sendEvent(
        target,
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(last),
            QPointF(last),
            QPointF(last),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    QApplication.processEvents()


def test_timeline_canvas_dimensions_include_layer_height_config_defaults():
    app = QApplication.instance() or QApplication([])
    height_config = timeline_layer_height_config()
    canvas = TimelineCanvas(build_demo_app().presentation())
    try:
        assert canvas._header_width == LAYER_HEADER_WIDTH_PX
        assert canvas._top_padding == LAYER_HEADER_TOP_PADDING_PX
        assert canvas._main_row_height == height_config.default_main_row_height_px
        assert canvas._take_row_height == height_config.take_row_height_px
        assert canvas._main_row_height < 72
        assert canvas._event_height == EVENT_BAR_HEIGHT_PX
    finally:
        canvas.close()


def test_timeline_canvas_uses_per_layer_type_default_row_heights():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    height_config = timeline_layer_height_config()
    presentation = build_demo_app().presentation()
    canvas = TimelineCanvas(presentation)
    try:
        audio_layer = next(layer for layer in presentation.layers if layer.kind is LayerKind.AUDIO)
        event_layer = next(layer for layer in presentation.layers if layer.kind is LayerKind.EVENT)

        assert canvas._main_row_height_for_layer(audio_layer) == (
            height_config.layer_kind_main_row_height_px[LayerKind.AUDIO]
        )
        assert canvas._main_row_height_for_layer(event_layer) == (
            height_config.layer_kind_main_row_height_px[LayerKind.EVENT]
        )
        assert canvas._main_row_height_for_layer(audio_layer) != canvas._main_row_height_for_layer(
            event_layer
        )
    finally:
        canvas.close()


def test_timeline_canvas_vertical_drag_resizes_main_layer_rows():
    app = QApplication.instance() or QApplication([])
    canvas = TimelineCanvas(build_demo_app().presentation())
    try:
        canvas.resize(1200, 440)
        canvas.show()
        canvas.repaint()
        app.processEvents()
        assert canvas._layer_row_resize_hit_rects

        resize_rect, layer_id = canvas._layer_row_resize_hit_rects[0]
        start_height = canvas._main_row_height_for_layer_id(layer_id)
        anchor = resize_rect.center().toPoint()
        _mouse_drag(
            canvas,
            [
                QPoint(anchor.x(), anchor.y()),
                QPoint(anchor.x(), anchor.y() + 20),
            ],
        )

        assert canvas._main_row_height_for_layer_id(layer_id) > start_height
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


def test_main_row_layout_keeps_expand_toggle_inside_header_width():
    layout = MainRowLayout.create(
        top=10,
        width=900,
        header_width=LAYER_HEADER_WIDTH_PX,
        row_height=LAYER_ROW_HEIGHT_PX,
    )

    assert layout.toggle_rect.right() <= layout.header_rect.right()


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
