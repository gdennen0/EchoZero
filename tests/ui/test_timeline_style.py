from PyQt6.QtWidgets import QApplication

from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock
from echozero.ui.qt.timeline.blocks.layer_header import LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.ruler import RulerBlock
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
from echozero.ui.qt.timeline.blocks.transport_bar_block import TransportBarBlock
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.style import (
    TIMELINE_STYLE,
    build_object_palette_stylesheet,
    build_timeline_scroll_area_stylesheet,
    fixture_color,
    fixture_take_action_label,
)
from echozero.ui.qt.timeline.widget import ObjectInfoPanel, TimelineWidget


def test_object_palette_stylesheet_uses_shared_tokens():
    stylesheet = build_object_palette_stylesheet()

    assert TIMELINE_STYLE.object_palette.background_hex in stylesheet
    assert TIMELINE_STYLE.object_palette.border_hex in stylesheet
    assert TIMELINE_STYLE.object_palette.title_object_name in stylesheet
    assert TIMELINE_STYLE.object_palette.body_object_name in stylesheet


def test_object_info_panel_layout_comes_from_style_module():
    app = QApplication.instance() or QApplication([])
    panel = ObjectInfoPanel()
    try:
        margins = panel.layout().contentsMargins()
        style = TIMELINE_STYLE.object_palette

        assert panel.objectName() == style.frame_object_name
        assert margins.left() == style.content_padding.left
        assert margins.top() == style.content_padding.top
        assert margins.right() == style.content_padding.right
        assert margins.bottom() == style.content_padding.bottom
        assert panel.layout().spacing() == style.section_spacing_px
    finally:
        panel.close()
        app.processEvents()


def test_timeline_scroll_area_stylesheet_uses_shared_tokens():
    stylesheet = build_timeline_scroll_area_stylesheet()

    assert TIMELINE_STYLE.scroll_area_background_hex in stylesheet
    assert "border: none;" in stylesheet


def test_timeline_blocks_default_to_shared_style_tokens():
    assert TransportBarBlock().style is TIMELINE_STYLE.transport_bar
    assert LayerHeaderBlock().style is TIMELINE_STYLE.layer_header
    assert TakeRowBlock().style is TIMELINE_STYLE.take_row
    assert EventLaneBlock().style is TIMELINE_STYLE.event_lane

    ruler = RulerBlock()
    assert ruler.style is TIMELINE_STYLE.ruler
    assert ruler.playhead_color_hex == TIMELINE_STYLE.playhead.color_hex


def test_timeline_widget_shell_uses_shared_style_tokens():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(build_demo_app().presentation())
    try:
        assert widget.windowTitle() == TIMELINE_STYLE.window_title
        assert widget._canvas._style is TIMELINE_STYLE
        assert build_timeline_scroll_area_stylesheet(widget._style) == widget._scroll.styleSheet()
    finally:
        widget.close()
        app.processEvents()


def test_timeline_fixture_tokens_are_discoverable_from_style_module():
    assert fixture_color("song") == TIMELINE_STYLE.fixture.layer_color_tokens["song"]
    assert fixture_color("sync") == TIMELINE_STYLE.fixture.layer_color_tokens["sync"]
    assert fixture_take_action_label("overwrite_main") == "Overwrite Main"
    assert fixture_take_action_label("merge_main") == "Merge Main"
