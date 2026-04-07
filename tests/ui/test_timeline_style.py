from PyQt6.QtWidgets import QApplication

from echozero.ui.qt.timeline.style import (
    TIMELINE_STYLE,
    build_object_palette_stylesheet,
    build_timeline_scroll_area_stylesheet,
)
from echozero.ui.qt.timeline.widget import ObjectInfoPanel


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
