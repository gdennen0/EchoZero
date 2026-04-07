from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from echozero.ui.style import SHELL_TOKENS
from echozero.ui.style.qt.qss import build_foundry_shell_qss, build_object_info_panel_qss
from echozero.ui.qt.timeline.widget import TimelineWidget
from tests.ui.test_timeline_shell import _palette_test_presentation


def test_object_info_panel_qss_builder_uses_shared_shell_tokens():
    qss = build_object_info_panel_qss()

    assert SHELL_TOKENS.panel_bg in qss
    assert SHELL_TOKENS.text_primary in qss
    assert f"border-radius: {SHELL_TOKENS.scales.panel_radius}px;" in qss


def test_timeline_widget_object_palette_stylesheet_uses_shared_builder():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_palette_test_presentation())
    try:
        assert widget._object_info_panel.styleSheet() == build_object_info_panel_qss()
        assert widget._object_info_panel.layout().spacing() == SHELL_TOKENS.scales.layout_gap
        assert widget._scroll.styleSheet() == f"background: {SHELL_TOKENS.canvas_bg}; border: none;"
    finally:
        widget.close()
        app.processEvents()


def test_foundry_shell_qss_builder_uses_shared_shell_tokens():
    qss = build_foundry_shell_qss()

    assert SHELL_TOKENS.window_bg in qss
    assert SHELL_TOKENS.control_bg in qss
    assert f"border-radius: {SHELL_TOKENS.scales.panel_radius}px;" in qss


def test_foundry_window_uses_shared_shell_stylesheet(tmp_path: Path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this environment")

    from echozero.foundry.ui import FoundryWindow

    app = QApplication.instance() or QApplication([])
    window = FoundryWindow(tmp_path)
    try:
        assert window.styleSheet() == build_foundry_shell_qss()
        assert window.centralWidget().objectName() == "foundryRoot"
        assert window.status_line.objectName() == "foundryStatusLine"
        assert window.centralWidget().layout().spacing() == SHELL_TOKENS.scales.layout_gap
    finally:
        window.close()
        app.processEvents()
