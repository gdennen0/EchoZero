from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from echozero.ui.style import SHELL_TOKENS
from echozero.ui.style.qt import ensure_qt_theme_installed
from echozero.ui.style.qt.qss import (
    build_echozero_app_qss,
    build_echozero_shell_qss,
    build_foundry_surface_qss,
    build_object_info_panel_qss,
)
from echozero.ui.qt.timeline.widget import TimelineWidget
from tests.ui.test_timeline_shell import _selection_test_presentation


def test_object_info_panel_qss_builder_uses_shared_shell_tokens():
    qss = build_object_info_panel_qss()

    assert SHELL_TOKENS.panel_bg in qss
    assert SHELL_TOKENS.text_primary in qss
    assert f"border-radius: {SHELL_TOKENS.scales.panel_radius}px;" in qss


def test_timeline_widget_object_palette_stylesheet_uses_shared_builder():
    app = QApplication.instance() or QApplication([])
    ensure_qt_theme_installed(app)
    widget = TimelineWidget(_selection_test_presentation())
    try:
        assert app.styleSheet() == build_echozero_app_qss()
        assert widget._object_info_panel.styleSheet() == ""
        assert widget._object_info_panel.layout().spacing() == SHELL_TOKENS.scales.layout_gap
        assert widget._scroll.styleSheet() == ""
    finally:
        widget.close()
        app.processEvents()


def test_echozero_shell_qss_builder_uses_shared_shell_tokens():
    qss = build_echozero_shell_qss()

    assert SHELL_TOKENS.window_bg in qss
    assert SHELL_TOKENS.control_bg in qss
    assert f"border-radius: {SHELL_TOKENS.scales.panel_radius}px;" in qss


def test_foundry_surface_qss_builder_stays_foundry_specific():
    qss = build_foundry_surface_qss()

    assert "QWidget#foundryRoot" in qss
    assert "QLabel#foundryStatusLine" in qss
    assert "QGroupBox" not in qss


def test_foundry_window_uses_shared_shell_stylesheet(tmp_path: Path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this environment")

    from echozero.foundry.ui import FoundryWindow

    app = QApplication.instance() or QApplication([])
    ensure_qt_theme_installed(app)
    window = FoundryWindow(tmp_path)
    try:
        assert app.styleSheet() == build_echozero_app_qss()
        assert build_echozero_shell_qss().strip() in app.styleSheet()
        assert build_foundry_surface_qss().strip() in app.styleSheet()
        assert window.styleSheet() == ""
        assert window.centralWidget().objectName() == "foundryRoot"
        assert window.status_line.objectName() == "foundryStatusLine"
        assert window.centralWidget().layout().spacing() == SHELL_TOKENS.scales.layout_gap
    finally:
        window.close()
        app.processEvents()
