from __future__ import annotations

from typing import Protocol, cast

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from echozero.ui.style.tokens import SHELL_TOKENS, ShellTokens

from .qss import build_echozero_app_qss

_THEME_PROPERTY = "echozero.qt.theme_bootstrap.done"


class _ThemeApplication(Protocol):
    """Minimal Qt app surface required to install the shared theme."""

    def property(self, name: str) -> object | None: ...

    def setProperty(self, name: str, value: object) -> None: ...

    def setPalette(self, palette: QPalette) -> None: ...

    def setStyleSheet(self, stylesheet: str) -> None: ...


def build_echozero_palette(tokens: ShellTokens = SHELL_TOKENS) -> QPalette:
    """Build the canonical application palette for standard Qt widgets."""

    palette = QPalette()

    def _set_all(role: QPalette.ColorRole, color: str) -> None:
        value = QColor(color)
        palette.setColor(QPalette.ColorGroup.Active, role, value)
        palette.setColor(QPalette.ColorGroup.Inactive, role, value)

    def _set_disabled(role: QPalette.ColorRole, color: str) -> None:
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(color))

    _set_all(QPalette.ColorRole.Window, tokens.window_bg)
    _set_all(QPalette.ColorRole.WindowText, tokens.text_primary)
    _set_all(QPalette.ColorRole.Base, tokens.panel_bg)
    _set_all(QPalette.ColorRole.AlternateBase, tokens.panel_alt_bg)
    _set_all(QPalette.ColorRole.ToolTipBase, tokens.panel_alt_bg)
    _set_all(QPalette.ColorRole.ToolTipText, tokens.text_primary)
    _set_all(QPalette.ColorRole.Text, tokens.text_primary)
    _set_all(QPalette.ColorRole.Button, tokens.control_bg)
    _set_all(QPalette.ColorRole.ButtonText, tokens.control_text)
    _set_all(QPalette.ColorRole.BrightText, tokens.text_primary)
    _set_all(QPalette.ColorRole.Highlight, tokens.control_bg_active)
    _set_all(QPalette.ColorRole.HighlightedText, tokens.text_primary)
    _set_all(QPalette.ColorRole.PlaceholderText, tokens.text_secondary)
    _set_all(QPalette.ColorRole.Light, tokens.panel_alt_bg)
    _set_all(QPalette.ColorRole.Midlight, tokens.panel_border)
    _set_all(QPalette.ColorRole.Mid, tokens.section_border)
    _set_all(QPalette.ColorRole.Dark, tokens.panel_border)
    _set_all(QPalette.ColorRole.Shadow, tokens.window_bg)

    _set_disabled(QPalette.ColorRole.WindowText, tokens.control_text_disabled)
    _set_disabled(QPalette.ColorRole.Text, tokens.control_text_disabled)
    _set_disabled(QPalette.ColorRole.ButtonText, tokens.control_text_disabled)
    _set_disabled(QPalette.ColorRole.HighlightedText, tokens.control_text_disabled)
    _set_disabled(QPalette.ColorRole.Button, tokens.control_bg_disabled)
    _set_disabled(QPalette.ColorRole.Base, tokens.control_bg_disabled)
    _set_disabled(QPalette.ColorRole.AlternateBase, tokens.panel_bg)
    _set_disabled(QPalette.ColorRole.PlaceholderText, tokens.control_text_disabled)
    return palette


def _resolve_theme_application(app: _ThemeApplication | None) -> _ThemeApplication | None:
    if app is not None:
        return app
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return cast(_ThemeApplication, instance)
    return None


def ensure_qt_theme_installed(app: _ThemeApplication | None = None) -> str:
    """Install the canonical EchoZero palette and stylesheet onto the Qt app once."""

    qt_app = _resolve_theme_application(app)
    if qt_app is None:
        return ""

    stylesheet = build_echozero_app_qss()
    if bool(qt_app.property(_THEME_PROPERTY)):
        return stylesheet

    qt_app.setPalette(build_echozero_palette())
    qt_app.setStyleSheet(stylesheet)
    qt_app.setProperty(_THEME_PROPERTY, True)
    return stylesheet
