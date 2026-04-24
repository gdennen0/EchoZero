from __future__ import annotations

from echozero.ui.style import SHELL_TOKENS
from echozero.ui.style.qt import build_echozero_palette, ensure_qt_theme_installed
from echozero.ui.style.qt.qss import build_echozero_app_qss


class _FakeApp:
    def __init__(self) -> None:
        self._props: dict[str, object] = {}
        self.palette_calls: list[object] = []
        self.stylesheet_calls: list[str] = []
        self._stylesheet = ""

    def property(self, name: str) -> object | None:
        return self._props.get(name)

    def setProperty(self, name: str, value: object) -> None:
        self._props[name] = value

    def setPalette(self, palette) -> None:
        self.palette_calls.append(palette)

    def setStyleSheet(self, stylesheet: str) -> None:
        self.stylesheet_calls.append(stylesheet)
        self._stylesheet = stylesheet

    def styleSheet(self) -> str:
        return self._stylesheet


def test_build_echozero_palette_uses_shared_shell_tokens():
    palette = build_echozero_palette()

    assert palette.color(palette.ColorGroup.Active, palette.ColorRole.Window).name() == (
        SHELL_TOKENS.window_bg
    )
    assert palette.color(palette.ColorGroup.Active, palette.ColorRole.Button).name() == (
        SHELL_TOKENS.control_bg
    )
    assert palette.color(palette.ColorGroup.Active, palette.ColorRole.Highlight).name() == (
        SHELL_TOKENS.control_bg_active
    )
    assert palette.color(palette.ColorGroup.Disabled, palette.ColorRole.ButtonText).name() == (
        SHELL_TOKENS.control_text_disabled
    )


def test_ensure_qt_theme_installed_sets_palette_and_stylesheet_once():
    app = _FakeApp()

    stylesheet = ensure_qt_theme_installed(app)

    assert stylesheet == build_echozero_app_qss()
    assert app.styleSheet() == build_echozero_app_qss()
    assert len(app.palette_calls) == 1
    assert app.stylesheet_calls == [build_echozero_app_qss()]
    assert app.property("echozero.qt.theme_bootstrap.done") is True

    second = ensure_qt_theme_installed(app)

    assert second == build_echozero_app_qss()
    assert len(app.palette_calls) == 1
    assert app.stylesheet_calls == [build_echozero_app_qss()]
