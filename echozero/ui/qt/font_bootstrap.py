from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

_BOOTSTRAP_PROPERTY = "echozero.qt.font_bootstrap.done"


class _ReadableFont(Protocol):
    """Minimal font contract needed to preserve the current point size."""

    def pointSize(self) -> int: ...


class _FontApplication(Protocol):
    """Minimal Qt app surface required for font bootstrap."""

    def property(self, name: str) -> object | None: ...

    def setProperty(self, name: str, value: object) -> None: ...

    def font(self) -> _ReadableFont: ...

    def setFont(self, font: object) -> None: ...


def _default_font_candidates() -> list[Path]:
    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    fonts_root = windows_root / "Fonts"
    return [
        fonts_root / "segoeui.ttf",
        fonts_root / "arial.ttf",
        fonts_root / "tahoma.ttf",
    ]


def _resolve_font_application(app: _FontApplication | None) -> _FontApplication | None:
    if app is not None:
        return app
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return cast(_FontApplication, instance)
    return None


def ensure_qt_fonts_available(
    app: _FontApplication | None = None,
    *,
    candidates: Sequence[Path] | None = None,
) -> list[str]:
    """Ensure at least one readable font family exists for headless Qt rendering."""

    qt_app = _resolve_font_application(app)
    if qt_app is None:
        return []

    existing = list(QFontDatabase.families())
    if existing:
        return existing

    if bool(qt_app.property(_BOOTSTRAP_PROPERTY)):
        return list(QFontDatabase.families())

    for candidate in candidates or _default_font_candidates():
        if not candidate.exists():
            continue

        font_id = QFontDatabase.addApplicationFont(str(candidate))
        if font_id < 0:
            continue

        loaded_families = QFontDatabase.applicationFontFamilies(font_id)
        if loaded_families:
            current_font = qt_app.font()
            point_size = current_font.pointSize() if current_font.pointSize() > 0 else 9
            qt_app.setFont(QFont(loaded_families[0], point_size))
            break

    qt_app.setProperty(_BOOTSTRAP_PROPERTY, True)
    return list(QFontDatabase.families())
