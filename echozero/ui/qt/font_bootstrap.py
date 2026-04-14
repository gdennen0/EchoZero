from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

_BOOTSTRAP_PROPERTY = "echozero.qt.font_bootstrap.done"


def _default_font_candidates() -> list[Path]:
    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    fonts_root = windows_root / "Fonts"
    return [
        fonts_root / "segoeui.ttf",
        fonts_root / "arial.ttf",
        fonts_root / "tahoma.ttf",
    ]


def ensure_qt_fonts_available(
    app: QApplication | None = None,
    *,
    candidates: list[Path] | None = None,
) -> list[str]:
    """Ensure at least one readable font family exists for headless Qt rendering."""

    qt_app = app or QApplication.instance()
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
