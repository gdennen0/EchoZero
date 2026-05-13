"""Visual Lab style preview widgets.
Exists for lab-only theme objects that are not production UI components.
Imported by catalog entries without adding editor-only logic to production runtime.
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from dev.visual_lab.tokens import VisualLabTokens


class GlobalColorPalettePreviewWidget(QWidget):
    """Preview shared global color tokens as a compact palette strip."""

    def __init__(self, tokens: VisualLabTokens) -> None:
        super().__init__()
        self.tokens = tokens
        self.colors = (
            ("bg", tokens.global_colors.app_background),
            ("surface", tokens.global_colors.surface),
            ("raised", tokens.global_colors.surface_raised),
            ("text", tokens.global_colors.text_primary),
            ("muted", tokens.global_colors.text_secondary),
            ("primary", tokens.global_colors.primary),
            ("secondary", tokens.global_colors.secondary),
            ("success", tokens.global_colors.success),
            ("warning", tokens.global_colors.warning),
            ("error", tokens.global_colors.error),
        )
        self.setMinimumSize(620, 170)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.tokens.global_colors.app_background))
        x = 18.0
        y = 24.0
        for index, (label, color) in enumerate(self.colors):
            row = index // 5
            column = index % 5
            rect = QRectF(x + column * 118, y + row * 62, 96, 42)
            painter.setBrush(QColor(color))
            painter.setPen(QPen(QColor(self.tokens.palette.border), 1))
            painter.drawRoundedRect(
                rect,
                self.tokens.metrics.control_radius_px,
                self.tokens.metrics.control_radius_px,
            )
            painter.setPen(QColor(self.tokens.palette.text))
            painter.setFont(_font(self.tokens))
            painter.drawText(
                QRectF(rect.left(), rect.bottom() + 4, rect.width(), 16),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )


def _font(tokens: VisualLabTokens) -> QFont:
    font = QFont(tokens.fonts.family)
    font.setPixelSize(tokens.fonts.small_px)
    return font
