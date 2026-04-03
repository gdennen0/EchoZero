from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPainter, QPen

from echozero.application.presentation.models import TimelinePresentation


@dataclass(slots=True)
class RulerLayout:
    rect: QRectF
    header_width: float


class RulerBlock:
    def paint(self, painter: QPainter, layout: RulerLayout, presentation: TimelinePresentation) -> None:
        rect = layout.rect
        painter.fillRect(rect, QColor('#0f1318'))
        painter.fillRect(QRectF(rect.left(), rect.bottom() - 1, rect.width(), 1), QColor('#2a303c'))
        painter.fillRect(QRectF(rect.left(), rect.top(), layout.header_width, rect.height()), QColor('#171c23'))
        painter.setPen(QColor('#9aa4b2'))
        painter.drawText(14, int(rect.top()) + 18, 'Timeline')

        pps = max(1.0, presentation.pixels_per_second)
        content_width = max(1.0, rect.width() - layout.header_width)
        for second, x in visible_ruler_seconds(
            scroll_x=presentation.scroll_x,
            pixels_per_second=pps,
            content_width=content_width,
            content_start_x=layout.header_width,
        ):
            painter.setPen(QPen(QColor('#3b4352'), 1))
            painter.drawLine(int(x), int(rect.bottom()) - 10, int(x), int(rect.bottom()))
            painter.setPen(QColor('#b8c0cc'))
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()) - 1)
            painter.drawText(int(x) + 4, int(rect.top()) + 12, f'{second}')


def visible_ruler_seconds(
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_width: float,
    content_start_x: float,
) -> list[tuple[int, float]]:
    """Compute visible (second, screen_x) marks for the current horizontal viewport."""
    pps = max(1.0, pixels_per_second)
    start_second = max(0, int(floor(scroll_x / pps)) - 1)
    end_second = int(ceil((scroll_x + content_width) / pps)) + 1

    marks: list[tuple[int, float]] = []
    for second in range(start_second, max(start_second, end_second) + 1):
        x = content_start_x + (second * pps) - scroll_x
        if (content_start_x - pps) <= x <= (content_start_x + content_width + pps):
            marks.append((second, x))
    return marks
