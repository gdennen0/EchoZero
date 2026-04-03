from __future__ import annotations

from dataclasses import dataclass

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
        seconds = int(max(10, (rect.width() - layout.header_width) / pps))
        for second in range(seconds + 2):
            x = layout.header_width + int(second * pps) - int(presentation.scroll_x)
            if x < layout.header_width:
                continue
            painter.setPen(QPen(QColor('#3b4352'), 1))
            painter.drawLine(x, int(rect.bottom()) - 10, x, int(rect.bottom()))
            painter.setPen(QColor('#b8c0cc'))
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()) - 1)
            painter.drawText(x + 4, int(rect.top()) + 12, f'{second}')
