from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush

from echozero.application.presentation.models import EventPresentation


@dataclass(slots=True)
class EventLanePresentation:
    layer_id: object
    events: list[EventPresentation]
    pixels_per_second: float
    scroll_x: float
    header_width: int
    event_height: int = 22
    dimmed: bool = False


class EventLaneBlock:
    def paint(self, painter: QPainter, top_y: int, presentation: EventLanePresentation) -> list[tuple[QRectF, object, object]]:
        rects: list[tuple[QRectF, object, object]] = []
        for event in presentation.events:
            pps = max(1.0, presentation.pixels_per_second)
            x = presentation.header_width + (event.start * pps) - presentation.scroll_x
            width = max(10.0, (event.duration * pps))
            rect = QRectF(x, top_y, width, presentation.event_height)
            rects.append((rect, presentation.layer_id, event.event_id))

            color = QColor(event.color or '#57a0ff')
            if presentation.dimmed:
                color.setAlpha(120)
            if event.is_selected:
                color = color.lighter(130)
            painter.setPen(QPen(color.darker(160), 2 if event.is_selected else 1))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(rect, 5, 5)
            painter.setPen(QColor('#0b1220'))
            painter.drawText(QRectF(x + 6, top_y, max(0, width - 12), presentation.event_height), Qt.AlignmentFlag.AlignVCenter, event.label)
        return rects
