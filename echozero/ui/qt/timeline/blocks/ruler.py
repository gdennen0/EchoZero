from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF

from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.timeline.style import RulerStyle, TIMELINE_STYLE


@dataclass(slots=True)
class RulerLayout:
    rect: QRectF
    header_width: float


class RulerBlock:
    def __init__(self, style: RulerStyle = TIMELINE_STYLE.ruler, *, playhead_color_hex: str = TIMELINE_STYLE.playhead.color_hex):
        self.style = style
        self.playhead_color_hex = playhead_color_hex

    def paint(self, painter: QPainter, layout: RulerLayout, presentation: TimelinePresentation) -> None:
        rect = layout.rect
        painter.fillRect(rect, QColor(self.style.background_hex))
        painter.fillRect(QRectF(rect.left(), rect.bottom() - 1, rect.width(), 1), QColor(self.style.divider_hex))
        painter.fillRect(QRectF(rect.left(), rect.top(), layout.header_width, rect.height()), QColor(self.style.header_background_hex))
        painter.setPen(QColor(self.style.title_hex))
        painter.drawText(14, int(rect.top()) + 18, 'Timeline')

        pps = max(1.0, presentation.pixels_per_second)
        content_width = max(1.0, rect.width() - layout.header_width)
        for second, x in visible_ruler_seconds(
            scroll_x=presentation.scroll_x,
            pixels_per_second=pps,
            content_width=content_width,
            content_start_x=layout.header_width,
        ):
            if x < layout.header_width or x > rect.right():
                continue
            painter.setPen(QPen(QColor(self.style.tick_hex), 1))
            painter.drawLine(int(x), int(rect.bottom()) - 10, int(x), int(rect.bottom()))
            painter.setPen(QColor(self.style.grid_hex))
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()) - 1)
            painter.setPen(QColor(self.style.label_hex))
            painter.drawText(int(x) + 4, int(rect.top()) + 12, f'{second}')

        playhead_x = timeline_x_for_time(
            presentation.playhead,
            scroll_x=presentation.scroll_x,
            pixels_per_second=pps,
            content_start_x=layout.header_width,
        )
        if layout.header_width <= playhead_x <= rect.right():
            head = playhead_head_polygon(playhead_x, rect.bottom() - 1)
            painter.setPen(QPen(QColor(self.playhead_color_hex), 1))
            painter.setBrush(QColor(self.playhead_color_hex))
            painter.drawPolygon(head)


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
        if content_start_x <= x <= (content_start_x + content_width):
            marks.append((second, x))
    return marks


def timeline_x_for_time(
    time_seconds: float,
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_start_x: float,
) -> float:
    pps = max(1.0, pixels_per_second)
    return content_start_x + (max(0.0, time_seconds) * pps) - scroll_x


def absolute_timeline_x_for_view_x(
    x: float,
    *,
    scroll_x: float,
    content_start_x: float,
) -> float:
    return max(0.0, x - content_start_x + scroll_x)


def seek_time_for_x(
    x: float,
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_start_x: float,
) -> float:
    pps = max(1.0, pixels_per_second)
    timeline_x = absolute_timeline_x_for_view_x(
        x,
        scroll_x=scroll_x,
        content_start_x=content_start_x,
    )
    return timeline_x / pps


def playhead_head_polygon(x: float, bottom_y: float) -> QPolygonF:
    return QPolygonF(
        [
            QPointF(x, bottom_y),
            QPointF(x - 7.0, bottom_y - 10.0),
            QPointF(x + 7.0, bottom_y - 10.0),
        ]
    )
