from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush

from echozero.application.presentation.models import EventPresentation
from echozero.perf import timed
from echozero.ui.FEEL import EVENT_LABEL_MIN_WIDTH_PX, EVENT_MIN_VISIBLE_WIDTH_PX
from echozero.ui.qt.timeline.style import EventLaneStyle, TIMELINE_STYLE


@dataclass(slots=True)
class EventLanePresentation:
    layer_id: object
    take_id: object | None
    events: list[EventPresentation]
    pixels_per_second: float
    scroll_x: float
    header_width: int
    event_height: int = 22
    dimmed: bool = False
    viewport_width: int = 1440


class EventLaneBlock:
    def __init__(self, style: EventLaneStyle = TIMELINE_STYLE.event_lane):
        self.style = style

    def paint(
        self,
        painter: QPainter,
        top_y: int,
        presentation: EventLanePresentation,
    ) -> list[tuple[QRectF, object, object | None, object]]:
        rects: list[tuple[QRectF, object, object | None, object]] = []

        pps = max(1.0, presentation.pixels_per_second)
        content_left = float(presentation.header_width)
        content_right = float(max(presentation.header_width + 1, presentation.viewport_width))
        visible_start_t = max(0.0, presentation.scroll_x / pps)
        visible_end_t = max(visible_start_t, (presentation.scroll_x + max(1.0, content_right - content_left)) / pps)

        with timed("timeline.paint.event_lane"):
            for event in presentation.events:
                if event.end < visible_start_t:
                    continue
                if event.start > visible_end_t:
                    break

                x = presentation.header_width + (event.start * pps) - presentation.scroll_x
                width = max(float(EVENT_MIN_VISIBLE_WIDTH_PX), (event.duration * pps))
                if x + width < content_left - 2 or x > content_right + 2:
                    continue

                rect = QRectF(x, top_y, width, presentation.event_height)
                rects.append((rect, presentation.layer_id, presentation.take_id, event.event_id))

                color = QColor(event.color or self.style.default_fill_hex)
                if presentation.dimmed:
                    color.setAlpha(self.style.dimmed_alpha)
                if event.is_selected:
                    color = color.lighter(self.style.selection_lighten_factor)
                border_width = self.style.selected_border_width_px if event.is_selected else self.style.normal_border_width_px
                painter.setPen(QPen(color.darker(self.style.border_darkness_factor), border_width))
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(rect, self.style.corner_radius, self.style.corner_radius)

                if width >= EVENT_LABEL_MIN_WIDTH_PX:
                    painter.setPen(QColor(self.style.text_hex))
                    painter.drawText(
                        QRectF(x + 6, top_y, max(0, width - 12), presentation.event_height),
                        Qt.AlignmentFlag.AlignVCenter,
                        event.label,
                    )
        return rects
