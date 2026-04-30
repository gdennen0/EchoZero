from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush

from echozero.application.presentation.models import EventPresentation
from echozero.application.shared.enums import LayerKind
from echozero.perf import timed
from echozero.ui.FEEL import (
    EVENT_LABEL_MIN_WIDTH_PX,
    EVENT_MIN_VISIBLE_WIDTH_PX,
    EVENT_SELECTION_BORDER_PX,
    EVENT_SELECTION_COLOR,
    EVENT_SELECTION_OUTLINE_EXPAND_PX,
    EVENT_SELECTION_TINY_WIDTH_EXTRA_PX,
    EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX,
)
from echozero.ui.qt.timeline.style import EventLaneStyle, TIMELINE_STYLE


@dataclass(slots=True)
class EventLanePresentation:
    layer_id: object
    take_id: object | None
    events: list[EventPresentation]
    pixels_per_second: float
    scroll_x: float
    header_width: int
    layer_kind: LayerKind = LayerKind.EVENT
    event_height: int = 22
    dimmed: bool = False
    viewport_width: int = 1440
    default_fill_hex: str | None = None


class EventLaneBlock:
    def __init__(self, style: EventLaneStyle = TIMELINE_STYLE.event_lane):
        self.style = style

    def _selected_outline_width_px(self, event_width: float) -> int:
        base_width = max(EVENT_SELECTION_BORDER_PX, self.style.selected_border_width_px)
        if event_width <= EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX:
            return base_width + EVENT_SELECTION_TINY_WIDTH_EXTRA_PX
        return base_width

    def _paint_selected_outline(
        self,
        painter: QPainter,
        *,
        rect: QRectF,
        corner_radius: float,
        dimmed: bool,
    ) -> None:
        outline = QColor(EVENT_SELECTION_COLOR)
        if dimmed:
            outline.setAlpha(210)
        outline_width = self._selected_outline_width_px(rect.width())
        expand = float(EVENT_SELECTION_OUTLINE_EXPAND_PX)
        painter.save()
        painter.setPen(QPen(outline, outline_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            rect.adjusted(-expand, -expand, expand, expand),
            corner_radius + expand,
            corner_radius + expand,
        )
        painter.restore()

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
                badge_tokens = {
                    str(badge).strip().lower()
                    for badge in getattr(event, "badges", [])
                    if str(badge).strip()
                }
                if "demoted" in badge_tokens:
                    color = QColor(self.style.demoted_fill_hex)
                else:
                    color = QColor(
                        event.color or presentation.default_fill_hex or self.style.default_fill_hex
                    )
                if presentation.dimmed:
                    color.setAlpha(self.style.dimmed_alpha)
                if event.is_selected:
                    color = color.lighter(self.style.selection_lighten_factor)
                rendered_rect = rect
                border_width = (
                    self.style.selected_border_width_px
                    if event.is_selected
                    else self.style.normal_border_width_px
                )
                painter.setPen(QPen(color.darker(self.style.border_darkness_factor), border_width))
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(rect, self.style.corner_radius, self.style.corner_radius)
                if event.is_selected:
                    self._paint_selected_outline(
                        painter,
                        rect=rect,
                        corner_radius=float(self.style.corner_radius),
                        dimmed=presentation.dimmed,
                    )

                if width >= EVENT_LABEL_MIN_WIDTH_PX:
                    painter.setPen(QColor(self.style.text_hex))
                    painter.drawText(
                        QRectF(x + 6, top_y, max(0, width - 12), presentation.event_height),
                        Qt.AlignmentFlag.AlignVCenter,
                        event.label,
                    )
                rects.append(
                    (rendered_rect, presentation.layer_id, presentation.take_id, event.event_id)
                )
        return rects
