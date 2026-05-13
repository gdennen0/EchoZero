from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF

from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.FEEL import RULER_FONT_SIZE, RULER_MIN_TICK_SPACING_PX
from echozero.ui.qt.timeline.time_grid import TimelineGridMode, visible_grid_lines
from echozero.ui.qt.timeline.style import RulerStyle, TIMELINE_STYLE


@dataclass(slots=True)
class RulerLayout:
    rect: QRectF
    header_width: float


class RulerBlock:
    def __init__(
        self,
        style: RulerStyle = TIMELINE_STYLE.ruler,
        *,
        playhead_color_hex: str = TIMELINE_STYLE.playhead.color_hex,
    ):
        self.style = style
        self.playhead_color_hex = playhead_color_hex

    def paint(
        self, painter: QPainter, layout: RulerLayout, presentation: TimelinePresentation
    ) -> None:
        rect = layout.rect
        painter.fillRect(rect, QColor(self.style.background_hex))
        painter.fillRect(
            QRectF(rect.left(), rect.bottom() - 1, rect.width(), 1), QColor(self.style.divider_hex)
        )
        painter.fillRect(
            QRectF(rect.left(), rect.top(), layout.header_width, rect.height()),
            QColor(self.style.header_background_hex),
        )
        painter.setPen(QColor(self.style.title_hex))
        painter.drawText(14, int(rect.top()) + 18, _ruler_title(presentation))
        label_font = QFont(painter.font())
        label_font.setPointSize(max(7, int(RULER_FONT_SIZE)))
        painter.setFont(label_font)

        pps = max(1.0, presentation.pixels_per_second)
        content_width = max(1.0, rect.width() - layout.header_width)
        for label, x in visible_ruler_marks(
            presentation=presentation,
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
            painter.drawText(int(x) + 4, int(rect.top()) + 12, label)

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
    major_step_seconds = max(1, int(ceil(float(RULER_MIN_TICK_SPACING_PX) / pps)))
    start_second = max(0, int(floor(scroll_x / pps)) - major_step_seconds)
    end_second = int(ceil((scroll_x + content_width) / pps)) + major_step_seconds
    first_mark_second = (start_second // major_step_seconds) * major_step_seconds

    marks: list[tuple[int, float]] = []
    for second in range(first_mark_second, max(start_second, end_second) + 1, major_step_seconds):
        if second < 0:
            continue
        x = content_start_x + (second * pps) - scroll_x
        if content_start_x <= x <= (content_start_x + content_width):
            marks.append((second, x))
    return marks


def visible_ruler_marks(
    *,
    presentation: TimelinePresentation,
    content_width: float,
    content_start_x: float,
) -> list[tuple[str, float]]:
    """Return visible ruler labels for either musical bars or plain seconds."""

    bpm = presentation.bpm
    if bpm is None or float(bpm) <= 0.0:
        return [
            (str(second), x)
            for second, x in visible_ruler_seconds(
                scroll_x=presentation.scroll_x,
                pixels_per_second=presentation.pixels_per_second,
                content_width=content_width,
                content_start_x=content_start_x,
            )
        ]

    beat_seconds = 60.0 / float(bpm)
    bar_lines = [
        line
        for line in visible_grid_lines(
            scroll_x=presentation.scroll_x,
            pixels_per_second=presentation.pixels_per_second,
            content_width=content_width,
            mode=TimelineGridMode.BEAT,
            bpm=bpm,
            beat_anchor_seconds=presentation.beat_anchor_seconds,
            min_spacing_px=max(RULER_MIN_TICK_SPACING_PX, beat_seconds * presentation.pixels_per_second),
        )
        if line.role == "bar"
    ]
    if not bar_lines:
        return []

    anchor = max(0.0, float(presentation.beat_anchor_seconds or 0.0))
    marks: list[tuple[str, float]] = []
    for line in bar_lines:
        beats_from_anchor = round((line.time_seconds - anchor) / beat_seconds) if beat_seconds else 0
        bar_number = max(0, beats_from_anchor // 4) + 1
        label = f"{bar_number}|1"
        x = timeline_x_for_time(
            line.time_seconds,
            scroll_x=presentation.scroll_x,
            pixels_per_second=presentation.pixels_per_second,
            content_start_x=content_start_x,
        )
        marks.append((label, x))
    return marks


def _ruler_title(presentation: TimelinePresentation) -> str:
    bpm = presentation.bpm
    if bpm is None or float(bpm) <= 0.0:
        return "Timeline"
    rounded_bpm = f"{float(bpm):.1f}".rstrip("0").rstrip(".")
    if presentation.bpm_confidence is not None and float(presentation.bpm_confidence) < 0.6:
        return f"Timeline · ~{rounded_bpm} BPM"
    return f"Timeline · {rounded_bpm} BPM"


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
