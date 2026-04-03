from __future__ import annotations

from dataclasses import dataclass
from math import floor

from PyQt6.QtGui import QColor, QPainter, QPen


@dataclass(slots=True)
class WaveformLanePresentation:
    color_hex: str
    row_height: int
    pixels_per_second: float
    scroll_x: float
    header_width: int
    width: int
    dimmed: bool = False


class WaveformLaneBlock:
    def paint(self, painter: QPainter, top: int, presentation: WaveformLanePresentation) -> None:
        base = QColor(presentation.color_hex)
        if presentation.dimmed:
            base.setAlpha(120)
        mid_y = top + presentation.row_height / 2
        pps = max(1.0, presentation.pixels_per_second)
        step = max(4, int(180 / pps * 14))
        amp_scale = min(1.0, max(0.35, pps / 220.0))
        painter.setPen(QPen(base, 1.5))
        for x, sample_index in visible_waveform_columns(presentation, step):
            amp = (((sample_index * 37) % 11) + 1) / 11.0
            h = amp * (presentation.row_height * 0.30) * amp_scale
            painter.drawLine(int(x), int(mid_y - h), int(x), int(mid_y + h))


def visible_waveform_columns(
    presentation: WaveformLanePresentation,
    step: int,
) -> list[tuple[float, int]]:
    """Return screen-space waveform columns while preserving scroll continuity."""
    content_left = presentation.header_width + 8
    content_right = max(content_left + 1, presentation.width - 8)
    visible_span = max(1.0, content_right - content_left)
    timeline_start_px = max(0.0, presentation.scroll_x)

    start_index = int(floor(timeline_start_px / step)) - 1
    end_index = int(floor((timeline_start_px + visible_span) / step)) + 2

    cols: list[tuple[float, int]] = []
    for idx in range(start_index, end_index + 1):
        x = content_left + ((idx * step) - timeline_start_px)
        if (content_left - step) <= x <= (content_right + step):
            cols.append((x, idx))
    return cols
