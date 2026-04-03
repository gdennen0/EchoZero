from __future__ import annotations

from dataclasses import dataclass

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
        x0 = presentation.header_width + 8 - presentation.scroll_x
        width = presentation.width - presentation.header_width - 16
        pps = max(1.0, presentation.pixels_per_second)
        step = max(4, int(180 / pps * 14))
        amp_scale = min(1.0, max(0.35, pps / 220.0))
        painter.setPen(QPen(base, 1.5))
        for i in range(int(width // step) + 3):
            x = x0 + i * step
            amp = (((i * 37) % 11) + 1) / 11.0
            h = amp * (presentation.row_height * 0.30) * amp_scale
            painter.drawLine(int(x), int(mid_y - h), int(x), int(mid_y + h))
