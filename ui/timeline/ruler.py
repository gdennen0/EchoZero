"""
ruler.py - Time ruler widget for EchoZero 2 Timeline Prototype
Fixed height, adaptive ticks, playhead triangle, click-to-set-playhead.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush, QPolygon,
    QMouseEvent,
)
from PyQt6.QtWidgets import QWidget

import FEEL as F
from model import TimelineState

_GRID_INTERVALS = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0,
    30.0, 60.0, 120.0, 300.0, 600.0,
]
_TARGET_MINOR_PX = 50
_MAJOR_FACTOR = 5


def _choose_interval(zoom: float) -> float:
    for iv in reversed(_GRID_INTERVALS):
        if iv * zoom >= _TARGET_MINOR_PX:
            return iv
    return _GRID_INTERVALS[-1]


def _format_time(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    t = abs(seconds)
    m = int(t) // 60
    s = t - m * 60
    if s == int(s):
        return f"{sign}{m}:{int(s):02d}"
    return f"{sign}{m}:{s:05.2f}"


class TimeRuler(QWidget):
    playhead_changed = pyqtSignal(float)

    def __init__(self, state: TimelineState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setFixedHeight(F.RULER_HEIGHT)
        self.setMouseTracking(True)

    def time_to_x(self, t: float) -> float:
        return (t - self.state.scroll_x) * self.state.zoom_level

    def x_to_time(self, x: float) -> float:
        return x / self.state.zoom_level + self.state.scroll_x

    def paintEvent(self, _event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, F.RULER_BG_COLOR)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        border_pen = QPen(F.RULER_BORDER_COLOR)
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawLine(0, h - 1, w, h - 1)

        time_start = self.x_to_time(0)
        time_end = self.x_to_time(w)
        minor_iv = _choose_interval(self.state.zoom_level)
        major_iv = minor_iv * _MAJOR_FACTOR

        font = QFont(F.RULER_FONT_FAMILY, F.RULER_FONT_SIZE)
        painter.setFont(font)

        t0 = math.floor(time_start / minor_iv) * minor_iv
        t = t0
        while t <= time_end + minor_iv:
            x = int(self.time_to_x(t))
            is_major = abs(round(t / major_iv) * major_iv - t) < minor_iv * 0.01
            if is_major:
                pen = QPen(F.RULER_TICK_MAJOR_COLOR)
                tick_h = F.RULER_TICK_MAJOR_HEIGHT
                painter.setPen(pen)
                painter.drawLine(x, h - tick_h, x, h - 1)
                label = _format_time(t)
                painter.setPen(F.RULER_LABEL_COLOR)
                painter.drawText(x + 3, 2, 80, h - tick_h - 2,
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 label)
            else:
                pen = QPen(F.RULER_TICK_MINOR_COLOR)
                tick_h = F.RULER_TICK_MINOR_HEIGHT
                painter.setPen(pen)
                painter.drawLine(x, h - tick_h, x, h - 1)
            t += minor_iv

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        px = int(self.time_to_x(self.state.playhead_time))
        if 0 <= px <= w:
            ts = F.PLAYHEAD_TRIANGLE_SIZE
            triangle = QPolygon([
                QPoint(px - ts // 2, h - ts - 2),
                QPoint(px + ts // 2, h - ts - 2),
                QPoint(px, h - 2),
            ])
            painter.setBrush(QBrush(F.PLAYHEAD_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(triangle)
            pen = QPen(F.PLAYHEAD_COLOR)
            pen.setWidth(F.PLAYHEAD_WIDTH)
            painter.setPen(pen)
            painter.drawLine(px, 0, px, h - ts - 2)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            t = max(0.0, self.x_to_time(event.position().x()))
            self.state.playhead_time = t
            self.playhead_changed.emit(t)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            t = max(0.0, self.x_to_time(event.position().x()))
            self.state.playhead_time = t
            self.playhead_changed.emit(t)
            self.update()
