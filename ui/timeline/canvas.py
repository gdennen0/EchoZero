"""
canvas.py - Core timeline canvas (QWidget) for EchoZero 2 Timeline Prototype
Batch render passes, aggressive culling for 60fps with 500 events.
"""
from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush, QPolygon,
    QMouseEvent, QWheelEvent, QKeyEvent,
)
from PyQt6.QtWidgets import QWidget

import FEEL as F
from model import TimelineEvent, TimelineState, visible_events

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


class TimelineCanvas(QWidget):
    playhead_changed = pyqtSignal(float)
    event_moved = pyqtSignal(str, float)
    selection_changed = pyqtSignal(object)  # set of event ids

    def __init__(self, state: TimelineState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._drag_event: Optional[TimelineEvent] = None
        self._drag_offset_time: float = 0.0
        self._pan_start: Optional[QPoint] = None
        self._pan_scroll_x_start: float = 0.0
        self._pan_scroll_y_start: float = 0.0
        self._is_panning: bool = False

    def time_to_x(self, t: float) -> float:
        return (t - self.state.scroll_x) * self.state.zoom_level

    def x_to_time(self, x: float) -> float:
        return x / self.state.zoom_level + self.state.scroll_x

    def layer_to_y(self, layer_index: int) -> float:
        return layer_index * F.LAYER_ROW_HEIGHT - self.state.scroll_y

    def y_to_layer(self, y: float) -> int:
        return int((y + self.state.scroll_y) / F.LAYER_ROW_HEIGHT)

    def _event_rect(self, event: TimelineEvent, layer_index: int) -> QRect:
        x = int(self.time_to_x(event.time))
        w = max(2, int(event.duration * self.state.zoom_level))
        y = int(self.layer_to_y(layer_index)) + F.EVENT_VERTICAL_PADDING
        return QRect(x, y, w, F.EVENT_HEIGHT)

    def _layer_index_of(self, layer_id: str) -> int:
        for i, la in enumerate(self.state.layers):
            if la.id == layer_id:
                return i
        return 0

    def _event_at(self, pos: QPoint) -> Optional[TimelineEvent]:
        t = self.x_to_time(pos.x())
        li = self.y_to_layer(pos.y())
        if li < 0 or li >= len(self.state.layers):
            return None
        target_lid = self.state.layers[li].id
        for ev in visible_events(self.state.events, t - 1, t + 1):
            if ev.layer_id != target_lid:
                continue
            if ev.time <= t <= ev.time + ev.duration:
                return ev
        return None

    def paintEvent(self, _event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        time_start = self.x_to_time(0)
        time_end = self.x_to_time(w)
        evs = visible_events(self.state.events, time_start, time_end)
        layer_index = {la.id: i for i, la in enumerate(self.state.layers)}

        # Pass 1: Background
        painter.fillRect(0, 0, w, h, F.BG_COLOR)
        for i in range(len(self.state.layers)):
            y = int(self.layer_to_y(i))
            if y > h:
                break
            if y + F.LAYER_ROW_HEIGHT < 0:
                continue
            if i % 2 == 1:
                band = QColor(F.BG_COLOR)
                band.setAlpha(F.SECTION_ALPHA + 10)
                painter.fillRect(0, y, w, F.LAYER_ROW_HEIGHT, band)

        # Pass 2: Grid lines (no anti-aliasing, pixel-snapped)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        minor_iv = _choose_interval(self.state.zoom_level)
        major_iv = minor_iv * _MAJOR_FACTOR
        t0 = math.floor(time_start / minor_iv) * minor_iv
        t = t0
        while t <= time_end + minor_iv:
            x = int(self.time_to_x(t))
            is_major = abs(round(t / major_iv) * major_iv - t) < minor_iv * 0.01
            pen = QPen(F.GRID_MAJOR_COLOR if is_major else F.GRID_MINOR_COLOR)
            pen.setWidth(F.GRID_MAJOR_WIDTH if is_major else F.GRID_MINOR_WIDTH)
            painter.setPen(pen)
            painter.drawLine(x, 0, x, h)
            t += minor_iv

        # Pass 3+4: Event backgrounds + borders; Pass 5: Labels
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(F.EVENT_FONT_FAMILY, F.EVENT_FONT_SIZE)
        painter.setFont(font)

        for ev in evs:
            li = layer_index.get(ev.layer_id, -1)
            if li < 0:
                continue
            rect = self._event_rect(ev, li)
            if rect.right() < 0 or rect.left() > w:
                continue

            r, g, b = ev.color
            bg = QColor(r, g, b, F.EVENT_ALPHA)
            painter.setBrush(QBrush(bg))

            is_selected = ev.id in self.state.selection
            if is_selected:
                pen = QPen(F.EVENT_SELECTED_BORDER_COLOR)
                pen.setWidth(F.EVENT_SELECTED_BORDER_WIDTH)
            else:
                border_color = QColor(r, g, b, F.EVENT_BORDER_ALPHA)
                pen = QPen(border_color)
                pen.setWidth(F.EVENT_BORDER_WIDTH)
            painter.setPen(pen)
            painter.drawRoundedRect(rect, F.EVENT_RADIUS, F.EVENT_RADIUS)

            if rect.width() >= F.EVENT_MIN_LABEL_WIDTH:
                label_color = QColor(F.EVENT_LABEL_COLOR)
                label_color.setAlpha(F.EVENT_LABEL_ALPHA)
                painter.setPen(label_color)
                painter.drawText(
                    rect.adjusted(4, 0, -2, 0),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    ev.label,
                )

        # Pass 6: Playhead
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        px = int(self.time_to_x(self.state.playhead_time))
        if 0 <= px <= w:
            pen = QPen(F.PLAYHEAD_COLOR)
            pen.setWidth(F.PLAYHEAD_WIDTH)
            painter.setPen(pen)
            painter.drawLine(px, 0, px, h)
            ts = F.PLAYHEAD_TRIANGLE_SIZE
            triangle = QPolygon([
                QPoint(px - ts // 2, 0),
                QPoint(px + ts // 2, 0),
                QPoint(px, ts),
            ])
            painter.setBrush(QBrush(F.PLAYHEAD_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(triangle)

        # Pass 7: Selection overlay (reserved for rubber-band selection)
        painter.end()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        cursor_x = event.position().x()
        time_under_cursor = self.x_to_time(cursor_x)
        factor = 1.0 + F.ZOOM_STEP * (1 if delta > 0 else -1)
        new_zoom = max(F.ZOOM_MIN, min(F.ZOOM_MAX, self.state.zoom_level * factor))
        self.state.zoom_level = new_zoom
        self.state.scroll_x = max(0.0, time_under_cursor - cursor_x / new_zoom)
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        button = event.button()
        mods = event.modifiers()
        alt_held = bool(mods & Qt.KeyboardModifier.AltModifier)
        shift_held = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton and alt_held
        ):
            self._is_panning = True
            self._pan_start = pos
            self._pan_scroll_x_start = self.state.scroll_x
            self._pan_scroll_y_start = self.state.scroll_y
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if button == Qt.MouseButton.LeftButton:
            hit = self._event_at(pos)
            if hit:
                if shift_held:
                    if hit.id in self.state.selection:
                        self.state.selection.discard(hit.id)
                    else:
                        self.state.selection.add(hit.id)
                else:
                    self.state.selection = {hit.id}
                self._drag_event = hit
                self._drag_offset_time = self.x_to_time(pos.x()) - hit.time
                self.selection_changed.emit(set(self.state.selection))
            else:
                self.state.selection.clear()
                self.selection_changed.emit(set())
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        if self._is_panning and self._pan_start is not None:
            dx = pos.x() - self._pan_start.x()
            dy = pos.y() - self._pan_start.y()
            self.state.scroll_x = max(0.0, self._pan_scroll_x_start - dx / self.state.zoom_level)
            self.state.scroll_y = max(0.0, self._pan_scroll_y_start - dy)
            self.update()
            return
        if self._drag_event and (event.buttons() & Qt.MouseButton.LeftButton):
            new_time = max(0.0, self.x_to_time(pos.x()) - self._drag_offset_time)
            self._drag_event.time = new_time
            self.event_moved.emit(self._drag_event.id, new_time)
            self.state.events.sort(key=lambda e: e.time)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._is_panning:
            self._is_panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._drag_event = None

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        zoom = self.state.zoom_level
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.state.zoom_level = min(F.ZOOM_MAX, zoom * (1 + F.ZOOM_STEP))
            self.update()
        elif key == Qt.Key.Key_Minus:
            self.state.zoom_level = max(F.ZOOM_MIN, zoom * (1 - F.ZOOM_STEP))
            self.update()
        elif key == Qt.Key.Key_Home:
            self.state.scroll_x = 0.0
            self.update()
        elif key == Qt.Key.Key_End:
            if self.state.events:
                last = max(e.time + e.duration for e in self.state.events)
                self.state.scroll_x = max(0.0, last - self.width() / self.state.zoom_level)
            self.update()
        else:
            super().keyPressEvent(event)
