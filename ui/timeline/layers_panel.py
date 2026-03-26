"""
layers_panel.py - Layer labels panel for EchoZero 2 Timeline Prototype
Fixed width, draws layer names + color swatches, syncs vertical scroll with canvas.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen,
    QMouseEvent,
)
from PyQt6.QtWidgets import QWidget

import FEEL as F
from model import TimelineState

# Borrow EVENT_HEIGHT from FEEL for swatch sizing
_SWATCH_HEIGHT = 14


class LayersPanel(QWidget):
    layer_selected = pyqtSignal(str)

    def __init__(self, state: TimelineState, parent=None):
        super().__init__(parent)
        self.state = state
        self.selected_layer_id: Optional[str] = None
        self._hover_layer_id: Optional[str] = None
        self.setFixedWidth(F.LAYERS_PANEL_WIDTH)
        self.setMouseTracking(True)

    def _layer_y(self, index: int) -> int:
        return int(index * F.LAYER_ROW_HEIGHT - self.state.scroll_y)

    def _layer_at_y(self, y: float) -> Optional[int]:
        idx = int((y + self.state.scroll_y) / F.LAYER_ROW_HEIGHT)
        if 0 <= idx < len(self.state.layers):
            return idx
        return None

    def paintEvent(self, _event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, F.LAYERS_PANEL_BG_COLOR)

        font = QFont(F.LAYER_FONT_FAMILY, F.LAYER_FONT_SIZE)
        painter.setFont(font)

        for i, layer in enumerate(self.state.layers):
            y = self._layer_y(i)
            row_h = F.LAYER_ROW_HEIGHT
            if y + row_h < 0:
                continue
            if y > h:
                break

            if layer.id == self.selected_layer_id:
                painter.fillRect(0, y, w, row_h, F.LAYER_SELECTED_BG)
            elif layer.id == self._hover_layer_id:
                painter.fillRect(0, y, w, row_h, F.LAYER_HOVER_BG)

            # Color swatch
            r, g, b = layer.color
            swatch_x = F.LAYER_SWATCH_MARGIN
            swatch_y = y + (row_h - _SWATCH_HEIGHT) // 2
            painter.fillRect(swatch_x, swatch_y, F.LAYER_SWATCH_WIDTH, _SWATCH_HEIGHT,
                             QColor(r, g, b))

            # Layer name
            text_x = F.LAYER_SWATCH_MARGIN + F.LAYER_SWATCH_WIDTH + F.LAYER_TEXT_PADDING
            painter.setPen(F.LAYER_LABEL_COLOR)
            painter.drawText(
                text_x, y, w - text_x - 4, row_h,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                layer.name,
            )

            # Row separator
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            sep_pen = QPen(F.LAYERS_PANEL_BORDER_COLOR)
            sep_pen.setWidth(1)
            painter.setPen(sep_pen)
            painter.drawLine(0, y + row_h - 1, w, y + row_h - 1)

        # Right border
        border_pen = QPen(F.LAYERS_PANEL_BORDER_COLOR)
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawLine(w - 1, 0, w - 1, h)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._layer_at_y(event.position().y())
            if idx is not None:
                lid = self.state.layers[idx].id
                self.selected_layer_id = lid
                self.layer_selected.emit(lid)
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        idx = self._layer_at_y(event.position().y())
        old = self._hover_layer_id
        self._hover_layer_id = self.state.layers[idx].id if idx is not None else None
        if self._hover_layer_id != old:
            self.update()

    def leaveEvent(self, _event):
        if self._hover_layer_id is not None:
            self._hover_layer_id = None
            self.update()

    def sync_scroll(self, scroll_y: float):
        self.state.scroll_y = scroll_y
        self.update()
