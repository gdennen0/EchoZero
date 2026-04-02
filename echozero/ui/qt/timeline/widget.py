"""Minimal read-only timeline shell for the new application architecture."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea

from echozero.application.presentation.models import TimelinePresentation, LayerPresentation, EventPresentation


class TimelineCanvas(QWidget):
    def __init__(self, presentation: TimelinePresentation, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self.setMinimumHeight(max(240, len(self.presentation.layers) * 72 + 60))
        self.setMinimumWidth(1200)
        self._row_height = 64
        self._header_width = 260
        self._event_height = 24

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#14161b"))

        self._draw_header_background(painter)
        self._draw_grid(painter)
        self._draw_layers(painter)
        self._draw_playhead(painter)

    def _draw_header_background(self, painter: QPainter) -> None:
        painter.fillRect(0, 0, self._header_width, self.height(), QColor("#1b1f27"))
        painter.fillRect(self._header_width, 0, 1, self.height(), QColor("#2c3340"))

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#232936"), 1))
        pps = max(1.0, self.presentation.pixels_per_second)
        seconds = int(max(10, (self.width() - self._header_width) / pps))
        for second in range(seconds + 2):
            x = self._header_width + int(second * pps)
            painter.drawLine(x, 0, x, self.height())

    def _draw_layers(self, painter: QPainter) -> None:
        for index, layer in enumerate(self.presentation.layers):
            top = index * self._row_height
            self._draw_layer_row(painter, layer, top)

    def _draw_layer_row(self, painter: QPainter, layer: LayerPresentation, top: int) -> None:
        bg = QColor("#1a2029") if layer.is_selected else QColor("#171b22")
        painter.fillRect(0, top, self.width(), self._row_height - 1, bg)
        painter.fillRect(0, top, self._header_width, self._row_height - 1, QColor("#1d222b"))
        painter.fillRect(0, top + self._row_height - 1, self.width(), 1, QColor("#252c38"))

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#f0f3f8"))
        painter.drawText(12, top + 22, layer.title)

        sub_font = QFont()
        sub_font.setPointSize(8)
        painter.setFont(sub_font)
        painter.setPen(QColor("#9aa4b2"))
        subtitle = layer.subtitle or layer.take_summary.compact_label
        painter.drawText(12, top + 40, subtitle)

        badge_x = 170
        for badge in layer.badges:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#2c6bed")))
            painter.drawRoundedRect(QRectF(badge_x, top + 12, 48, 18), 6, 6)
            painter.setPen(QColor("white"))
            painter.drawText(QRectF(badge_x, top + 12, 48, 18), Qt.AlignmentFlag.AlignCenter, badge[:6])
            badge_x += 54

        if layer.take_summary.total_take_count > 1:
            caret = "▾" if layer.is_expanded else "▸"
            painter.setPen(QColor("#d7dce4"))
            painter.drawText(235, top + 22, f"{caret} {layer.take_summary.total_take_count} takes")

        for event in layer.events:
            self._draw_event(painter, event, top)

    def _draw_event(self, painter: QPainter, event: EventPresentation, top: int) -> None:
        pps = max(1.0, self.presentation.pixels_per_second)
        x = self._header_width + (event.start * pps) - self.presentation.scroll_x
        width = max(10.0, (event.duration * pps))
        y = top + (self._row_height - self._event_height) / 2

        color = QColor(event.color or "#57a0ff")
        if event.is_selected:
            color = color.lighter(130)

        painter.setPen(QPen(color.darker(160), 1))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(QRectF(x, y, width, self._event_height), 5, 5)

        painter.setPen(QColor("#0b1220"))
        painter.drawText(QRectF(x + 6, y, max(0, width - 12), self._event_height), Qt.AlignmentFlag.AlignVCenter, event.label)

    def _draw_playhead(self, painter: QPainter) -> None:
        x = self._header_width + (self.presentation.playhead * self.presentation.pixels_per_second) - self.presentation.scroll_x
        painter.setPen(QPen(QColor("#ff5f57"), 2))
        painter.drawLine(int(x), 0, int(x), self.height())


class TimelineWidget(QWidget):
    def __init__(self, presentation: TimelinePresentation, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self.setWindowTitle("EchoZero Timeline Preview")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(16)
        header.setStyleSheet("background: #10141a; color: #f0f3f8;")

        title = QLabel(self.presentation.title)
        title.setStyleSheet("font-size: 14px; font-weight: 700;")
        status = QLabel(
            f"{'Playing' if self.presentation.is_playing else 'Stopped'}  •  "
            f"Layers: {len(self.presentation.layers)}  •  "
            f"Playhead: {self.presentation.playhead:.2f}s"
        )
        status.setStyleSheet("color: #9aa4b2;")

        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(status)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: #14161b; border: none;")
        canvas = TimelineCanvas(self.presentation)
        scroll.setWidget(canvas)
        layout.addWidget(scroll)
