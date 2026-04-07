from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont

from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, TransportBarStyle


class TransportBarBlock:
    def __init__(self, style: TransportBarStyle = TIMELINE_STYLE.transport_bar):
        self.style = style

    def paint(self, painter: QPainter, layout: TransportLayout, presentation: TimelinePresentation) -> dict[str, object]:
        painter.fillRect(layout.rect, QColor(self.style.background_hex))

        painter.setPen(QColor(self.style.title_hex))
        painter.drawText(layout.title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, presentation.title)

        play_rect = layout.controls_rect.adjusted(0, 0, -64, 0)
        stop_rect = layout.controls_rect.adjusted(60, 0, 0, 0)
        self._draw_button(painter, play_rect, 'Pause' if presentation.is_playing else 'Play')
        self._draw_button(painter, stop_rect, 'Stop')

        painter.setPen(QColor(self.style.time_hex))
        painter.drawText(layout.time_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{presentation.current_time_label} / {presentation.end_time_label}")

        painter.setPen(QColor(self.style.meta_hex))
        separator = "\u2022"
        painter.drawText(layout.meta_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{'Playing' if presentation.is_playing else 'Stopped'}  {separator}  Layers: {len(presentation.layers)}  {separator}  Zoom: {presentation.pixels_per_second:.0f}px/s")

        return {
            'play': play_rect,
            'stop': stop_rect,
        }

    def _draw_button(self, painter: QPainter, rect, label: str) -> None:
        button_style = self.style.button
        painter.setPen(QPen(QColor(button_style.border_hex), 1))
        painter.setBrush(QBrush(QColor(button_style.fill_hex)))
        painter.drawRoundedRect(rect, button_style.corner_radius, button_style.corner_radius)
        painter.setPen(QColor(button_style.text_hex))
        prior_font = painter.font()
        button_font = QFont(prior_font)
        button_font.setPointSize(button_style.font.point_size)
        button_font.setBold(button_style.font.bold)
        painter.setFont(button_font)
        painter.drawText(rect.adjusted(0, -1, 0, -1), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, label)
        painter.setFont(prior_font)
