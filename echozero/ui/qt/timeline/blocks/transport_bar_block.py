from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen

from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout


class TransportBarBlock:
    def paint(self, painter: QPainter, layout: TransportLayout, presentation: TimelinePresentation) -> dict[str, object]:
        painter.fillRect(layout.rect, QColor('#0e1217'))

        painter.setPen(QColor('#f0f3f8'))
        painter.drawText(layout.title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, presentation.title)

        play_rect = layout.controls_rect.adjusted(0, 0, -64, 0)
        stop_rect = layout.controls_rect.adjusted(60, 0, 0, 0)
        self._draw_button(painter, play_rect, 'Pause' if presentation.is_playing else 'Play')
        self._draw_button(painter, stop_rect, 'Stop')

        painter.setPen(QColor('#f6f8fb'))
        painter.drawText(layout.time_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{presentation.current_time_label} / {presentation.end_time_label}")

        painter.setPen(QColor('#93a0b1'))
        painter.drawText(layout.meta_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{'Playing' if presentation.is_playing else 'Stopped'}  •  Layers: {len(presentation.layers)}  •  Zoom: {presentation.pixels_per_second:.0f}px/s")

        return {
            'play': play_rect,
            'stop': stop_rect,
        }

    def _draw_button(self, painter: QPainter, rect, label: str) -> None:
        painter.setPen(QPen(QColor('#334055'), 1))
        painter.setBrush(QBrush(QColor('#1b2330')))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QColor('white'))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
