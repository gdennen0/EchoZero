from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QBrush, QFont

from echozero.application.presentation.models import LayerPresentation


@dataclass(slots=True)
class HeaderSlots:
    rect: QRectF
    title_rect: QRectF
    subtitle_rect: QRectF
    status_rect: QRectF
    controls_rect: QRectF
    toggle_rect: QRectF
    badges_origin_x: float
    badges_y: float


class LayerHeaderBlock:
    def paint(self, painter: QPainter, slots: HeaderSlots, layer: LayerPresentation, *, dimmed: bool = False) -> None:
        rect = slots.rect
        painter.fillRect(rect, QColor('#202833' if layer.is_selected and not dimmed else '#151922' if dimmed else '#1b212a'))

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        painter.setFont(title_font)
        painter.setPen(QColor('#cbd3df' if dimmed else '#f0f3f8'))
        painter.drawText(slots.title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, layer.title)

        sub_font = QFont()
        sub_font.setPointSize(8)
        painter.setFont(sub_font)
        painter.setPen(QColor('#6f7a88' if dimmed else '#98a3b3'))
        painter.drawText(slots.subtitle_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, layer.subtitle)

        self._draw_badges(painter, layer, slots, dimmed)
        self._draw_status_chips(painter, slots.status_rect, layer)
        self._draw_ms_button(painter, QRectF(slots.controls_rect.left(), slots.controls_rect.top(), 24, 18), 'M', active=layer.muted, dimmed=dimmed)
        self._draw_ms_button(painter, QRectF(slots.controls_rect.left() + 28, slots.controls_rect.top(), 24, 18), 'S', active=layer.soloed, dimmed=dimmed)

        painter.setPen(QColor('#445065'))
        painter.setBrush(QBrush(QColor('#141922')))
        painter.drawRoundedRect(slots.toggle_rect, 6, 6)
        painter.setPen(QColor('#d7dce4'))
        painter.drawText(slots.toggle_rect, Qt.AlignmentFlag.AlignCenter, 'v' if layer.is_expanded else '>')

    def _draw_badges(self, painter: QPainter, layer: LayerPresentation, slots: HeaderSlots, dimmed: bool) -> None:
        mx = slots.badges_origin_x
        for badge in layer.badges:
            badge_rect = QRectF(mx, slots.badges_y, max(44, 12 + len(badge) * 6), 14)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor('#314056' if dimmed else '#2b6bf0')))
            painter.drawRoundedRect(badge_rect, 5, 5)
            painter.setPen(QColor('#dce3eb'))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge)
            mx += badge_rect.width() + 4

    def _draw_status_chips(self, painter: QPainter, rect: QRectF, layer: LayerPresentation) -> None:
        x = rect.left()
        if layer.status.stale:
            x = self._draw_chip(painter, QRectF(x, rect.top(), 46, 16), 'STALE', '#7a5b16', '#f8c555') + 6
        if layer.status.manually_modified:
            self._draw_chip(painter, QRectF(x, rect.top(), 52, 16), 'EDITED', '#184c39', '#7fd1ae')

    def _draw_chip(self, painter: QPainter, rect: QRectF, text: str, fill: str, fg: str) -> float:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(fill)))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QColor(fg))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        return rect.right()

    def _draw_ms_button(self, painter: QPainter, rect: QRectF, label: str, *, active: bool, dimmed: bool) -> None:
        fill = QColor('#2b6bf0' if active else '#18202a')
        if dimmed and not active:
            fill = QColor('#10151b')
        painter.setPen(QColor('#4b5669'))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QColor('#ffffff' if active else '#b8c0cc'))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
