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
    metadata_rect: QRectF


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

        self._draw_metadata_symbols(painter, layer, slots, dimmed)
        self._draw_status_chips(painter, slots.status_rect, layer)
        self._draw_ms_button(painter, QRectF(slots.controls_rect.left(), slots.controls_rect.top(), 24, 18), 'M', active=layer.muted, dimmed=dimmed)
        self._draw_ms_button(painter, QRectF(slots.controls_rect.left() + 28, slots.controls_rect.top(), 24, 18), 'S', active=layer.soloed, dimmed=dimmed)

        painter.setPen(QColor('#445065'))
        painter.setBrush(QBrush(QColor('#141922')))
        painter.drawRoundedRect(slots.toggle_rect, 6, 6)
        painter.setPen(QColor('#d7dce4'))
        prior_font = painter.font()
        toggle_font = QFont(prior_font)
        toggle_font.setPointSize(9)
        toggle_font.setBold(True)
        painter.setFont(toggle_font)
        painter.drawText(
            slots.toggle_rect.adjusted(0, -1, 0, -1),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            'v' if layer.is_expanded else '>',
        )
        painter.setFont(prior_font)

    def _draw_metadata_symbols(self, painter: QPainter, layer: LayerPresentation, slots: HeaderSlots, dimmed: bool) -> None:
        tokens = self._metadata_tokens(layer.badges)
        if not tokens:
            return

        rect = slots.metadata_rect
        painter.save()
        try:
            painter.setClipRect(rect)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor('#1f2b3a' if dimmed else '#1d2e45')))
            painter.drawRoundedRect(rect, 5, 5)

            prior_font = painter.font()
            meta_font = QFont(prior_font)
            meta_font.setPointSize(7)
            meta_font.setBold(True)
            painter.setFont(meta_font)

            x = rect.left() + 6
            for token in tokens:
                chip_w = 13
                chip = QRectF(x, rect.top() + 1, chip_w, rect.height() - 2)
                painter.setBrush(QBrush(QColor('#2b4260' if not dimmed else '#253447')))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(chip, 4, 4)
                painter.setPen(QColor('#c5dcf5' if not dimmed else '#9db3cb'))
                painter.drawText(
                    chip.adjusted(0, -1, 0, -1),
                    Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
                    token,
                )
                x += chip_w + 4
                if x > rect.right() - 12:
                    break

            painter.setFont(prior_font)
        finally:
            painter.restore()

    @staticmethod
    def _metadata_tokens(badges: list[str]) -> list[str]:
        if not badges:
            return []

        priority = ["main", "stem", "audio", "event", "classifier-preview", "real-data"]
        symbol_map = {
            "main": "M",
            "stem": "S",
            "audio": "A",
            "event": "E",
            "classifier-preview": "C",
            "real-data": "R",
        }

        normalized = [str(b).strip().lower() for b in badges if str(b).strip()]
        ordered: list[str] = []
        for key in priority:
            if key in normalized:
                ordered.append(key)
        for key in normalized:
            if key not in ordered:
                ordered.append(key)

        visible = ordered[:4]
        tokens = [symbol_map.get(v, v[:1].upper()) for v in visible]
        remaining = max(0, len(ordered) - len(visible))
        if remaining:
            tokens.append(f"+{remaining}")
        return tokens

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
        prior_font = painter.font()
        chip_font = QFont(prior_font)
        chip_font.setPointSize(8)
        chip_font.setBold(True)
        painter.setFont(chip_font)
        painter.drawText(rect.adjusted(0, -1, 0, -1), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, text)
        painter.setFont(prior_font)
        return rect.right()

    def _draw_ms_button(self, painter: QPainter, rect: QRectF, label: str, *, active: bool, dimmed: bool) -> None:
        fill = QColor('#2b6bf0' if active else '#18202a')
        if dimmed and not active:
            fill = QColor('#10151b')
        painter.setPen(QColor('#4b5669'))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QColor('#ffffff' if active else '#b8c0cc'))
        prior_font = painter.font()
        button_font = QFont(prior_font)
        button_font.setPointSize(8)
        button_font.setBold(True)
        painter.setFont(button_font)
        painter.drawText(rect.adjusted(0, -1, 0, -1), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, label)
        painter.setFont(prior_font)
