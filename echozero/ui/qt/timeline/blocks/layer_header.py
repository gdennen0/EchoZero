from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QBrush, QFont, QFontMetrics

from echozero.application.presentation.models import LayerHeaderControlPresentation, LayerPresentation
from echozero.ui.qt.timeline.style import LayerHeaderStyle, StatusChipStyle, TIMELINE_STYLE


@dataclass(slots=True)
class HeaderSlots:
    rect: QRectF
    title_rect: QRectF
    subtitle_rect: QRectF
    status_rect: QRectF
    controls_rect: QRectF
    toggle_rect: QRectF
    metadata_rect: QRectF


@dataclass(slots=True)
class HeaderHitTargets:
    control_rects: tuple[tuple[str, QRectF], ...]


class LayerHeaderBlock:
    def __init__(self, style: LayerHeaderStyle = TIMELINE_STYLE.layer_header):
        self.style = style

    def paint(
        self,
        painter: QPainter,
        slots: HeaderSlots,
        layer: LayerPresentation,
        *,
        dimmed: bool = False,
    ) -> HeaderHitTargets:
        rect = slots.rect
        fill_hex = self.style.selected_background_hex if layer.is_selected and not dimmed else self.style.dimmed_background_hex if dimmed else self.style.background_hex
        painter.fillRect(rect, QColor(fill_hex))

        title_font = QFont()
        title_font.setBold(self.style.title_font.bold)
        title_font.setPointSize(self.style.title_font.point_size)
        painter.setFont(title_font)
        painter.setPen(QColor(self.style.dimmed_title_hex if dimmed else self.style.title_hex))
        painter.drawText(
            slots.title_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine,
            self._elided_title_text(layer.title, title_font, slots.title_rect.width()),
        )

        self._draw_status_chips(painter, slots.status_rect, layer)
        control_rects = self._draw_header_controls(
            painter,
            slots.controls_rect,
            layer.header_controls,
            dimmed=dimmed,
        )

        if layer.takes:
            painter.setPen(QColor(self.style.toggle_border_hex))
            painter.setBrush(QBrush(QColor(self.style.toggle_fill_hex)))
            painter.drawRoundedRect(slots.toggle_rect, self.style.toggle_corner_radius, self.style.toggle_corner_radius)
            painter.setPen(QColor(self.style.toggle_text_hex))
            prior_font = painter.font()
            toggle_font = QFont(prior_font)
            toggle_font.setPointSize(self.style.toggle_font.point_size)
            toggle_font.setBold(self.style.toggle_font.bold)
            painter.setFont(toggle_font)
            painter.drawText(
                slots.toggle_rect.adjusted(0, -1, 0, -1),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
                'v' if layer.is_expanded else '>',
            )
            painter.setFont(prior_font)
        return HeaderHitTargets(control_rects=tuple(control_rects))

    @staticmethod
    def _elided_title_text(text: str, font: QFont, width: float) -> str:
        metrics = QFontMetrics(font)
        return metrics.elidedText(
            str(text),
            Qt.TextElideMode.ElideRight,
            max(0, int(width)),
        )

    def _draw_header_controls(
        self,
        painter: QPainter,
        controls_rect: QRectF,
        controls: list[LayerHeaderControlPresentation],
        *,
        dimmed: bool,
    ) -> list[tuple[str, QRectF]]:
        control_rects: list[tuple[str, QRectF]] = []
        x = controls_rect.left()
        for control in controls:
            width = self._control_width(control)
            rect = QRectF(x, controls_rect.top(), width, 18)
            if control.kind == "toggle":
                self._draw_active_button(
                    painter,
                    rect,
                    active=control.active,
                    dimmed=dimmed,
                    label=control.label,
                )
            else:
                self._draw_action_button(
                    painter,
                    rect,
                    control.label,
                    dimmed=dimmed or not control.enabled,
                )
            control_rects.append((control.control_id, rect))
            x += width + 6
        return control_rects

    @staticmethod
    def _control_width(control: LayerHeaderControlPresentation) -> float:
        if control.kind == "toggle":
            return 52.0
        return max(40.0, 10.0 + (len(control.label) * 7.0))

    def _draw_status_chips(self, painter: QPainter, rect: QRectF, layer: LayerPresentation) -> None:
        x = rect.left()
        if layer.status.stale:
            x = self._draw_chip(painter, QRectF(x, rect.top(), 46, 16), 'STALE', self.style.status.stale) + 6
        if layer.status.manually_modified:
            self._draw_chip(painter, QRectF(x, rect.top(), 52, 16), 'EDITED', self.style.status.edited)

    def _draw_chip(self, painter: QPainter, rect: QRectF, text: str, style: StatusChipStyle) -> float:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(style.fill_hex)))
        painter.drawRoundedRect(rect, style.corner_radius, style.corner_radius)
        painter.setPen(QColor(style.text_hex))
        prior_font = painter.font()
        chip_font = QFont(prior_font)
        chip_font.setPointSize(style.font.point_size)
        chip_font.setBold(style.font.bold)
        painter.setFont(chip_font)
        painter.drawText(rect.adjusted(0, -1, 0, -1), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, text)
        painter.setFont(prior_font)
        return rect.right()

    def _draw_active_button(
        self,
        painter: QPainter,
        rect: QRectF,
        *,
        active: bool,
        dimmed: bool,
        label: str = "ACTIVE",
    ) -> None:
        button_style = self.style.mute_solo
        state_style = button_style.active if active else button_style.inactive
        fill_hex = state_style.fill_hex
        if dimmed and not active:
            fill_hex = button_style.dimmed_inactive_fill_hex
        painter.setPen(QColor(button_style.border_hex))
        painter.setBrush(QBrush(QColor(fill_hex)))
        painter.drawRoundedRect(rect, button_style.corner_radius, button_style.corner_radius)
        painter.setPen(QColor(state_style.text_hex))
        prior_font = painter.font()
        button_font = QFont(prior_font)
        button_font.setPointSize(button_style.font.point_size)
        button_font.setBold(button_style.font.bold)
        painter.setFont(button_font)
        painter.drawText(
            rect.adjusted(0, -1, 0, -1),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            label,
        )
        painter.setFont(prior_font)

    def _draw_action_button(self, painter: QPainter, rect: QRectF, label: str, *, dimmed: bool) -> None:
        button_style = self.style.mute_solo
        fill_hex = button_style.inactive.fill_hex
        if dimmed:
            fill_hex = button_style.dimmed_inactive_fill_hex
        painter.setPen(QColor(button_style.border_hex))
        painter.setBrush(QBrush(QColor(fill_hex)))
        painter.drawRoundedRect(rect, button_style.corner_radius, button_style.corner_radius)
        painter.setPen(QColor(button_style.inactive.text_hex))
        prior_font = painter.font()
        button_font = QFont(prior_font)
        button_font.setPointSize(max(7, button_style.font.point_size - 1))
        button_font.setBold(button_style.font.bold)
        painter.setFont(button_font)
        painter.drawText(rect.adjusted(0, -1, 0, -1), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, label)
        painter.setFont(prior_font)
