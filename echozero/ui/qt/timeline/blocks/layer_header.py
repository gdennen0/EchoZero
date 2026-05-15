from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QBrush, QFont, QFontMetrics, QPolygonF

from echozero.application.presentation.models import (
    LayerHeaderControlPresentation,
    LayerPresentation,
)
from echozero.ui.qt.timeline.style import LayerHeaderStyle, StatusChipStyle, TIMELINE_STYLE


@dataclass(slots=True)
class HeaderSlots:
    rect: QRectF
    title_rect: QRectF
    subtitle_rect: QRectF
    status_rect: QRectF
    controls_rect: QRectF
    active_rect: QRectF
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
        has_child_layers: bool = False,
    ) -> HeaderHitTargets:
        rect = slots.rect
        fill_hex = (
            self.style.selected_background_hex
            if layer.is_selected and not dimmed
            else self.style.dimmed_background_hex if dimmed else self.style.background_hex
        )
        painter.fillRect(rect, QColor(fill_hex))

        title_font = QFont()
        title_font.setBold(self.style.title_font.bold)
        title_font.setPointSize(self.style.title_font.point_size)
        painter.setFont(title_font)
        painter.setPen(QColor(self.style.dimmed_title_hex if dimmed else self.style.title_hex))
        painter.drawText(
            slots.title_rect,
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextSingleLine,
            self._elided_title_text(layer.title, title_font, slots.title_rect.width()),
        )
        control_rects: list[tuple[str, QRectF]] = []
        if not layer.is_fully_collapsed:
            self._draw_status_chips(painter, slots.status_rect, layer)
            control_rects = self._draw_header_controls(
                painter,
                slots.controls_rect,
                slots.active_rect,
                layer.header_controls,
                dimmed=dimmed,
            )

        if layer.takes or layer.is_fully_collapsed or has_child_layers:
            self._draw_toggle_glyph(
                painter,
                slots.toggle_rect,
                is_fully_collapsed=layer.is_fully_collapsed,
                is_expanded=layer.is_expanded,
            )
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
        active_rect: QRectF,
        controls: list[LayerHeaderControlPresentation],
        *,
        dimmed: bool,
    ) -> list[tuple[str, QRectF]]:
        del active_rect
        control_rects: list[tuple[str, QRectF]] = []
        control_gap = 4.0
        total_controls_width = (
            sum(self._control_width(control) for control in controls)
            + max(0, len(controls) - 1) * control_gap
        )
        x = max(controls_rect.left(), controls_rect.right() - total_controls_width)
        for control in controls:
            width = self._control_width(control)
            if x + width > controls_rect.right():
                width = max(0.0, controls_rect.right() - x)
            if width <= 0.0:
                break
            rect = QRectF(x, controls_rect.top(), width, 16)
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
            x += width + control_gap
        return control_rects

    @staticmethod
    def _control_width(control: LayerHeaderControlPresentation) -> float:
        if control.kind == "toggle":
            if control.control_id in {"set_layer_mute", "set_layer_solo"}:
                return 24.0
            return 52.0
        label = str(control.label or "")
        compact = LayerHeaderBlock._compact_action_label(label)
        if control.control_id == "layer_pipeline_actions" or compact == "⋯":
            return 18.0
        return max(26.0, 10.0 + (len(compact) * 6.0))

    @staticmethod
    def _compact_action_label(label: str) -> str:
        normalized = str(label or "").strip().lower()
        mapping = {
            "pipelines": "⋯",
            "pipeline": "⋯",
            "push": "PUSH",
            "pull": "PULL",
            "sections": "SEC",
            "section": "SEC",
            "settings": "CFG",
        }
        if normalized in mapping:
            return mapping[normalized]
        return str(label or "").strip().upper()[:4]

    def _draw_status_chips(
        self, painter: QPainter, rect: QRectF, layer: LayerPresentation
    ) -> None:
        if rect.height() < 10:
            return
        x = rect.left()
        if layer.status.stale:
            x = (
                self._draw_chip(
                    painter,
                    QRectF(x, rect.top(), 46, 16),
                    "STALE",
                    self.style.status.stale,
                )
                + 6
            )
        if layer.status.manually_modified:
            self._draw_chip(
                painter, QRectF(x, rect.top(), 52, 16), "EDITED", self.style.status.edited
            )

    def _draw_chip(
        self, painter: QPainter, rect: QRectF, text: str, style: StatusChipStyle
    ) -> float:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(style.fill_hex)))
        painter.drawRoundedRect(rect, style.corner_radius, style.corner_radius)
        painter.setPen(QColor(style.text_hex))
        prior_font = painter.font()
        chip_font = QFont(prior_font)
        chip_font.setPointSize(style.font.point_size)
        chip_font.setBold(style.font.bold)
        painter.setFont(chip_font)
        painter.drawText(
            rect.adjusted(0, -1, 0, -1),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            text,
        )
        painter.setFont(prior_font)
        return rect.right()

    def _draw_toggle_glyph(
        self,
        painter: QPainter,
        rect: QRectF,
        *,
        is_fully_collapsed: bool,
        is_expanded: bool,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(self.style.toggle_text_hex)))
        if is_fully_collapsed:
            center_x = rect.center().x()
            center_y = rect.center().y()
            painter.drawRect(QRectF(center_x - 3.5, center_y - 0.75, 7.0, 1.5))
            painter.drawRect(QRectF(center_x - 0.75, center_y - 3.5, 1.5, 7.0))
            painter.restore()
            return

        inset = 4.0
        left = rect.left() + inset
        right = rect.right() - inset
        top = rect.top() + inset
        bottom = rect.bottom() - inset
        mid_x = rect.center().x()
        mid_y = rect.center().y()
        if is_expanded:
            polygon = QPolygonF(
                [
                    QPointF(left, top),
                    QPointF(right, top),
                    QPointF(mid_x, bottom),
                ]
            )
        else:
            polygon = QPolygonF(
                [
                    QPointF(left, top),
                    QPointF(right, mid_y),
                    QPointF(left, bottom),
                ]
            )
        painter.drawPolygon(polygon)
        painter.restore()

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
        painter.drawRect(rect)
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

    def _draw_action_button(
        self, painter: QPainter, rect: QRectF, label: str, *, dimmed: bool
    ) -> None:
        label = self._compact_action_label(label)
        button_style = self.style.mute_solo
        fill_hex = button_style.inactive.fill_hex
        if dimmed:
            fill_hex = button_style.dimmed_inactive_fill_hex
        painter.setPen(QColor(button_style.border_hex))
        painter.setBrush(QBrush(QColor(fill_hex)))
        painter.drawRect(rect)
        painter.setPen(QColor(button_style.inactive.text_hex))
        prior_font = painter.font()
        button_font = QFont(prior_font)
        button_font.setPointSize(max(7, button_style.font.point_size - 1))
        button_font.setBold(button_style.font.bold)
        painter.setFont(button_font)
        painter.drawText(
            rect.adjusted(0, -1, 0, -1),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            label,
        )
        painter.setFont(prior_font)
