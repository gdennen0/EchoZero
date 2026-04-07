from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QFont

from echozero.application.presentation.models import LayerPresentation, TakeActionPresentation, TakeLanePresentation
from echozero.ui.qt.timeline.blocks.layouts import TakeRowLayout
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, TakeRowStyle


@dataclass(slots=True)
class TakeRowHitTargets:
    take_rect: tuple[QRectF, object, object]
    options_toggle_rect: tuple[QRectF, object, object] | None = None
    action_rects: list[tuple[QRectF, object, object, str]] = field(default_factory=list)


class TakeRowBlock:
    def __init__(self, style: TakeRowStyle = TIMELINE_STYLE.take_row):
        self.style = style

    def paint_header(
        self,
        painter: QPainter,
        layout: TakeRowLayout,
        layer: LayerPresentation,
        take: TakeLanePresentation,
        *,
        options_open: bool,
        dimmed: bool = False,
    ) -> TakeRowHitTargets:
        painter.fillRect(layout.row_rect, QColor(self.style.dimmed_row_fill_hex if dimmed else self.style.row_fill_hex))
        painter.fillRect(layout.header_rect, QColor(self.style.dimmed_header_fill_hex if dimmed else self.style.header_fill_hex))
        painter.fillRect(0, int(layout.row_rect.bottom()), int(layout.row_rect.width()), 1, QColor(self.style.divider_hex))

        painter.setPen(QColor(self.style.dimmed_label_hex if dimmed else self.style.label_hex))
        painter.drawText(layout.label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, take.name)

        button_bg_hex = self.style.options_button_open_fill_hex if options_open else self.style.options_button_closed_fill_hex
        if dimmed:
            button_bg_hex = self.style.options_button_dimmed_fill_hex
        painter.fillRect(layout.options_button_rect, QColor(button_bg_hex))
        painter.setPen(QColor(self.style.options_button_open_text_hex if options_open else self.style.options_button_closed_text_hex))
        button_text = 'Options \u25be' if options_open else 'Options \u25b8'
        prior_font = painter.font()
        button_font = QFont(prior_font)
        button_font.setPointSize(self.style.options_button_font.point_size)
        button_font.setBold(self.style.options_button_font.bold)
        painter.setFont(button_font)
        painter.drawText(
            layout.options_button_rect.adjusted(0, -1, 0, -1),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            button_text,
        )
        painter.setFont(prior_font)

        actions = self._actions_for_take(take)
        action_rects: list[tuple[QRectF, object, object, str]] = []
        if options_open and actions:
            painter.fillRect(layout.options_area_rect, QColor(self.style.options_area_fill_hex))
            x = layout.options_area_rect.left() + 4
            y = layout.options_area_rect.top() + 1
            h = max(12.0, layout.options_area_rect.height() - 2)
            for action in actions:
                chip_w = max(74.0, min(108.0, 14 + (len(action.label) * 5.6)))
                rect = QRectF(x, y, chip_w, h)
                if rect.right() > layout.options_area_rect.right() - 2:
                    break
                painter.fillRect(rect, QColor(self.style.action_chip.fill_hex))
                painter.setPen(QColor(self.style.action_chip.text_hex))
                prior_chip_font = painter.font()
                chip_font = QFont(prior_chip_font)
                chip_font.setPointSize(self.style.action_chip.font.point_size)
                chip_font.setBold(self.style.action_chip.font.bold)
                painter.setFont(chip_font)
                painter.drawText(
                    rect.adjusted(0, -1, 0, -1),
                    Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
                    action.label,
                )
                painter.setFont(prior_chip_font)
                action_rects.append((rect, layer.layer_id, take.take_id, action.action_id))
                x += chip_w + 6

        return TakeRowHitTargets(
            take_rect=(layout.header_rect, layer.layer_id, take.take_id),
            options_toggle_rect=(layout.options_button_rect, layer.layer_id, take.take_id),
            action_rects=action_rects,
        )

    @staticmethod
    def _actions_for_take(take: TakeLanePresentation) -> list[TakeActionPresentation]:
        if take.actions:
            return take.actions
        return [
            TakeActionPresentation(action_id='overwrite_main', label='Overwrite Main'),
            TakeActionPresentation(action_id='merge_main', label='Merge Main'),
        ]
