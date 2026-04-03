from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter

from echozero.application.presentation.models import LayerPresentation, TakeActionPresentation, TakeLanePresentation
from echozero.ui.qt.timeline.blocks.layouts import TakeRowLayout


@dataclass(slots=True)
class TakeRowHitTargets:
    take_rect: tuple[QRectF, object, object]
    options_toggle_rect: tuple[QRectF, object, object] | None = None
    action_rects: list[tuple[QRectF, object, object, str]] = field(default_factory=list)


class TakeRowBlock:
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
        painter.fillRect(layout.row_rect, QColor('#0f141b' if dimmed else '#121821'))
        painter.fillRect(layout.header_rect, QColor('#141922' if dimmed else '#171d26'))
        painter.fillRect(0, int(layout.row_rect.bottom()), int(layout.row_rect.width()), 1, QColor('#222936'))

        painter.setPen(QColor('#8e98a6' if dimmed else '#aeb8c6'))
        painter.drawText(layout.label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, take.name)

        button_bg = QColor('#263244' if options_open else '#1f2938')
        if dimmed:
            button_bg = QColor('#1a2230')
        painter.fillRect(layout.options_button_rect, button_bg)
        painter.setPen(QColor('#9fcbff' if options_open else '#8ea4bf'))
        button_text = 'Options ▾' if options_open else 'Options ▸'
        painter.drawText(layout.options_button_rect, Qt.AlignmentFlag.AlignCenter, button_text)

        actions = self._actions_for_take(take)
        action_rects: list[tuple[QRectF, object, object, str]] = []
        if options_open and actions:
            painter.fillRect(layout.options_area_rect, QColor('#101822'))
            x = layout.options_area_rect.left() + 4
            y = layout.options_area_rect.top() + 1
            h = max(12.0, layout.options_area_rect.height() - 2)
            for action in actions:
                chip_w = max(74.0, min(108.0, 14 + (len(action.label) * 5.6)))
                rect = QRectF(x, y, chip_w, h)
                if rect.right() > layout.options_area_rect.right() - 2:
                    break
                painter.fillRect(rect, QColor('#22364f'))
                painter.setPen(QColor('#d0e4ff'))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, action.label)
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
