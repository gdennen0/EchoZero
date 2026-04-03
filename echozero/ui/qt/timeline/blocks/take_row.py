from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter

from echozero.application.presentation.models import LayerPresentation, TakeLanePresentation
from echozero.ui.qt.timeline.blocks.layouts import TakeRowLayout


@dataclass(slots=True)
class TakeRowHitTargets:
    take_rect: tuple[QRectF, object, object]


class TakeRowBlock:
    def paint_header(self, painter: QPainter, layout: TakeRowLayout, layer: LayerPresentation, take: TakeLanePresentation, *, dimmed: bool = False) -> TakeRowHitTargets:
        painter.fillRect(layout.row_rect, QColor('#0f141b' if dimmed else '#121821'))
        painter.fillRect(layout.header_rect, QColor('#141922' if dimmed else '#171d26'))
        painter.fillRect(0, int(layout.row_rect.bottom()), int(layout.row_rect.width()), 1, QColor('#222936'))

        painter.setPen(QColor('#8e98a6' if dimmed else '#aeb8c6'))
        painter.drawText(layout.label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, take.name)

        return TakeRowHitTargets(take_rect=(layout.header_rect, layer.layer_id, take.take_id))
