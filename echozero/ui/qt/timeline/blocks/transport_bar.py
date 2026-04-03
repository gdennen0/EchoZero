from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF


@dataclass(slots=True)
class TransportLayout:
    rect: QRectF
    title_rect: QRectF
    controls_rect: QRectF
    time_rect: QRectF
    meta_rect: QRectF

    @staticmethod
    def create(*, width: float, height: float = 44) -> 'TransportLayout':
        rect = QRectF(0, 0, width, height)
        return TransportLayout(
            rect=rect,
            title_rect=QRectF(12, 8, 160, 24),
            controls_rect=QRectF(180, 8, 120, 28),
            time_rect=QRectF(320, 4, 180, 34),
            meta_rect=QRectF(width - 320, 8, 300, 24),
        )
