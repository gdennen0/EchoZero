from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF

from echozero.ui.FEEL import TIMELINE_TRANSPORT_BUTTON_HEIGHT_PX, TIMELINE_TRANSPORT_HEIGHT_PX


@dataclass(slots=True)
class TransportLayout:
    rect: QRectF
    title_rect: QRectF
    controls_rect: QRectF
    time_rect: QRectF
    meta_rect: QRectF

    @staticmethod
    def create(
        *,
        width: float,
        height: float = float(TIMELINE_TRANSPORT_HEIGHT_PX),
    ) -> "TransportLayout":
        rect = QRectF(0, 0, width, height)
        center_y = rect.center().y()

        def centered_rect(x: float, item_width: float, item_height: float) -> QRectF:
            return QRectF(x, center_y - (item_height / 2.0), item_width, item_height)

        return TransportLayout(
            rect=rect,
            title_rect=centered_rect(12.0, 160.0, 20.0),
            controls_rect=centered_rect(
                180.0,
                124.0,
                float(TIMELINE_TRANSPORT_BUTTON_HEIGHT_PX),
            ),
            time_rect=centered_rect(324.0, 184.0, 24.0),
            meta_rect=centered_rect(width - 320.0, 300.0, 20.0),
        )
