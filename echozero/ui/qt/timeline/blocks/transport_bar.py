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

        horizontal_padding = 12.0
        section_gap = 12.0
        title_width = max(130.0, min(220.0, width * 0.2))
        controls_width = 256.0
        meta_width = max(190.0, min(360.0, width * 0.26))
        min_clock_width = 170.0

        title_x = horizontal_padding
        controls_x = title_x + title_width + section_gap
        controls_right = controls_x + controls_width

        meta_right = max(horizontal_padding, width - horizontal_padding)
        meta_left = max(
            controls_right + section_gap + min_clock_width + section_gap,
            meta_right - meta_width,
        )
        resolved_meta_width = max(120.0, meta_right - meta_left)

        clock_left = controls_right + section_gap
        clock_right = meta_left - section_gap
        resolved_clock_width = max(0.0, clock_right - clock_left)
        clock_width = max(0.0, min(280.0, resolved_clock_width))
        clock_x = clock_left + max(0.0, (resolved_clock_width - clock_width) / 2.0)

        return TransportLayout(
            rect=rect,
            title_rect=centered_rect(title_x, title_width, 30.0),
            controls_rect=centered_rect(
                controls_x,
                controls_width,
                float(TIMELINE_TRANSPORT_BUTTON_HEIGHT_PX),
            ),
            time_rect=centered_rect(clock_x, clock_width, 34.0),
            meta_rect=centered_rect(meta_left, resolved_meta_width, 30.0),
        )
