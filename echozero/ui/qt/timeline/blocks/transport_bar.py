from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF

from echozero.ui.FEEL import TIMELINE_TRANSPORT_BUTTON_HEIGHT_PX, TIMELINE_TRANSPORT_HEIGHT_PX


@dataclass(slots=True)
class TransportLayout:
    rect: QRectF
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

        horizontal_padding = 24.0
        section_gap = 10.0
        inner_left = horizontal_padding
        inner_right = max(inner_left, width - horizontal_padding)
        inner_width = max(0.0, inner_right - inner_left)

        controls_width = min(360.0, max(184.0, inner_width * 0.44))
        controls_width = min(controls_width, inner_width)
        controls_x = inner_left
        controls_right = controls_x + controls_width

        cursor = controls_right
        time_x = cursor
        time_width = 0.0
        meta_x = inner_right
        meta_width = 0.0

        remaining_after_controls = inner_right - cursor
        if remaining_after_controls > section_gap:
            cursor += section_gap
            available = max(0.0, inner_right - cursor)

            clock_pref_width = 260.0
            clock_min_width = 90.0
            meta_min_width = 96.0

            tentative_clock = min(clock_pref_width, available * 0.58)
            tentative_meta = available - tentative_clock
            if (
                tentative_meta >= section_gap + meta_min_width
                and tentative_clock >= clock_min_width
            ):
                time_x = cursor
                time_width = tentative_clock
                meta_x = time_x + time_width + section_gap
                meta_width = max(0.0, inner_right - meta_x)
            else:
                time_width = min(clock_pref_width, available)
                time_x = cursor + max(0.0, (available - time_width) / 2.0)
                meta_x = inner_right
                meta_width = 0.0

        return TransportLayout(
            rect=rect,
            controls_rect=centered_rect(
                controls_x,
                controls_width,
                float(TIMELINE_TRANSPORT_BUTTON_HEIGHT_PX),
            ),
            time_rect=centered_rect(time_x, time_width, 34.0),
            meta_rect=centered_rect(meta_x, meta_width, 30.0),
        )
