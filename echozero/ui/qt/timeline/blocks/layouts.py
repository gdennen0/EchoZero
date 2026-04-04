from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF


@dataclass(slots=True)
class MainRowLayout:
    row_rect: QRectF
    header_rect: QRectF
    content_rect: QRectF
    title_rect: QRectF
    subtitle_rect: QRectF
    status_rect: QRectF
    controls_rect: QRectF
    toggle_rect: QRectF
    metadata_rect: QRectF

    @staticmethod
    def create(*, top: float, width: float, header_width: float, row_height: float) -> 'MainRowLayout':
        row_rect = QRectF(0, top, width, row_height - 1)
        header_rect = QRectF(0, top, header_width, row_height - 1)
        content_rect = QRectF(header_width, top, width - header_width, row_height - 1)
        return MainRowLayout(
            row_rect=row_rect,
            header_rect=header_rect,
            content_rect=content_rect,
            title_rect=QRectF(14, top + 7, 170, 18),
            subtitle_rect=QRectF(14, top + 24, 0, 0),
            status_rect=QRectF(14, top + 46, 170, 16),
            controls_rect=QRectF(198, top + 14, 56, 18),
            toggle_rect=QRectF(278, top + 12, 28, 18),
            metadata_rect=QRectF(14, top + 28, 170, 14),
        )


@dataclass(slots=True)
class TakeRowLayout:
    row_rect: QRectF
    header_rect: QRectF
    content_rect: QRectF
    label_rect: QRectF
    options_button_rect: QRectF
    options_area_rect: QRectF

    @staticmethod
    def create(*, top: float, width: float, header_width: float, row_height: float) -> 'TakeRowLayout':
        row_rect = QRectF(0, top, width, row_height - 1)
        header_rect = QRectF(0, top, header_width, row_height - 1)
        content_rect = QRectF(header_width, top, width - header_width, row_height - 1)
        options_button_rect = QRectF(header_width - 98, top + 4, 84, 16)
        options_area_rect = QRectF(26, top + 24, header_width - 52, 16)
        return TakeRowLayout(
            row_rect=row_rect,
            header_rect=header_rect,
            content_rect=content_rect,
            label_rect=QRectF(26, top + 4, header_width - 132, 16),
            options_button_rect=options_button_rect,
            options_area_rect=options_area_rect,
        )
