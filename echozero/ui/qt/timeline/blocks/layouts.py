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
    active_rect: QRectF
    toggle_rect: QRectF
    metadata_rect: QRectF

    @staticmethod
    def create(*, top: float, width: float, header_width: float, row_height: float) -> 'MainRowLayout':
        safe_height = max(24.0, float(row_height))
        header_left = 14.0
        controls_top = top + min(14.0, max(6.0, (safe_height - 18.0) * 0.35))
        toggle_top = top + min(12.0, max(6.0, (safe_height - 16.0) * 0.30))
        toggle_rect = QRectF(header_width - 28, toggle_top, 16, 16)
        active_rect = QRectF(toggle_rect.left() - 20, toggle_top + 1, 14, 14)
        controls_left = 144.0
        controls_right = active_rect.left() - 8.0
        controls_rect = QRectF(
            controls_left,
            controls_top,
            max(0.0, controls_right - controls_left),
            18,
        )
        title_right = min(controls_rect.left(), active_rect.left(), toggle_rect.left()) - 8.0
        row_rect = QRectF(0, top, width, safe_height - 1)
        header_rect = QRectF(0, top, header_width, safe_height - 1)
        content_rect = QRectF(header_width, top, width - header_width, safe_height - 1)
        status_height = 16.0 if safe_height >= 62.0 else 0.0
        status_top = top + safe_height - (status_height + 2.0) if status_height > 0.0 else top
        metadata_height = 14.0 if safe_height >= 54.0 else 0.0
        metadata_top = (
            top + safe_height - (metadata_height + 6.0) if metadata_height > 0.0 else top
        )
        return MainRowLayout(
            row_rect=row_rect,
            header_rect=header_rect,
            content_rect=content_rect,
            title_rect=QRectF(
                header_left,
                top + min(7.0, max(4.0, controls_top - 3.0)),
                max(0.0, title_right - header_left),
                18.0 if safe_height >= 60.0 else 16.0,
            ),
            subtitle_rect=QRectF(14, top + 24, 0, 0),
            status_rect=QRectF(14, status_top, 170, status_height),
            controls_rect=controls_rect,
            active_rect=active_rect,
            toggle_rect=toggle_rect,
            metadata_rect=QRectF(14, metadata_top, 170, metadata_height),
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
