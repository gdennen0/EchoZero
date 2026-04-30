from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen

from echozero.application.shared.enums import FollowMode
from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, TransportBarStyle


class TransportBarBlock:
    def __init__(self, style: TransportBarStyle = TIMELINE_STYLE.transport_bar):
        self.style = style

    def paint(
        self,
        painter: QPainter,
        layout: TransportLayout,
        presentation: TimelinePresentation,
    ) -> dict[str, object]:
        painter.fillRect(layout.rect, QColor(self.style.background_hex))

        panel_rect = layout.rect.adjusted(8.0, 4.0, -8.0, -4.0)
        panel_fill = QColor(self.style.background_hex).lighter(112)
        panel_border = QColor(self.style.button.border_hex)
        panel_border.setAlpha(220)
        painter.setPen(QPen(panel_border, 1))
        painter.setBrush(QBrush(panel_fill))
        painter.drawRoundedRect(panel_rect, 10, 10)

        side_text_rect = layout.title_rect.adjusted(0.0, -1.0, 0.0, -4.0)
        title_label_rect = QRectF(
            side_text_rect.left(),
            side_text_rect.top(),
            side_text_rect.width(),
            side_text_rect.height() * 0.46,
        )
        title_value_rect = QRectF(
            side_text_rect.left(),
            title_label_rect.bottom(),
            side_text_rect.width(),
            max(0.0, side_text_rect.height() - title_label_rect.height()),
        )
        self._draw_title_block(
            painter,
            title_label_rect,
            title_value_rect,
            presentation.title.strip() or "Timeline",
        )

        play_rect, stop_rect, follow_rect = self._button_rects(layout.controls_rect)
        self._draw_button(
            painter,
            play_rect,
            "⏸ Pause" if presentation.is_playing else "▶ Play",
            primary=True,
            active=presentation.is_playing,
        )
        self._draw_button(painter, stop_rect, "■ Stop", primary=False, active=False)
        follow_enabled = presentation.follow_mode != FollowMode.OFF
        self._draw_button(
            painter,
            follow_rect,
            "◎ Follow" if follow_enabled else "○ Follow",
            primary=False,
            active=follow_enabled,
        )

        self._draw_clock_badge(
            painter,
            layout.time_rect,
            f"{presentation.current_time_label} / {presentation.end_time_label}",
        )

        status_color = QColor("#7fd1ae") if presentation.is_playing else QColor(self.style.meta_hex)
        painter.setPen(status_color)
        prior_font = painter.font()
        meta_font = QFont(prior_font)
        meta_font.setPointSize(max(8, prior_font.pointSize() - 1))
        painter.setFont(meta_font)
        meta_text = self._status_meta_text(
            presentation=presentation,
            available_width=layout.meta_rect.width(),
            font_metrics=painter.fontMetrics(),
        )
        painter.drawText(
            layout.meta_rect.adjusted(0.0, -1.0, 0.0, -4.0),
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextSingleLine,
            meta_text,
        )
        painter.setFont(prior_font)

        return {
            "play": play_rect,
            "stop": stop_rect,
            "follow": follow_rect,
        }

    def _status_meta_text(
        self,
        *,
        presentation: TimelinePresentation,
        available_width: float,
        font_metrics: QFontMetrics,
    ) -> str:
        status_text = "PLAYING" if presentation.is_playing else "STOPPED"
        layer_count = len(presentation.layers)
        zoom_speed = f"{presentation.pixels_per_second:.0f}px/s"
        separator = "\u2022"
        candidates = (
            f"{status_text}  {separator}  {layer_count} layers  {separator}  Zoom: {zoom_speed}",
            f"{status_text}  {separator}  {layer_count} layers  {separator}  {zoom_speed}",
            f"{status_text}  {separator}  {layer_count}L  {separator}  {zoom_speed}",
            f"{status_text}  {separator}  {layer_count}L",
            status_text,
        )
        max_text_width = max(0, int(available_width) - 4)
        for candidate in candidates:
            if font_metrics.horizontalAdvance(candidate) <= max_text_width:
                return candidate
        return font_metrics.elidedText(
            status_text,
            Qt.TextElideMode.ElideRight,
            max_text_width,
        )

    def _button_rects(self, controls_rect: QRectF) -> tuple[QRectF, QRectF, QRectF]:
        button_gap = 8.0
        button_width = (controls_rect.width() - (button_gap * 2.0)) / 3.0
        play_rect = QRectF(
            controls_rect.left(),
            controls_rect.top(),
            button_width,
            controls_rect.height(),
        )
        stop_rect = QRectF(
            play_rect.right() + button_gap,
            controls_rect.top(),
            button_width,
            controls_rect.height(),
        )
        follow_rect = QRectF(
            stop_rect.right() + button_gap,
            controls_rect.top(),
            button_width,
            controls_rect.height(),
        )
        return play_rect, stop_rect, follow_rect

    def _draw_title_block(
        self,
        painter: QPainter,
        label_rect: QRectF,
        value_rect: QRectF,
        value: str,
    ) -> None:
        prior_font = painter.font()
        label_font = QFont(prior_font)
        label_font.setPointSize(max(8, prior_font.pointSize() - 1))
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.setPen(QColor(self.style.meta_hex))
        painter.drawText(
            label_rect,
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignBottom
            | Qt.TextFlag.TextSingleLine,
            "PLAYBACK",
        )

        value_font = QFont(prior_font)
        value_font.setPointSize(max(9, prior_font.pointSize() + 1))
        value_font.setBold(True)
        painter.setFont(value_font)
        painter.setPen(QColor(self.style.title_hex))
        painter.drawText(
            value_rect,
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextSingleLine,
            value,
        )
        painter.setFont(prior_font)

    def _draw_clock_badge(self, painter: QPainter, rect: QRectF, text: str) -> None:
        if rect.width() <= 1.0:
            return
        badge_rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)
        badge_fill = QColor(self.style.background_hex).lighter(132)
        badge_border = QColor(self.style.button.border_hex).lighter(120)
        painter.setPen(QPen(badge_border, 1))
        painter.setBrush(QBrush(badge_fill))
        painter.drawRoundedRect(badge_rect, 8, 8)

        prior_font = painter.font()
        clock_font = QFont(prior_font)
        clock_font.setPointSize(max(10, prior_font.pointSize() + 2))
        clock_font.setBold(True)
        painter.setFont(clock_font)
        painter.setPen(QColor(self.style.time_hex))
        painter.drawText(
            badge_rect,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            text,
        )
        painter.setFont(prior_font)

    def _draw_button(
        self,
        painter: QPainter,
        rect: QRectF,
        label: str,
        *,
        primary: bool,
        active: bool,
    ) -> None:
        button_style = self.style.button
        fill_color = QColor(button_style.fill_hex)
        border_color = QColor(button_style.border_hex)
        text_color = QColor(button_style.text_hex)
        if active:
            fill_color = QColor("#2a6a45")
            border_color = QColor("#57a678")
        elif primary:
            fill_color = fill_color.lighter(122)
            border_color = border_color.lighter(132)
        else:
            fill_color = fill_color.darker(108)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(QBrush(fill_color))
        painter.drawRoundedRect(rect, button_style.corner_radius, button_style.corner_radius)
        painter.setPen(text_color)
        prior_font = painter.font()
        button_font = QFont(prior_font)
        button_font.setPointSize(max(9, button_style.font.point_size))
        button_font.setBold(button_style.font.bold)
        painter.setFont(button_font)
        painter.drawText(
            rect.adjusted(0, -1, 0, -1),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            label,
        )
        painter.setFont(prior_font)
