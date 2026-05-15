from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen

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

        panel_rect = layout.rect.adjusted(1.0, 0.5, -1.0, -0.5)
        panel_fill = QColor(self.style.background_hex).lighter(104)
        panel_border = QColor("#1b222d")
        painter.setPen(QPen(panel_border, 1))
        painter.setBrush(QBrush(panel_fill))
        painter.drawRoundedRect(panel_rect, 3.0, 3.0)

        play_rect, stop_rect, follow_rect = self._button_rects(layout.controls_rect)
        self._draw_button(
            painter,
            play_rect,
            "pause" if presentation.is_playing else "play",
            primary=True,
            active=presentation.is_playing,
        )
        self._draw_button(painter, stop_rect, "stop", primary=False, active=False)
        follow_enabled = presentation.follow_mode != FollowMode.OFF
        self._draw_button(
            painter,
            follow_rect,
            "latch_on" if follow_enabled else "latch_off",
            primary=False,
            active=follow_enabled,
        )

        self._draw_clock_badge(
            painter,
            layout.time_rect,
            f"{presentation.current_time_label} / {presentation.end_time_label}",
        )

        if layout.meta_rect.width() > 1.0 and QGuiApplication.instance() is not None:
            status_color = (
                QColor("#d8d2cb") if presentation.is_playing else QColor(self.style.meta_hex)
            )
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
            meta_draw_rect = layout.meta_rect.adjusted(0.0, -1.0, 0.0, -4.0)
            painter.save()
            painter.setClipRect(meta_draw_rect)
            painter.drawText(
                meta_draw_rect,
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
                | Qt.TextFlag.TextSingleLine,
                meta_text,
            )
            painter.restore()
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
        bpm_text = _format_bpm_text(
            bpm=presentation.bpm,
            bpm_confidence=presentation.bpm_confidence,
        )
        separator = "\u2022"
        candidates = (
            f"{bpm_text}  {separator}  {status_text}  {separator}  {layer_count} layers  {separator}  Zoom: {zoom_speed}",
            f"{bpm_text}  {separator}  {status_text}  {separator}  {layer_count}L  {separator}  {zoom_speed}",
            f"{bpm_text}  {separator}  {status_text}  {separator}  {layer_count}L",
            f"{bpm_text}  {separator}  {status_text}",
            f"{status_text}  {separator}  {layer_count} layers  {separator}  Zoom: {zoom_speed}",
            f"{status_text}  {separator}  {layer_count} layers  {separator}  {zoom_speed}",
            f"{status_text}  {separator}  {layer_count}L  {separator}  {zoom_speed}",
            f"{status_text}  {separator}  {layer_count}L",
            status_text,
            bpm_text,
        )
        max_text_width = max(0, int(available_width) - 4)
        if max_text_width < 8:
            return ""
        for candidate in candidates:
            if font_metrics.horizontalAdvance(candidate) <= max_text_width:
                return candidate
        return font_metrics.elidedText(
            status_text,
            Qt.TextElideMode.ElideRight,
            max_text_width,
        )

    def _button_rects(self, controls_rect: QRectF) -> tuple[QRectF, QRectF, QRectF]:
        button_gap = 6.0
        button_width = max(0.0, (controls_rect.width() - (button_gap * 2.0)) / 3.0)
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

    def _draw_clock_badge(self, painter: QPainter, rect: QRectF, text: str) -> None:
        if rect.width() <= 1.0:
            return
        badge_rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)
        badge_fill = QColor(self.style.background_hex).lighter(118)
        badge_border = QColor("#4a4749")
        painter.setPen(QPen(badge_border, 1))
        painter.setBrush(QBrush(badge_fill))
        painter.drawRoundedRect(badge_rect, 3.0, 3.0)

        if QGuiApplication.instance() is None:
            return
        prior_font = painter.font()
        clock_font = QFont(prior_font)
        clock_font.setPointSize(max(9, prior_font.pointSize() + 1))
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
        if rect.width() <= 1.0 or rect.height() <= 1.0:
            return
        button_style = self.style.button
        fill_color = QColor(button_style.fill_hex)
        border_color = QColor(button_style.border_hex)
        text_color = QColor(button_style.text_hex)
        if active:
            fill_color = QColor("#28262a")
            border_color = QColor("#8f8a84")
            text_color = QColor("#f6f3ee")
        elif primary:
            fill_color = fill_color.lighter(112)
            border_color = QColor("#4a4749")
        else:
            fill_color = fill_color.darker(106)
            border_color = QColor("#3a383a")
        radius = float(max(1, button_style.corner_radius))
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(QBrush(fill_color))
        painter.drawRoundedRect(rect, radius, radius)
        if QGuiApplication.instance() is None:
            return
        self._draw_transport_icon(painter, rect, label, text_color)

    def _draw_transport_icon(
        self,
        painter: QPainter,
        rect: QRectF,
        icon: str,
        color: QColor,
    ) -> None:
        center = rect.center()
        size = max(6.0, min(rect.width(), rect.height()) * 0.44)
        painter.save()
        painter.setPen(QPen(color, 1))
        painter.setBrush(QBrush(color))
        if icon == "play":
            half_h = size * 0.55
            painter.drawPolygon(
                QPointF(center.x() - size * 0.36, center.y() - half_h),
                QPointF(center.x() - size * 0.36, center.y() + half_h),
                QPointF(center.x() + size * 0.48, center.y()),
            )
        elif icon == "pause":
            bar_w = max(2.0, size * 0.20)
            gap = max(2.0, size * 0.18)
            bar_h = size * 1.08
            top = center.y() - (bar_h * 0.5)
            painter.drawRect(QRectF(center.x() - gap * 0.5 - bar_w, top, bar_w, bar_h))
            painter.drawRect(QRectF(center.x() + gap * 0.5, top, bar_w, bar_h))
        elif icon == "stop":
            side = size * 0.78
            painter.drawRect(QRectF(center.x() - side * 0.5, center.y() - side * 0.5, side, side))
        elif icon in {"latch_on", "latch_off"}:
            radius = size * 0.36
            latch_rect = QRectF(
                center.x() - radius,
                center.y() - radius,
                radius * 2.0,
                radius * 2.0,
            )
            if icon == "latch_off":
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(color, 1.4))
            painter.drawEllipse(latch_rect)
        painter.restore()


def _format_bpm_text(*, bpm: float | None, bpm_confidence: float | None) -> str:
    if bpm is None or float(bpm) <= 0.0:
        return "No BPM"
    rounded_bpm = f"{float(bpm):.1f}".rstrip("0").rstrip(".")
    if bpm_confidence is not None and float(bpm_confidence) < 0.6:
        return f"~{rounded_bpm} BPM"
    return f"{rounded_bpm} BPM"
