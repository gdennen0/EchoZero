from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import EventPresentation
from echozero.application.shared.cue_numbers import cue_number_text
from echozero.application.shared.enums import LayerKind
from echozero.perf import timed
from echozero.ui.FEEL import (
    EVENT_LABEL_MIN_WIDTH_PX,
    EVENT_MIN_VISIBLE_WIDTH_PX,
    EVENT_SELECTION_BORDER_PX,
    EVENT_SELECTION_COLOR,
    EVENT_SELECTION_OUTLINE_EXPAND_PX,
    EVENT_SELECTION_TINY_WIDTH_EXTRA_PX,
    EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX,
)
from echozero.ui.qt.timeline.style import EventLaneStyle, TIMELINE_STYLE

_MARKER_LABEL_MIN_WIDTH_PX = 88.0
_MARKER_LABEL_HORIZONTAL_PADDING_PX = 8.0
_MARKER_LABEL_BOTTOM_INSET_PX = 7.0
_MARKER_STEM_HALF_WIDTH_PX = 1.0


@dataclass(slots=True)
class EventLanePresentation:
    layer_id: object
    take_id: object | None
    events: list[EventPresentation]
    pixels_per_second: float
    scroll_x: float
    header_width: int
    layer_kind: LayerKind = LayerKind.EVENT
    event_height: int = 22
    dimmed: bool = False
    viewport_width: int = 1440
    default_fill_hex: str | None = None


class EventLaneBlock:
    def __init__(self, style: EventLaneStyle = TIMELINE_STYLE.event_lane):
        self.style = style

    def _selected_outline_width_px(self, event_width: float) -> int:
        base_width = max(EVENT_SELECTION_BORDER_PX, self.style.selected_border_width_px)
        if event_width <= EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX:
            return base_width + EVENT_SELECTION_TINY_WIDTH_EXTRA_PX
        return base_width

    def _paint_selected_outline(
        self,
        painter: QPainter,
        *,
        rect: QRectF,
        corner_radius: float,
        dimmed: bool,
    ) -> None:
        outline = QColor(EVENT_SELECTION_COLOR)
        if dimmed:
            outline.setAlpha(210)
        outline_width = self._selected_outline_width_px(rect.width())
        expand = float(EVENT_SELECTION_OUTLINE_EXPAND_PX)
        painter.save()
        painter.setPen(QPen(outline, outline_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            rect.adjusted(-expand, -expand, expand, expand),
            corner_radius + expand,
            corner_radius + expand,
        )
        painter.restore()

    @staticmethod
    def marker_event_label(event: EventPresentation) -> str:
        cue_number = cue_number_text(getattr(event, "cue_number", None))
        label = str(event.label or "").strip()
        if cue_number is None:
            return label
        prefix = f"Q{cue_number}"
        if not label:
            return prefix
        normalized_label = label.casefold()
        normalized_prefix = prefix.casefold()
        cue_label = f"Cue {cue_number}"
        if normalized_label in {normalized_prefix, cue_label.casefold()}:
            return prefix
        if normalized_label.startswith(normalized_prefix):
            return label
        if normalized_label.startswith(cue_label.casefold()):
            suffix = label[len(cue_label) :].strip(" -:|")
            return prefix if not suffix else f"{prefix} {suffix}"
        return f"{prefix} {label}"

    @staticmethod
    def section_event_label(event: EventPresentation) -> str:
        cue_ref = str(event.cue_ref or "").strip()
        label = str(event.label or "").strip()
        if not cue_ref:
            return label or "Section"
        if not label:
            return cue_ref
        if label.casefold() == cue_ref.casefold():
            return cue_ref
        if label.casefold().startswith(cue_ref.casefold()):
            return label
        return f"{cue_ref} {label}"

    def paint(
        self,
        painter: QPainter,
        top_y: int,
        presentation: EventLanePresentation,
    ) -> list[tuple[QRectF, object, object | None, object]]:
        rects: list[tuple[QRectF, object, object | None, object]] = []

        pps = max(1.0, presentation.pixels_per_second)
        content_left = float(presentation.header_width)
        content_right = float(max(presentation.header_width + 1, presentation.viewport_width))
        visible_start_t = max(0.0, presentation.scroll_x / pps)
        visible_end_t = max(visible_start_t, (presentation.scroll_x + max(1.0, content_right - content_left)) / pps)

        with timed("timeline.paint.event_lane"):
            for event in presentation.events:
                if event.end < visible_start_t:
                    continue
                if event.start > visible_end_t:
                    break

                x = presentation.header_width + (event.start * pps) - presentation.scroll_x
                width = max(float(EVENT_MIN_VISIBLE_WIDTH_PX), (event.duration * pps))
                if x + width < content_left - 2 or x > content_right + 2:
                    continue

                rect = QRectF(x, top_y, width, presentation.event_height)
                badge_tokens = {
                    str(badge).strip().lower()
                    for badge in getattr(event, "badges", [])
                    if str(badge).strip()
                }
                if "demoted" in badge_tokens:
                    color = QColor(self.style.demoted_fill_hex)
                else:
                    color = QColor(
                        event.color or presentation.default_fill_hex or self.style.default_fill_hex
                    )
                if presentation.dimmed:
                    color.setAlpha(self.style.dimmed_alpha)
                if event.is_selected:
                    color = color.lighter(self.style.selection_lighten_factor)
                if presentation.layer_kind is LayerKind.MARKER:
                    rendered_rect = self._paint_marker_event(
                        painter,
                        rect=rect,
                        event=event,
                        color=color,
                        content_right=content_right,
                        dimmed=presentation.dimmed,
                    )
                elif presentation.layer_kind is LayerKind.SECTION:
                    rendered_rect = self._paint_section_event(
                        painter,
                        rect=rect,
                        event=event,
                        color=color,
                        content_right=content_right,
                        dimmed=presentation.dimmed,
                    )
                else:
                    rendered_rect = rect
                    border_width = (
                        self.style.selected_border_width_px
                        if event.is_selected
                        else self.style.normal_border_width_px
                    )
                    painter.setPen(QPen(color.darker(self.style.border_darkness_factor), border_width))
                    painter.setBrush(QBrush(color))
                    painter.drawRoundedRect(rect, self.style.corner_radius, self.style.corner_radius)
                    if event.is_selected:
                        self._paint_selected_outline(
                            painter,
                            rect=rect,
                            corner_radius=float(self.style.corner_radius),
                            dimmed=presentation.dimmed,
                        )

                    if width >= EVENT_LABEL_MIN_WIDTH_PX:
                        painter.setPen(QColor(self.style.text_hex))
                        painter.drawText(
                            QRectF(x + 6, top_y, max(0, width - 12), presentation.event_height),
                            Qt.AlignmentFlag.AlignVCenter,
                            event.label,
                        )
                rects.append(
                    (rendered_rect, presentation.layer_id, presentation.take_id, event.event_id)
                )
        return rects

    def _paint_marker_event(
        self,
        painter: QPainter,
        *,
        rect: QRectF,
        event: EventPresentation,
        color: QColor,
        content_right: float,
        dimmed: bool,
    ) -> QRectF:
        label_text = self.marker_event_label(event)
        label_height = max(16.0, rect.height() - _MARKER_LABEL_BOTTOM_INSET_PX)
        label_left = rect.left()
        available_width = max(18.0, content_right - label_left - 4.0)
        desired_width = self._marker_label_width(painter, label_text)
        label_width = min(available_width, desired_width)
        label_rect = QRectF(label_left, rect.top(), label_width, label_height)
        text_width = max(
            0,
            int(round(label_rect.width() - (_MARKER_LABEL_HORIZONTAL_PADDING_PX * 2.0))),
        )
        text = self._marker_label_text(painter, label_text, text_width)

        painter.setPen(
            QPen(
                color.darker(self.style.border_darkness_factor),
                self.style.selected_border_width_px if event.is_selected else self.style.normal_border_width_px,
            )
        )
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(label_rect, self.style.corner_radius + 1, self.style.corner_radius + 1)
        if event.is_selected:
            self._paint_selected_outline(
                painter,
                rect=label_rect,
                corner_radius=float(self.style.corner_radius + 1),
                dimmed=dimmed,
            )

        if QApplication.instance() is not None and text:
            painter.setPen(QColor(self.style.text_hex))
            painter.drawText(
                label_rect.adjusted(
                    _MARKER_LABEL_HORIZONTAL_PADDING_PX,
                    0.0,
                    -_MARKER_LABEL_HORIZONTAL_PADDING_PX,
                    0.0,
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )

        stem_x = rect.left() + min(8.0, max(0.0, rect.width() * 0.5))
        stem_top = label_rect.bottom() - 1.0
        stem_bottom = rect.bottom()
        painter.drawLine(int(round(stem_x)), int(round(stem_top)), int(round(stem_x)), int(round(stem_bottom)))
        stem_rect = QRectF(
            stem_x - _MARKER_STEM_HALF_WIDTH_PX,
            stem_top,
            _MARKER_STEM_HALF_WIDTH_PX * 2.0,
            max(1.0, stem_bottom - stem_top),
        )
        return label_rect.united(stem_rect)

    def _paint_section_event(
        self,
        painter: QPainter,
        *,
        rect: QRectF,
        event: EventPresentation,
        color: QColor,
        content_right: float,
        dimmed: bool,
    ) -> QRectF:
        label_text = self.section_event_label(event)
        label_height = max(16.0, rect.height() - 2.0)
        label_left = rect.left()
        available_width = max(20.0, content_right - label_left - 4.0)
        desired_width = self._marker_label_width(painter, label_text)
        label_width = min(available_width, max(_MARKER_LABEL_MIN_WIDTH_PX, desired_width))
        label_rect = QRectF(label_left, rect.top(), label_width, label_height)
        text_width = max(
            0,
            int(round(label_rect.width() - (_MARKER_LABEL_HORIZONTAL_PADDING_PX * 2.0))),
        )
        text = self._marker_label_text(painter, label_text, text_width)

        painter.setPen(
            QPen(
                color.darker(self.style.border_darkness_factor),
                self.style.selected_border_width_px if event.is_selected else self.style.normal_border_width_px,
            )
        )
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(label_rect, self.style.corner_radius + 1, self.style.corner_radius + 1)
        if event.is_selected:
            self._paint_selected_outline(
                painter,
                rect=label_rect,
                corner_radius=float(self.style.corner_radius + 1),
                dimmed=dimmed,
            )

        if QApplication.instance() is not None and text:
            painter.setPen(QColor(self.style.text_hex))
            painter.drawText(
                label_rect.adjusted(
                    _MARKER_LABEL_HORIZONTAL_PADDING_PX,
                    0.0,
                    -_MARKER_LABEL_HORIZONTAL_PADDING_PX,
                    0.0,
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )

        boundary_x = rect.left()
        painter.drawLine(
            int(round(boundary_x)),
            int(round(label_rect.bottom() - 1.0)),
            int(round(boundary_x)),
            int(round(rect.bottom())),
        )
        boundary_rect = QRectF(
            boundary_x - _MARKER_STEM_HALF_WIDTH_PX,
            label_rect.bottom() - 1.0,
            _MARKER_STEM_HALF_WIDTH_PX * 2.0,
            max(1.0, rect.bottom() - label_rect.bottom() + 1.0),
        )
        return label_rect.united(boundary_rect)

    @staticmethod
    def _marker_label_text(painter: QPainter, label_text: str, text_width: int) -> str:
        if text_width <= 0:
            return ""
        if QApplication.instance() is None:
            estimated_chars = max(1, int(text_width // 7))
            if len(label_text) <= estimated_chars:
                return label_text
            if estimated_chars <= 1:
                return label_text[:estimated_chars]
            return f"{label_text[: max(0, estimated_chars - 1)]}…"
        font_metrics = painter.fontMetrics()
        return font_metrics.elidedText(
            label_text,
            Qt.TextElideMode.ElideRight,
            text_width,
        )

    @staticmethod
    def _marker_label_width(painter: QPainter, label_text: str) -> float:
        if QApplication.instance() is None:
            return max(
                _MARKER_LABEL_MIN_WIDTH_PX,
                (len(label_text) * 7.0) + (_MARKER_LABEL_HORIZONTAL_PADDING_PX * 2.0),
            )
        font_metrics = painter.fontMetrics()
        return max(
            _MARKER_LABEL_MIN_WIDTH_PX,
            float(font_metrics.horizontalAdvance(label_text))
            + (_MARKER_LABEL_HORIZONTAL_PADDING_PX * 2.0),
        )
