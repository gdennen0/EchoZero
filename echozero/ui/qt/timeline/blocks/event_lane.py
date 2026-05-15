from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush

from echozero.application.presentation.models import EventPresentation
from echozero.application.shared.enums import LayerKind
from echozero.perf import timed
from echozero.ui.FEEL import (
    EVENT_LABEL_MIN_WIDTH_PX,
    EVENT_MIN_HIT_WIDTH_PX,
    EVENT_MIN_VISIBLE_WIDTH_PX,
    EVENT_SELECTION_BORDER_PX,
    EVENT_SELECTION_COLOR,
    EVENT_SELECTION_OUTLINE_EXPAND_PX,
    EVENT_SELECTION_TINY_WIDTH_EXTRA_PX,
    EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX,
)
from echozero.ui.qt.timeline.style import EventLaneStyle, TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import (
    CachedWaveform,
    get_cached_waveform,
    register_waveform_from_audio_file,
)
from echozero.ui.qt.timeline.blocks.waveform_lane import (
    iter_compacted_waveform_columns,
    waveform_column_step_px,
)

_EVENT_WAVEFORM_REGISTER_ATTEMPTS: set[str] = set()


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
    event_hit_min_width_px: float | None = None
    dimmed: bool = False
    viewport_width: int = 1440
    default_fill_hex: str | None = None
    waveform_key: str | None = None
    source_audio_path: str | None = None
    render_audio_shape: bool = False


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

    def _resolve_cached_event_waveform(
        self,
        presentation: EventLanePresentation,
    ) -> CachedWaveform | None:
        cached = get_cached_waveform(presentation.waveform_key)
        if cached is not None:
            return cached
        key = str(presentation.waveform_key or "").strip()
        source_audio_path = str(presentation.source_audio_path or "").strip()
        if not key or not source_audio_path:
            return None
        attempt_key = f"{key}|{source_audio_path}"
        if attempt_key in _EVENT_WAVEFORM_REGISTER_ATTEMPTS:
            return None
        _EVENT_WAVEFORM_REGISTER_ATTEMPTS.add(attempt_key)
        try:
            register_waveform_from_audio_file(key, source_audio_path)
        except Exception:
            return None
        return get_cached_waveform(key)

    def _paint_event_audio_shape(
        self,
        painter: QPainter,
        *,
        rect: QRectF,
        event_start: float,
        presentation: EventLanePresentation,
        cached: CachedWaveform,
        base_color: QColor,
    ) -> None:
        if cached.peaks.size == 0:
            return
        pps = max(1.0, presentation.pixels_per_second)
        header_width = float(presentation.header_width)
        visible_left_x = max(float(rect.left()), header_width)
        visible_right_x = min(float(rect.right()), float(presentation.viewport_width))
        if visible_right_x <= header_width or visible_right_x <= visible_left_x:
            return
        start_time = max(
            0.0,
            (presentation.scroll_x + visible_left_x - header_width) / pps - event_start,
        )
        end_time = min(
            float(cached.resolved_duration_seconds),
            max(
                start_time,
                (presentation.scroll_x + visible_right_x - header_width) / pps - event_start,
            ),
        )
        if end_time <= start_time:
            return
        spp = cached.seconds_per_peak
        start_idx = max(0, int(floor(start_time / spp)) - 1)
        end_idx = min(cached.peaks.shape[0] - 1, int(ceil(end_time / spp)) + 1)
        if end_idx < start_idx:
            return

        shape_color = QColor(base_color).lighter(138)
        shape_color.setAlpha(160 if not presentation.dimmed else 110)
        center_y = rect.center().y()
        amp_px = max(1.0, rect.height() * 0.34)
        painter.save()
        painter.setClipRect(rect.adjusted(1.0, 1.0, -1.0, -1.0))
        painter.setPen(QPen(shape_color, 1))
        adjusted_scroll_x = presentation.scroll_x - (event_start * pps)
        for x, vmin, vmax in iter_compacted_waveform_columns(
            cached=cached,
            start_idx=start_idx,
            end_idx=end_idx,
            pixels_per_second=pps,
            scroll_x=adjusted_scroll_x,
            content_start_x=float(presentation.header_width),
            pixel_step_px=waveform_column_step_px(presentation.pixels_per_second),
        ):
            if x < int(rect.left()) or x > int(rect.right()):
                continue
            y1 = center_y - (float(vmax) * amp_px)
            y2 = center_y - (float(vmin) * amp_px)
            painter.drawLine(int(x), int(y1), int(x), int(y2))
        painter.restore()

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
        visible_end_t = max(
            visible_start_t, (presentation.scroll_x + max(1.0, content_right - content_left)) / pps
        )

        cached_event_waveform = (
            self._resolve_cached_event_waveform(presentation)
            if presentation.render_audio_shape
            else None
        )
        audio_shape_duration = (
            float(cached_event_waveform.resolved_duration_seconds)
            if cached_event_waveform is not None
            else 0.0
        )

        with timed("timeline.paint.event_lane"):
            for event in presentation.events:
                visual_duration = (
                    audio_shape_duration if audio_shape_duration > 0.0 else event.duration
                )
                visual_end = float(event.start) + max(0.0, float(visual_duration))
                if visual_end < visible_start_t:
                    continue
                if event.start > visible_end_t:
                    break

                x = presentation.header_width + (event.start * pps) - presentation.scroll_x
                raw_width = max(0.0, visual_duration * pps)
                width = max(float(EVENT_MIN_VISIBLE_WIDTH_PX), raw_width)
                if x + width < content_left - 2 or x > content_right + 2:
                    continue

                rect = QRectF(x, top_y, width, presentation.event_height)
                is_zoomed_out_event = raw_width <= float(EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX)
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
                elif is_zoomed_out_event:
                    color = color.lighter(112)
                rendered_rect = rect
                border_width = (
                    self.style.selected_border_width_px
                    if event.is_selected
                    else self.style.normal_border_width_px
                )
                border_color = (
                    color.lighter(128)
                    if is_zoomed_out_event and not event.is_selected
                    else color.darker(self.style.border_darkness_factor)
                )
                painter.setPen(QPen(border_color, border_width))
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(rect, self.style.corner_radius, self.style.corner_radius)
                if cached_event_waveform is not None and raw_width >= 8.0:
                    self._paint_event_audio_shape(
                        painter,
                        rect=rect,
                        event_start=float(event.start),
                        presentation=presentation,
                        cached=cached_event_waveform,
                        base_color=color,
                    )
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
                hit_rect = rendered_rect
                min_hit_width = max(
                    float(EVENT_MIN_HIT_WIDTH_PX),
                    float(presentation.event_hit_min_width_px or 0.0),
                )
                if min_hit_width > rendered_rect.width():
                    available_width = max(0.0, content_right - content_left)
                    target_width = min(min_hit_width, available_width)
                    if target_width > rendered_rect.width():
                        left = rendered_rect.center().x() - (target_width * 0.5)
                        left = max(content_left, min(left, content_right - target_width))
                        hit_rect = QRectF(
                            left,
                            rendered_rect.y(),
                            target_width,
                            rendered_rect.height(),
                        )
                rects.append(
                    (hit_rect, presentation.layer_id, presentation.take_id, event.event_id)
                )
        return rects
