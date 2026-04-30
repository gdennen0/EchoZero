from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterator

import numpy as np
from PyQt6.QtGui import QColor, QPainter, QPen

from echozero.ui.FEEL import WAVEFORM_COLUMN_STEP_MAX_PX, WAVEFORM_COLUMN_STEP_REFERENCE_PPS
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, WaveformLaneStyle
from echozero.ui.qt.timeline.waveform_cache import (
    CachedWaveform,
    get_cached_waveform,
    register_waveform_from_audio_file,
)


@dataclass(slots=True)
class WaveformLanePresentation:
    color_hex: str
    row_height: int
    pixels_per_second: float
    scroll_x: float
    header_width: int
    width: int
    dimmed: bool = False
    waveform_key: str | None = None
    source_audio_path: str | None = None
    unavailable_reason: str | None = None


_WAVEFORM_REGISTER_ATTEMPTS: set[str] = set()


class WaveformLaneBlock:
    def __init__(self, style: WaveformLaneStyle = TIMELINE_STYLE.waveform_lane):
        self.style = style

    def paint(self, painter: QPainter, top: int, presentation: WaveformLanePresentation) -> None:
        base = QColor(presentation.color_hex)
        if presentation.dimmed:
            base.setAlpha(self.style.dimmed_alpha)

        cached = self._resolve_cached_waveform(presentation)
        if cached is not None:
            self._paint_cached_waveform(painter, top, presentation, cached, base)
            return

        self._paint_waveform_unavailable_state(painter, top, presentation, base)

    def _resolve_cached_waveform(
        self,
        presentation: WaveformLanePresentation,
    ) -> CachedWaveform | None:
        cached = get_cached_waveform(presentation.waveform_key)
        if cached is not None:
            return cached
        key = str(presentation.waveform_key or "").strip()
        source_audio_path = str(presentation.source_audio_path or "").strip()
        if not key or not source_audio_path:
            return None
        attempt_key = f"{key}|{source_audio_path}"
        if attempt_key in _WAVEFORM_REGISTER_ATTEMPTS:
            return None
        _WAVEFORM_REGISTER_ATTEMPTS.add(attempt_key)
        try:
            register_waveform_from_audio_file(key, source_audio_path)
        except Exception:
            return None
        return get_cached_waveform(key)

    def _paint_waveform_unavailable_state(
        self,
        painter: QPainter,
        top: int,
        presentation: WaveformLanePresentation,
        base: QColor,
    ) -> None:
        reason = str(presentation.unavailable_reason or "").strip() or "Waveform unavailable"
        text_color = QColor(base)
        text_color.setAlpha(210 if not presentation.dimmed else 170)
        painter.setPen(QPen(text_color, 1))
        painter.drawText(
            int(presentation.header_width + 12),
            int(top + max(16.0, presentation.row_height * 0.55)),
            reason,
        )

    def _paint_cached_waveform(
        self,
        painter: QPainter,
        top: int,
        presentation: WaveformLanePresentation,
        cached: CachedWaveform,
        color: QColor,
    ) -> None:
        # Alignment contract: waveform time origin must match ruler/event/playhead
        # origin (header_width) exactly, with no extra inset offset.
        content_left = float(presentation.header_width)
        content_right = max(content_left + 1.0, float(presentation.width))
        content_width = max(1.0, content_right - content_left)
        pps = max(1.0, presentation.pixels_per_second)

        start_time = max(0.0, presentation.scroll_x / pps)
        end_time = max(start_time, (presentation.scroll_x + content_width) / pps)

        spp = cached.seconds_per_peak
        start_idx = max(0, int(floor(start_time / spp)) - 1)
        end_idx = min(cached.peaks.shape[0] - 1, int(ceil(end_time / spp)) + 1)
        if end_idx < start_idx:
            return

        center_y = top + (presentation.row_height / 2.0)
        amp_px = presentation.row_height * self.style.cached_amp_row_factor
        column_step_px = waveform_column_step_px(presentation.pixels_per_second)

        painter.setPen(QPen(color, self.style.cached_pen_width_px))
        for x, vmin, vmax in iter_compacted_waveform_columns(
            cached=cached,
            start_idx=start_idx,
            end_idx=end_idx,
            pixels_per_second=pps,
            scroll_x=presentation.scroll_x,
            content_start_x=content_left,
            pixel_step_px=column_step_px,
        ):
            if x < int(content_left - 1) or x > int(content_right + 1):
                continue
            y1 = center_y - (float(vmax) * amp_px)
            y2 = center_y - (float(vmin) * amp_px)
            painter.drawLine(x, int(y1), x, int(y2))


def waveform_x_for_time(
    time_seconds: float,
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_start_x: float,
) -> float:
    pps = max(1.0, pixels_per_second)
    return content_start_x + (max(0.0, time_seconds) * pps) - scroll_x


def iter_compacted_waveform_columns(
    *,
    cached: CachedWaveform,
    start_idx: int,
    end_idx: int,
    pixels_per_second: float,
    scroll_x: float,
    content_start_x: float,
    pixel_step_px: int = 1,
) -> Iterator[tuple[int, float, float]]:
    """Yield one min/max envelope per on-screen pixel column."""
    if cached.peaks.size == 0 or end_idx < start_idx:
        return

    start = max(0, int(start_idx))
    end = min(int(end_idx), int(cached.peaks.shape[0] - 1))
    if end < start:
        return

    step_px = max(1, int(pixel_step_px))
    x_buckets, mins, maxs = _compact_peak_span_numpy(
        peaks=cached.peaks,
        start=start,
        end=end,
        seconds_per_peak=float(cached.seconds_per_peak),
        pixels_per_second=max(1.0, float(pixels_per_second)),
        scroll_x=float(scroll_x),
        content_start_x=float(content_start_x),
        pixel_step_px=step_px,
    )
    for i in range(int(x_buckets.shape[0])):
        yield int(x_buckets[i]), float(mins[i]), float(maxs[i])


def waveform_column_step_px(pixels_per_second: float) -> int:
    pps = max(1.0, float(pixels_per_second))
    if pps >= float(WAVEFORM_COLUMN_STEP_REFERENCE_PPS):
        return 1
    step = int(round(float(WAVEFORM_COLUMN_STEP_REFERENCE_PPS) / pps))
    return max(1, min(int(WAVEFORM_COLUMN_STEP_MAX_PX), step))


def _compact_peak_span_numpy(
    *,
    peaks: np.ndarray,
    start: int,
    end: int,
    seconds_per_peak: float,
    pixels_per_second: float,
    scroll_x: float,
    content_start_x: float,
    pixel_step_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized compaction: collapse many peaks into min/max by screen x bucket."""
    span = peaks[start : end + 1]
    if span.size == 0:
        empty_i = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float32)
        return empty_i, empty_f, empty_f

    idx = np.arange(start, end + 1, dtype=np.float64)
    x = (content_start_x + (idx * seconds_per_peak * pixels_per_second) - scroll_x).astype(np.int32)
    if pixel_step_px > 1:
        x = (x // pixel_step_px) * pixel_step_px

    if x.shape[0] == 1:
        return (
            x.astype(np.int32, copy=False),
            span[:, 0].astype(np.float32, copy=False),
            span[:, 1].astype(np.float32, copy=False),
        )

    boundaries = np.flatnonzero(np.diff(x)) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))

    mins = np.minimum.reduceat(span[:, 0], starts).astype(np.float32, copy=False)
    maxs = np.maximum.reduceat(span[:, 1], starts).astype(np.float32, copy=False)
    x_values = x[starts].astype(np.int32, copy=False)
    return x_values, mins, maxs


def visible_waveform_columns(
    presentation: WaveformLanePresentation,
    step: int,
) -> list[tuple[float, int]]:
    """Return screen-space waveform columns while preserving scroll continuity."""
    # Keep waveform columns on the same horizontal time origin as ruler/events.
    content_left = float(presentation.header_width)
    content_right = max(content_left + 1.0, float(presentation.width))
    visible_span = max(1.0, content_right - content_left)
    timeline_start_px = max(0.0, presentation.scroll_x)

    start_index = int(floor(timeline_start_px / step)) - 1
    end_index = int(floor((timeline_start_px + visible_span) / step)) + 2

    cols: list[tuple[float, int]] = []
    for idx in range(start_index, end_index + 1):
        x = content_left + ((idx * step) - timeline_start_px)
        if (content_left - step) <= x <= (content_right + step):
            cols.append((x, idx))
    return cols
