from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from PyQt6.QtGui import QColor, QPainter, QPen

from echozero.ui.qt.timeline.waveform_cache import CachedWaveform, get_cached_waveform


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


class WaveformLaneBlock:
    def paint(self, painter: QPainter, top: int, presentation: WaveformLanePresentation) -> None:
        base = QColor(presentation.color_hex)
        if presentation.dimmed:
            base.setAlpha(120)

        cached = get_cached_waveform(presentation.waveform_key)
        if cached is not None:
            self._paint_cached_waveform(painter, top, presentation, cached, base)
            return

        # Fallback placeholder when no real waveform data is registered.
        mid_y = top + presentation.row_height / 2
        pps = max(1.0, presentation.pixels_per_second)
        step = max(2, int(180 / pps * 10))
        amp_scale = min(1.0, max(0.35, pps / 220.0))
        painter.setPen(QPen(base, 1.2))
        for x, sample_index in visible_waveform_columns(presentation, step):
            amp = (((sample_index * 37) % 11) + 1) / 11.0
            h = amp * (presentation.row_height * 0.30) * amp_scale
            painter.drawLine(int(x), int(mid_y - h), int(x), int(mid_y + h))

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
        amp_px = presentation.row_height * 0.38

        painter.setPen(QPen(color, 1.0))
        for idx in range(start_idx, end_idx + 1):
            t = idx * spp
            x = waveform_x_for_time(
                t,
                scroll_x=presentation.scroll_x,
                pixels_per_second=pps,
                content_start_x=content_left,
            )
            if x < (content_left - 1) or x > (content_right + 1):
                continue
            vmin, vmax = cached.peaks[idx]
            y1 = center_y - (float(vmax) * amp_px)
            y2 = center_y - (float(vmin) * amp_px)
            painter.drawLine(int(x), int(y1), int(x), int(y2))


def waveform_x_for_time(
    time_seconds: float,
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_start_x: float,
) -> float:
    pps = max(1.0, pixels_per_second)
    return content_start_x + (max(0.0, time_seconds) * pps) - scroll_x


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
