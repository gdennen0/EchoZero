"""Object info panel event-preview helpers.
Exists to keep waveform preview state and rendering out of the panel shell.
Connects inspector preview actions to compact cached-waveform rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget

from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import (
    CachedWaveform,
    get_cached_waveform,
    register_waveform_from_audio_file,
)


@dataclass(slots=True)
class EventPreviewState:
    layer_id: object
    take_id: object | None
    event_id: object
    source_ref: str
    source_audio_path: str | None
    waveform_key: str | None
    start_seconds: float
    end_seconds: float
    duration_seconds: float


class EventPreviewWaveform(QFrame):
    """Compact waveform strip for the currently selected event preview clip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timeline_object_info_event_preview_waveform")
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._preview: EventPreviewState | None = None

    def sizeHint(self) -> QSize:
        return QSize(240, 60)

    def set_preview(self, preview: EventPreviewState | None) -> None:
        self._preview = preview
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        rect = self.rect().adjusted(6, 6, -6, -6)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        panel_style = TIMELINE_STYLE.object_palette
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(panel_style.button_bg_hex))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        preview = self._preview
        if preview is None or preview.duration_seconds <= 0.0:
            return

        cached = self._resolve_cached_waveform(preview)
        if cached is None or cached.peaks.size == 0:
            self._paint_placeholder(painter, rect)
            return

        start_seconds = max(0.0, float(preview.start_seconds))
        end_seconds = max(start_seconds, float(preview.end_seconds))
        peak_columns = clip_waveform_columns(
            cached,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            column_count=max(1, rect.width()),
        )
        if not peak_columns:
            self._paint_placeholder(painter, rect)
            return

        center_y = rect.center().y()
        amp_px = rect.height() * 0.40
        accent = QColor(TIMELINE_STYLE.fixture.layer_color_tokens["event_preview"])
        painter.setPen(QPen(accent, 1.0))
        for column_index, (vmin, vmax) in enumerate(peak_columns):
            x = rect.left() + column_index
            y1 = center_y - (float(vmax) * amp_px)
            y2 = center_y - (float(vmin) * amp_px)
            painter.drawLine(int(x), int(round(y1)), int(x), int(round(y2)))

        marker_pen = QPen(QColor(panel_style.border_hex), 1.0)
        marker_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(marker_pen)
        painter.drawRect(rect)

    @staticmethod
    def _paint_placeholder(painter: QPainter, rect: QRect) -> None:
        accent = QColor(TIMELINE_STYLE.fixture.layer_color_tokens["event_preview"])
        accent.setAlpha(140)
        painter.setPen(QPen(accent, 1.0))
        mid_y = rect.center().y()
        painter.drawLine(rect.left(), mid_y, rect.right(), mid_y)

    def _resolve_cached_waveform(
        self,
        preview: EventPreviewState,
    ) -> CachedWaveform | None:
        if preview.waveform_key:
            cached = get_cached_waveform(preview.waveform_key)
            if cached is not None:
                return cached

        source_path = preview.source_audio_path or preview.source_ref
        if not source_path:
            return None
        candidate = Path(str(source_path))
        if not candidate.exists():
            return None
        waveform_key = preview.waveform_key or f"object-info:{candidate.resolve()}"
        cached = get_cached_waveform(waveform_key)
        if cached is None:
            try:
                cached = register_waveform_from_audio_file(waveform_key, candidate)
            except Exception:
                return None
        preview.waveform_key = waveform_key
        if preview.source_audio_path is None:
            preview.source_audio_path = str(candidate)
        return cached


def clip_waveform_columns(
    cached: CachedWaveform,
    *,
    start_seconds: float,
    end_seconds: float,
    column_count: int,
) -> list[tuple[float, float]]:
    """Resample one clip span to screen columns for readable inspector previews."""

    if column_count <= 0 or cached.peaks.size == 0:
        return []

    start = max(0.0, float(start_seconds))
    end = max(start, float(end_seconds))
    span = end - start
    if span <= 0.0:
        return []

    seconds_per_peak = cached.seconds_per_peak
    peak_count = int(cached.peaks.shape[0])
    if seconds_per_peak <= 0.0 or peak_count <= 0:
        return []

    columns: list[tuple[float, float]] = []
    for column_index in range(column_count):
        column_start = start + (span * (column_index / column_count))
        column_end = start + (span * ((column_index + 1) / column_count))
        start_idx = max(0, int(floor(column_start / seconds_per_peak)))
        end_idx = min(
            peak_count - 1,
            max(start_idx, int(ceil(column_end / seconds_per_peak)) - 1),
        )
        segment = cached.peaks[start_idx : end_idx + 1]
        if segment.size == 0:
            columns.append((0.0, 0.0))
            continue
        columns.append(
            (
                float(segment[:, 0].min()),
                float(segment[:, 1].max()),
            )
        )
    return columns


def event_preview_from_action(action: InspectorAction | None) -> EventPreviewState | None:
    if action is None:
        return None
    params = action.params
    source_ref = str(params.get("source_ref", "")).strip()
    start_seconds = _coerce_param_float(params.get("start_seconds"))
    end_seconds = _coerce_param_float(params.get("end_seconds"))
    if start_seconds is None or end_seconds is None:
        return None
    if not source_ref or end_seconds <= start_seconds:
        return None
    source_audio_path = params.get("source_audio_path")
    waveform_key = params.get("waveform_key")
    duration_seconds = _coerce_param_float(params.get("duration_seconds"))
    return EventPreviewState(
        layer_id=params.get("layer_id"),
        take_id=params.get("take_id"),
        event_id=params.get("event_id"),
        source_ref=source_ref,
        source_audio_path=(
            str(source_audio_path).strip() if source_audio_path not in (None, "") else None
        ),
        waveform_key=str(waveform_key).strip() if waveform_key not in (None, "") else None,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=(
            duration_seconds if duration_seconds is not None else end_seconds - start_seconds
        ),
    )


def _coerce_param_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def event_preview_meta_text(preview: EventPreviewState) -> str:
    source_label = Path(preview.source_audio_path or preview.source_ref).name
    return (
        f"{preview.duration_seconds:.2f}s clip · "
        f"{preview.start_seconds:.2f}s to {preview.end_seconds:.2f}s\n"
        f"Source: {source_label}"
    )
