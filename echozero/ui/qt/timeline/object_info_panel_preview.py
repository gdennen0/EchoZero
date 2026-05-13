"""Object info panel preview helpers.
Exists to keep typed inspector preview parsing and rendering out of the panel shell.
Connects inspector preview payloads to compact Qt renderer surfaces by preview kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget

from echozero.application.presentation.inspector_contract import InspectorAction
_AUDIO_EVENT_CLIP_PREVIEW_KIND = "audio_event_clip"
_AUDIO_EVENT_PREVIEW_VARIANT_BARS = "bars"
_AUDIO_EVENT_PREVIEW_VARIANT_FILLED = "filled"
_AUDIO_EVENT_PREVIEW_VARIANT_OUTLINE = "outline"
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import (
    CachedWaveform,
    get_cached_waveform,
    register_waveform_from_audio_file,
)


@dataclass(slots=True)
class AudioEventPreviewState:
    layer_id: object
    take_id: object | None
    event_id: object
    source_ref: str
    source_audio_path: str | None
    waveform_key: str | None
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    kind: str = _AUDIO_EVENT_CLIP_PREVIEW_KIND
    variant: str | None = None


class EventPreviewWaveform(QFrame):
    """Compact waveform strip for the currently selected event preview clip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timeline_object_info_event_preview_waveform")
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._preview: AudioEventPreviewState | None = None
        self._variant = _AUDIO_EVENT_PREVIEW_VARIANT_BARS

    def sizeHint(self) -> QSize:
        return QSize(240, 60)

    def set_preview(self, preview: AudioEventPreviewState | None) -> None:
        self._preview = preview
        self.update()

    def set_variant(self, variant: str) -> None:
        if variant not in audio_event_preview_variants():
            return
        self._variant = variant
        self.update()

    def paintEvent(self, _event: object) -> None:
        rect = self.rect().adjusted(6, 6, -6, -6)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        panel_style = TIMELINE_STYLE.object_palette
        background_painter = QPainter(self)
        background_painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        background_painter.setPen(Qt.PenStyle.NoPen)
        background_painter.setBrush(QColor(panel_style.button_bg_hex))
        background_painter.drawRoundedRect(rect, 8.0, 8.0)
        background_painter.end()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)

        preview = self._preview
        if preview is None or preview.duration_seconds <= 0.0:
            painter.end()
            return

        cached = self._resolve_cached_waveform(preview)
        if cached is None or cached.peaks.size == 0:
            self._paint_placeholder(painter, rect)
            painter.end()
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
            painter.end()
            return

        center_y = rect.center().y()
        amp_px = rect.height() * 0.40
        accent = QColor(TIMELINE_STYLE.fixture.layer_color_tokens["event_preview"])
        painter.setPen(QPen(accent, 1.0))
        for column_index, (vmin, vmax) in enumerate(peak_columns):
            x = rect.left() + column_index
            top_y = int(round(center_y - (float(vmax) * amp_px)))
            bottom_y = int(round(center_y - (float(vmin) * amp_px)))
            _paint_audio_event_preview_column(
                painter,
                x=x,
                top_y=top_y,
                bottom_y=bottom_y,
                accent=accent,
                variant=self._variant,
            )

        marker_pen = QPen(QColor(panel_style.border_hex), 1.0)
        marker_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(marker_pen)
        painter.drawRect(rect)
        painter.end()

    @staticmethod
    def _paint_placeholder(painter: QPainter, rect: QRect) -> None:
        accent = QColor(TIMELINE_STYLE.fixture.layer_color_tokens["event_preview"])
        accent.setAlpha(140)
        painter.setPen(QPen(accent, 1.0))
        mid_y = rect.center().y()
        painter.drawLine(rect.left(), mid_y, rect.right(), mid_y)

    def _resolve_cached_waveform(
        self,
        preview: AudioEventPreviewState,
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


def audio_event_preview_variants() -> tuple[str, ...]:
    """Return the supported user-facing audio preview display variants."""

    return (
        _AUDIO_EVENT_PREVIEW_VARIANT_BARS,
        _AUDIO_EVENT_PREVIEW_VARIANT_FILLED,
        _AUDIO_EVENT_PREVIEW_VARIANT_OUTLINE,
    )


def audio_event_preview_variant_label(variant: str) -> str:
    """Return the button label for one audio preview display variant."""

    labels = {
        _AUDIO_EVENT_PREVIEW_VARIANT_BARS: "Bars",
        _AUDIO_EVENT_PREVIEW_VARIANT_FILLED: "Fill",
        _AUDIO_EVENT_PREVIEW_VARIANT_OUTLINE: "Outline",
    }
    return labels.get(variant, "Bars")


def _paint_audio_event_preview_column(
    painter: QPainter,
    *,
    x: int,
    top_y: int,
    bottom_y: int,
    accent: QColor,
    variant: str,
) -> None:
    upper_y = min(top_y, bottom_y)
    lower_y = max(top_y, bottom_y)
    if variant == _AUDIO_EVENT_PREVIEW_VARIANT_FILLED:
        fill_color = QColor(accent)
        fill_color.setAlpha(120)
        painter.fillRect(x, upper_y, 1, max(1, lower_y - upper_y + 1), fill_color)
        painter.fillRect(x, upper_y, 1, 1, accent)
        painter.fillRect(x, lower_y, 1, 1, accent)
        return
    if variant == _AUDIO_EVENT_PREVIEW_VARIANT_OUTLINE:
        painter.fillRect(x, upper_y, 1, 1, accent)
        painter.fillRect(x, lower_y, 1, 1, accent)
        return
    painter.fillRect(x, upper_y, 1, max(1, lower_y - upper_y + 1), accent)


def build_waveform_envelope_points(
    rect: QRect,
    *,
    peak_columns: list[tuple[float, float]],
) -> tuple[list[QPoint], list[QPoint]]:
    """Convert cached clip peaks into top/bottom envelope points for outline painting."""

    center_y = rect.center().y()
    amp_px = rect.height() * 0.40
    top_envelope: list[QPoint] = []
    bottom_envelope: list[QPoint] = []
    for column_index, (vmin, vmax) in enumerate(peak_columns):
        x = rect.left() + column_index
        top_envelope.append(
            QPoint(int(x), int(round(center_y - (float(vmax) * amp_px))))
        )
        bottom_envelope.append(
            QPoint(int(x), int(round(center_y - (float(vmin) * amp_px))))
        )
    return top_envelope, bottom_envelope


def preview_state_from_action(action: InspectorAction | None) -> AudioEventPreviewState | None:
    if action is None:
        return None
    preview_payload = action.params.get("preview")
    if isinstance(preview_payload, dict):
        preview_kind = str(preview_payload.get("kind", "")).strip()
        parser = _PREVIEW_PAYLOAD_PARSERS.get(preview_kind)
        if parser is not None:
            return parser(action, preview_payload)
    return _legacy_audio_event_preview_state(action)


def _audio_event_preview_state(
    action: InspectorAction,
    preview_payload: dict[str, object],
) -> AudioEventPreviewState | None:
    source_ref = str(preview_payload.get("source_ref", "")).strip()
    start_seconds = _coerce_param_float(preview_payload.get("start_seconds"))
    end_seconds = _coerce_param_float(preview_payload.get("end_seconds"))
    if start_seconds is None or end_seconds is None:
        return None
    if not source_ref or end_seconds <= start_seconds:
        return None
    source_audio_path = preview_payload.get("source_audio_path")
    waveform_key = preview_payload.get("waveform_key")
    duration_seconds = _coerce_param_float(preview_payload.get("duration_seconds"))
    return AudioEventPreviewState(
        kind=_AUDIO_EVENT_CLIP_PREVIEW_KIND,
        layer_id=action.params.get("layer_id"),
        take_id=action.params.get("take_id"),
        event_id=action.params.get("event_id"),
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


def _legacy_audio_event_preview_state(action: InspectorAction) -> AudioEventPreviewState | None:
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
    return AudioEventPreviewState(
        kind=_AUDIO_EVENT_CLIP_PREVIEW_KIND,
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


def preview_meta_text(preview: AudioEventPreviewState) -> str:
    formatter = _PREVIEW_META_FORMATTERS.get(preview.kind)
    if formatter is None:
        return ""
    return formatter(preview)


def _audio_event_preview_meta_text(preview: AudioEventPreviewState) -> str:
    source_label = Path(preview.source_audio_path or preview.source_ref).name
    return (
        f"{preview.duration_seconds:.2f}s clip · "
        f"{preview.start_seconds:.2f}s to {preview.end_seconds:.2f}s\n"
        f"Source: {source_label}"
    )


_PREVIEW_PAYLOAD_PARSERS: dict[
    str,
    Callable[[InspectorAction, dict[str, object]], AudioEventPreviewState | None],
] = {
    _AUDIO_EVENT_CLIP_PREVIEW_KIND: _audio_event_preview_state,
}

_PREVIEW_META_FORMATTERS: dict[str, Callable[[AudioEventPreviewState], str]] = {
    _AUDIO_EVENT_CLIP_PREVIEW_KIND: _audio_event_preview_meta_text,
}

# Compatibility aliases while adjacent inspector surfaces migrate to typed preview names.
EventPreviewState = AudioEventPreviewState
event_preview_from_action = preview_state_from_action
event_preview_meta_text = preview_meta_text
