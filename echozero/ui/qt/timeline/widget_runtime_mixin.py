"""Timeline widget runtime and viewport behavior.
Exists to keep follow-scroll, zoom, transport state, and runtime-audio updates out of the shell constructor.
Connects presentation updates and runtime timing to the reusable timeline controls and canvas.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol, cast

from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QFrame, QLabel, QScrollArea, QScrollBar, QWidget

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.intents import Seek, Stop, TimelineIntent
from echozero.ui.FEEL import (
    TIMELINE_ZOOM_MAX_PPS,
    TIMELINE_ZOOM_MIN_PPS,
    TIMELINE_ZOOM_STEP_FACTOR,
)
from echozero.ui.qt.timeline.blocks.ruler import seek_time_for_x
from echozero.ui.qt.timeline.runtime_audio import (
    RuntimeAudioTimingSnapshot,
    TimelineRuntimeAudioController,
)
from echozero.ui.qt.timeline.time_grid import TimelineGridMode
from echozero.ui.qt.timeline.widget_canvas import TimelineCanvas
from echozero.ui.qt.timeline.widget_controls import (
    TimelineEditorModeBar,
    TimelineRuler,
    TransportBar,
)
from echozero.ui.qt.timeline.widget_viewport import (
    compute_follow_scroll_x,
    compute_scroll_bounds,
)


class _RuntimeShellWithPipelineUpdate(Protocol):
    def consume_pipeline_run_presentation_update(self) -> TimelinePresentation | None: ...


class _TimelineWidgetRuntimeHost(Protocol):
    presentation: TimelinePresentation
    _on_intent: Callable[[TimelineIntent], TimelinePresentation | None] | None
    _runtime_audio: TimelineRuntimeAudioController | None
    _runtime_source_signature: tuple[tuple[str, str], ...] | None
    _runtime_playhead_floor: float | None
    _runtime_timing_snapshot: RuntimeAudioTimingSnapshot | None
    _edit_mode: str
    _snap_enabled: bool
    _grid_mode: str
    _scroll: QScrollArea
    _hscroll: QScrollBar
    _canvas: TimelineCanvas
    _ruler: TimelineRuler
    _transport: TransportBar
    _editor_bar: TimelineEditorModeBar
    _pipeline_status: QFrame
    _pipeline_status_label: QLabel

    def width(self) -> int: ...
    def _refresh_object_info_panel(self) -> None: ...
    def _resolve_runtime_shell(self) -> object | None: ...
    def set_presentation(self, presentation: TimelinePresentation) -> None: ...
    def _update_horizontal_scroll_bounds(self, *, sync_bar_value: bool) -> None: ...
    def _reset_scroll_area_horizontal_offset(self) -> None: ...
    def _sync_editor_state(self) -> None: ...
    def _sync_pipeline_status_banner(self) -> None: ...
    def _set_pipeline_status_tone(self, tone: str) -> None: ...
    def _set_snap_enabled(self, enabled: bool) -> None: ...
    def _set_grid_mode(self, mode: str) -> None: ...
    def _apply_zoom_factor(self, factor: float, *, anchor_x: float) -> None: ...
    def _sample_runtime_playhead(self) -> tuple[float, bool]: ...
    def _stabilize_runtime_playhead(self, runtime_time: float, *, playing: bool) -> float: ...
    def _current_runtime_timing_snapshot(self) -> RuntimeAudioTimingSnapshot | None: ...
    def _resolve_runtime_time(self, snapshot: RuntimeAudioTimingSnapshot | None) -> float: ...


class TimelineWidgetRuntimeMixin:
    _runtime_source_signature: tuple[tuple[str, str], ...] | None
    _runtime_playhead_floor: float | None
    _runtime_timing_snapshot: RuntimeAudioTimingSnapshot | None

    def apply_external_presentation_update(
        self: _TimelineWidgetRuntimeHost,
        presentation: TimelinePresentation,
    ) -> None:
        """Apply a runtime-driven refresh while preserving local viewport state when compatible."""

        self.set_presentation(self._with_local_viewport(presentation))

    def set_presentation(
        self: _TimelineWidgetRuntimeHost,
        presentation: TimelinePresentation,
    ) -> None:
        viewport_widget = self._scroll.viewport()
        viewport = max(1, viewport_widget.width() if viewport_widget is not None else self.width())
        followed = compute_follow_scroll_x(
            presentation,
            viewport,
            header_width=self._canvas._header_width,
        )
        self.presentation = replace(presentation, scroll_x=followed)
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._reset_scroll_area_horizontal_offset()
        self._transport.set_presentation(self.presentation)
        song_browser = getattr(self, "_song_browser_panel", None)
        if song_browser is not None and callable(getattr(song_browser, "set_presentation", None)):
            song_browser.set_presentation(self.presentation)
        self._sync_editor_state()
        self._sync_pipeline_status_banner()
        self._refresh_object_info_panel()
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation)
        if self._runtime_audio is not None:
            if hasattr(self._runtime_audio, "presentation_signature"):
                runtime_signature = self._runtime_audio.presentation_signature(self.presentation)
            else:
                runtime_signature = tuple(
                    (str(layer.layer_id), layer.source_audio_path or "")
                    for layer in self.presentation.layers
                    if layer.source_audio_path
                )
            if runtime_signature != self._runtime_source_signature:
                self._runtime_audio.build_for_presentation(self.presentation)
                self._runtime_source_signature = runtime_signature
            else:
                self._runtime_audio.apply_mix_state(self.presentation)

    def _with_local_viewport(
        self: _TimelineWidgetRuntimeHost,
        presentation: TimelinePresentation,
    ) -> TimelinePresentation:
        current = self.presentation
        if current.timeline_id != presentation.timeline_id:
            return presentation
        return replace(
            presentation,
            scroll_x=current.scroll_x,
            scroll_y=current.scroll_y,
            pixels_per_second=current.pixels_per_second,
        )

    def set_runtime_audio_controller(
        self: _TimelineWidgetRuntimeHost,
        runtime_audio: TimelineRuntimeAudioController | None,
    ) -> None:
        """Swap the runtime-audio controller and resync it from the current presentation."""

        self._runtime_audio = runtime_audio
        self._runtime_source_signature = None
        self._runtime_playhead_floor = None
        self._runtime_timing_snapshot = None
        self.set_presentation(self.presentation)

    def resizeEvent(
        self: _TimelineWidgetRuntimeHost,
        event: QResizeEvent | None,
    ) -> None:
        QWidget.resizeEvent(cast(QWidget, self), event)
        self._update_horizontal_scroll_bounds(sync_bar_value=False)

    def _update_horizontal_scroll_bounds(
        self: _TimelineWidgetRuntimeHost,
        *,
        sync_bar_value: bool,
    ) -> None:
        viewport_widget = self._scroll.viewport()
        viewport = max(1, viewport_widget.width() if viewport_widget is not None else self.width())
        _, max_scroll = compute_scroll_bounds(self.presentation, viewport)

        current = int(round(self.presentation.scroll_x))
        clamped = max(0, min(current, max_scroll))

        self._hscroll.blockSignals(True)
        self._hscroll.setRange(0, max_scroll)
        self._hscroll.setPageStep(viewport)
        if sync_bar_value or self._hscroll.value() != clamped:
            self._hscroll.setValue(clamped)
        self._hscroll.blockSignals(False)

        if clamped != current:
            self.presentation = replace(self.presentation, scroll_x=float(clamped))

    def _on_horizontal_scroll_changed(
        self: _TimelineWidgetRuntimeHost,
        value: int,
    ) -> None:
        next_scroll = float(max(0, value))
        if abs(next_scroll - self.presentation.scroll_x) < 0.5:
            return
        self.presentation = replace(self.presentation, scroll_x=next_scroll)
        self._reset_scroll_area_horizontal_offset()
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)
        self._sync_pipeline_status_banner()

    def _scroll_horizontally_by_steps(
        self: _TimelineWidgetRuntimeHost,
        delta: float,
    ) -> None:
        if delta == 0:
            return
        if abs(delta) >= 120.0:
            notches = max(-6, min(6, int(delta / 120)))
            scroll_delta = float(notches * self._hscroll.singleStep())
        else:
            scroll_delta = delta
        next_value = int(round(self._hscroll.value() + scroll_delta))
        self._hscroll.setValue(
            max(self._hscroll.minimum(), min(self._hscroll.maximum(), next_value))
        )

    def _reset_scroll_area_horizontal_offset(self: _TimelineWidgetRuntimeHost) -> None:
        bar = self._scroll.horizontalScrollBar()
        if bar is None or bar.value() == 0:
            return
        bar.blockSignals(True)
        bar.setValue(0)
        bar.blockSignals(False)

    def _sync_editor_state(self: _TimelineWidgetRuntimeHost) -> None:
        beat_available = self.presentation.bpm is not None and float(self.presentation.bpm) > 0.0
        if self._grid_mode == TimelineGridMode.BEAT.value and not beat_available:
            self._grid_mode = TimelineGridMode.AUTO.value
        self._editor_bar.set_state(
            edit_mode=self._edit_mode,
            snap_enabled=self._snap_enabled,
            grid_mode=self._grid_mode,
            beat_available=beat_available,
        )
        self._canvas.set_editor_state(
            edit_mode=self._edit_mode,
            snap_enabled=self._snap_enabled,
            grid_mode=self._grid_mode,
        )
        self._ruler.set_editor_mode(self._edit_mode)

    def _sync_pipeline_status_banner(self: _TimelineWidgetRuntimeHost) -> None:
        banner = self.presentation.pipeline_run_banner
        if banner is None:
            self._pipeline_status_label.setText("")
            self._set_pipeline_status_tone("")
            self._pipeline_status.setVisible(False)
            return
        detail = (banner.message or banner.status.replace("_", " ").title()).strip()
        percent_text = ""
        if banner.percent is not None and not banner.is_error:
            percent_text = f" ({int(round(float(banner.percent) * 100.0))}%)"
        prefix = "Pipeline Failed" if banner.is_error else "Pipeline Running"
        self._pipeline_status_label.setText(
            f"{prefix}: {banner.title} · {detail}{percent_text}"
        )
        self._set_pipeline_status_tone("error" if banner.is_error else "running")
        self._pipeline_status.setVisible(True)

    def _set_pipeline_status_tone(self: _TimelineWidgetRuntimeHost, tone: str) -> None:
        current_tone = str(self._pipeline_status.property("tone") or "")
        if current_tone == tone:
            return
        self._pipeline_status.setProperty("tone", tone)
        style = self._pipeline_status.style()
        if style is not None:
            style.unpolish(self._pipeline_status)
            style.polish(self._pipeline_status)
        self._pipeline_status.update()

    def _set_edit_mode(self: _TimelineWidgetRuntimeHost, mode: str) -> None:
        normalized = (mode or "select").strip().lower()
        if normalized not in {"select", "draw", "erase", "move", "region"}:
            return
        self._edit_mode = normalized
        self._sync_editor_state()

    def _set_snap_enabled(self: _TimelineWidgetRuntimeHost, enabled: bool) -> None:
        self._snap_enabled = bool(enabled)
        self._sync_editor_state()

    def _toggle_snap_enabled(self: _TimelineWidgetRuntimeHost) -> None:
        self._set_snap_enabled(not self._snap_enabled)

    def _set_grid_mode(self: _TimelineWidgetRuntimeHost, mode: str) -> None:
        normalized = (mode or TimelineGridMode.AUTO.value).strip().lower()
        try:
            resolved = TimelineGridMode(normalized)
        except ValueError:
            resolved = TimelineGridMode.AUTO
        if resolved is TimelineGridMode.BEAT and not (
            self.presentation.bpm and float(self.presentation.bpm) > 0.0
        ):
            resolved = TimelineGridMode.AUTO
        self._grid_mode = resolved.value
        self._sync_editor_state()

    def _cycle_grid_mode(self: _TimelineWidgetRuntimeHost) -> None:
        beat_available = self.presentation.bpm is not None and float(self.presentation.bpm) > 0.0
        modes = (
            [TimelineGridMode.AUTO, TimelineGridMode.BEAT, TimelineGridMode.OFF]
            if beat_available
            else [TimelineGridMode.AUTO, TimelineGridMode.OFF]
        )
        current = TimelineGridMode(self._grid_mode)
        try:
            index = modes.index(current)
        except ValueError:
            index = 0
        self._set_grid_mode(modes[(index + 1) % len(modes)].value)

    def _zoom_from_input(
        self: _TimelineWidgetRuntimeHost,
        delta: int,
        anchor_x: float,
    ) -> None:
        if delta == 0:
            return
        factor = TIMELINE_ZOOM_STEP_FACTOR if delta > 0 else (1.0 / TIMELINE_ZOOM_STEP_FACTOR)
        self._apply_zoom_factor(factor, anchor_x=anchor_x)

    def _apply_zoom_factor(
        self: _TimelineWidgetRuntimeHost,
        factor: float,
        *,
        anchor_x: float,
    ) -> None:
        current_pps = max(1.0, float(self.presentation.pixels_per_second))
        target_pps = max(
            TIMELINE_ZOOM_MIN_PPS,
            min(TIMELINE_ZOOM_MAX_PPS, current_pps * float(factor)),
        )
        if abs(target_pps - current_pps) < 0.001:
            return

        content_start_x = float(self._canvas._header_width)
        anchor_view_x = max(content_start_x, float(anchor_x))
        anchor_time = seek_time_for_x(
            anchor_view_x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=current_pps,
            content_start_x=content_start_x,
        )
        new_scroll = max(0.0, (anchor_time * target_pps) - (anchor_view_x - content_start_x))

        self.presentation = replace(
            self.presentation,
            pixels_per_second=target_pps,
            scroll_x=new_scroll,
        )
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._transport.set_presentation(self.presentation)
        self._refresh_object_info_panel()
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)
        self._sync_pipeline_status_banner()

    def _dispatch(self: _TimelineWidgetRuntimeHost, intent: TimelineIntent) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is None:
            return

        updated = self._with_local_viewport(updated)
        if self._runtime_audio is not None:
            runtime_time, runtime_playing = self._sample_runtime_playhead()
            if isinstance(intent, Seek):
                runtime_time = max(0.0, float(intent.position))
                self._runtime_timing_snapshot = None
                self._runtime_playhead_floor = runtime_time if runtime_playing else None
            elif isinstance(intent, Stop):
                runtime_time = 0.0
                self._runtime_timing_snapshot = None
                self._runtime_playhead_floor = None
            else:
                runtime_time = self._stabilize_runtime_playhead(
                    runtime_time,
                    playing=runtime_playing,
                )
            updated = replace(
                updated,
                playhead=runtime_time,
                is_playing=runtime_playing,
                current_time_label=_format_time_label(runtime_time),
            )
        self.set_presentation(updated)

    def _on_runtime_tick(self: _TimelineWidgetRuntimeHost) -> None:
        runtime = self._resolve_runtime_shell()
        consume_pipeline_update = (
            cast(_RuntimeShellWithPipelineUpdate, runtime).consume_pipeline_run_presentation_update
            if runtime is not None
            and callable(getattr(runtime, "consume_pipeline_run_presentation_update", None))
            else None
        )
        if callable(consume_pipeline_update):
            updated = consume_pipeline_update()
            if updated is not None:
                updated = replace(
                    self._with_local_viewport(updated),
                    playhead=self.presentation.playhead,
                    is_playing=self.presentation.is_playing,
                    current_time_label=self.presentation.current_time_label,
                )
                self.set_presentation(updated)

        if self._runtime_audio is None:
            return

        current_time, playing = self._sample_runtime_playhead()
        current_time = self._stabilize_runtime_playhead(current_time, playing=playing)
        current_label = _format_time_label(current_time)
        if (
            abs(current_time - self.presentation.playhead) < 0.001
            and playing == self.presentation.is_playing
            and current_label == self.presentation.current_time_label
        ):
            return

        next_presentation = replace(
            self.presentation,
            playhead=current_time,
            is_playing=playing,
            current_time_label=current_label,
        )
        viewport_widget = self._scroll.viewport()
        followed = compute_follow_scroll_x(
            next_presentation,
            max(1, viewport_widget.width() if viewport_widget is not None else self.width()),
            header_width=self._canvas._header_width,
        )
        self.presentation = replace(next_presentation, scroll_x=followed)
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._transport.set_presentation(self.presentation)
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)
        self._sync_pipeline_status_banner()

    def _sample_runtime_playhead(
        self: _TimelineWidgetRuntimeHost,
    ) -> tuple[float, bool]:
        runtime_audio = self._runtime_audio
        if runtime_audio is None:
            return 0.0, False
        snapshot = self._current_runtime_timing_snapshot()
        playing = snapshot.is_playing if snapshot is not None else runtime_audio.is_playing()
        current_time = self._resolve_runtime_time(snapshot)
        if snapshot is None:
            current_time = max(0.0, float(runtime_audio.current_time_seconds()))
        return current_time, playing

    def _current_runtime_timing_snapshot(
        self: _TimelineWidgetRuntimeHost,
    ) -> RuntimeAudioTimingSnapshot | None:
        if self._runtime_audio is None or not hasattr(self._runtime_audio, "timing_snapshot"):
            self._runtime_timing_snapshot = None
            return None
        snapshot = self._runtime_audio.timing_snapshot()
        if not isinstance(snapshot, RuntimeAudioTimingSnapshot):
            self._runtime_timing_snapshot = None
            return None
        self._runtime_timing_snapshot = snapshot
        return snapshot

    @staticmethod
    def _resolve_runtime_time(snapshot: RuntimeAudioTimingSnapshot | None) -> float:
        if snapshot is None:
            return 0.0
        base_time = max(0.0, float(snapshot.audible_time_seconds))
        if not snapshot.is_playing or snapshot.snapshot_monotonic_seconds is None:
            return base_time
        elapsed = max(0.0, time.monotonic() - float(snapshot.snapshot_monotonic_seconds))
        return max(0.0, min(float(snapshot.clock_time_seconds), base_time + elapsed))

    def _stabilize_runtime_playhead(
        self: _TimelineWidgetRuntimeHost,
        runtime_time: float,
        *,
        playing: bool,
    ) -> float:
        next_time = max(0.0, float(runtime_time))
        if not playing:
            self._runtime_playhead_floor = None
            return next_time
        if self._runtime_playhead_floor is None:
            self._runtime_playhead_floor = next_time
            return next_time
        if next_time + 1e-6 < self._runtime_playhead_floor:
            return self._runtime_playhead_floor
        self._runtime_playhead_floor = next_time
        return next_time


def _format_time_label(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


__all__ = ["TimelineWidgetRuntimeMixin"]
