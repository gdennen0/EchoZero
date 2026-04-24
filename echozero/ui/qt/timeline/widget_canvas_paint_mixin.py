"""Timeline canvas paint helpers.
Exists to keep timeline drawing and tooltip formatting out of the canvas root.
Connects presentation rows and FEEL-backed geometry to the canvas render path.
"""

from __future__ import annotations

from typing import Any, cast

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPaintEvent, QPainter, QPen

from echozero.application.presentation.models import LayerPresentation, TakeLanePresentation
from echozero.application.shared.ids import LayerId, TakeId
from echozero.perf import timed
from echozero.ui.FEEL import (
    EVENT_SELECTION_COLOR,
    GRID_BAR_LINE_ALPHA,
    GRID_BEAT_LINE_ALPHA,
    GRID_LINE_ALPHA,
    GRID_LINE_COLOR,
)
from echozero.ui.qt.timeline.blocks.event_lane import EventLanePresentation
from echozero.ui.qt.timeline.blocks.layouts import MainRowLayout, TakeRowLayout
from echozero.ui.qt.timeline.blocks.ruler import playhead_head_polygon, timeline_x_for_time
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLanePresentation
from echozero.ui.qt.timeline.time_grid import visible_grid_lines
from echozero.ui.qt.timeline.widget_canvas_types import EventRect, TakeActionRect, TakeRect


def badge_tooltip_labels(badges: list[str]) -> list[str]:
    mapping = {
        "main": "Main take",
        "stem": "Stem output",
        "audio": "Audio lane",
        "event": "Event lane",
        "classifier-preview": "Classifier preview",
        "real-data": "Real data",
    }
    labels: list[str] = []
    for badge in badges:
        key = str(badge).strip().lower()
        if not key:
            continue
        labels.append(mapping.get(key, key.replace("-", " ").title()))
    return labels


class _TimelineCanvasPaintMixin:
    def paintEvent(self: Any, event: QPaintEvent | None) -> None:
        if event is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self._style.canvas.background_hex))
        self._take_rects.clear()
        self._take_option_rects.clear()
        self._take_action_rects.clear()
        self._toggle_rects.clear()
        self._active_rects.clear()
        self._pipeline_action_rects.clear()
        self._push_rects.clear()
        self._pull_rects.clear()
        self._event_rects.clear()
        self._event_lane_rects.clear()
        self._header_select_rects.clear()
        self._row_body_select_rects.clear()
        self._header_hover_rects.clear()
        self._event_drop_rects.clear()
        with timed("timeline.paint.layers"):
            self._draw_layers(painter)
        with timed("timeline.paint.playhead"):
            self._draw_playhead(painter)
        self._draw_interaction_overlays(painter)

    def _draw_time_grid_band(self: Any, painter: QPainter, *, top: int, row_height: int) -> None:
        content_left = float(self._header_width)
        content_width = max(1.0, float(self.width()) - content_left)
        lines = visible_grid_lines(
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_width=content_width,
            mode=self._grid_mode,
            bpm=self.presentation.bpm,
        )
        if not lines:
            return

        band_top = int(top)
        band_bottom = int(top + max(1, row_height) - 1)
        for line in lines:
            x = timeline_x_for_time(
                line.time_seconds,
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=content_left,
            )
            if x < content_left:
                continue
            alpha = GRID_LINE_ALPHA
            if line.role == "beat":
                alpha = GRID_BEAT_LINE_ALPHA
            elif line.role in {"bar", "major"}:
                alpha = GRID_BAR_LINE_ALPHA
            grid_color = QColor(GRID_LINE_COLOR)
            grid_color.setAlpha(max(0, min(255, alpha)))
            painter.setPen(QPen(grid_color, 1))
            painter.drawLine(int(x), band_top, int(x), band_bottom)

    def _draw_region_overlay_band(self: Any, painter: QPainter, *, top: int, row_height: int) -> None:
        if not self.presentation.regions:
            return
        content_left = float(self._header_width)
        content_right = float(self.width())
        if content_right <= content_left:
            return
        for index, region in enumerate(self.presentation.regions):
            start_x = timeline_x_for_time(
                region.start,
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=content_left,
            )
            end_x = timeline_x_for_time(
                region.end,
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=content_left,
            )
            left = max(content_left, min(start_x, end_x))
            right = min(content_right, max(start_x, end_x))
            width = max(0.0, right - left)
            if width <= 0.0:
                continue
            fill_hex = (
                region.color
                or (
                    self._style.canvas.region_even_hex
                    if index % 2 == 0
                    else self._style.canvas.region_odd_hex
                )
            )
            fill_color = QColor(fill_hex)
            fill_color.setAlpha(max(0, min(255, int(self._style.canvas.region_alpha))))
            rect = QRectF(left, float(top), width, float(max(1, row_height) - 1))
            painter.fillRect(rect, fill_color)
            if region.is_selected:
                painter.save()
                painter.setPen(QPen(QColor(self._style.canvas.region_selected_outline_hex), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect.adjusted(1.0, 1.0, -1.0, -1.0))
                painter.restore()

    def _draw_layers(self: Any, painter: QPainter) -> None:
        y = self._top_padding
        for layer in self.presentation.layers:
            self._draw_main_row(painter, layer, y)
            y += self._main_row_height
            if layer.is_expanded:
                for take in layer.takes:
                    self._draw_take_row(painter, layer, take, y)
                    y += self._take_row_height

    @staticmethod
    def _header_tooltip_text(layer: LayerPresentation) -> str:
        labels = badge_tooltip_labels(layer.badges)
        parts: list[str] = []
        if labels:
            parts.append(" | ".join(labels))
        if layer.status.stale:
            stale_text = "Status: Stale"
            stale_reason = getattr(layer.status, "stale_reason", "")
            if stale_reason:
                stale_text = f"{stale_text} ({stale_reason})"
            parts.append(stale_text)
        if layer.status.manually_modified:
            parts.append("Status: Manually modified")
        if layer.status.source_label:
            parts.append(layer.status.source_label)
        source_layer_id = getattr(layer.status, "source_layer_id", "")
        if source_layer_id:
            parts.append(f"Source layer: {source_layer_id}")
        source_song_version_id = getattr(layer.status, "source_song_version_id", "")
        if source_song_version_id:
            parts.append(f"Source song version: {source_song_version_id}")
        pipeline_id = getattr(layer.status, "pipeline_id", "")
        if pipeline_id:
            parts.append(f"Pipeline: {pipeline_id}")
        output_name = getattr(layer.status, "output_name", "")
        if output_name:
            parts.append(f"Output: {output_name}")
        source_run_id = getattr(layer.status, "source_run_id", "")
        if source_run_id:
            parts.append(f"Run: {source_run_id}")
        if layer.status.sync_label and layer.status.sync_label.lower() != "no sync":
            parts.append(f"Sync: {layer.status.sync_label}")
        return "\n".join(parts)

    def _draw_main_row(self: Any, painter: QPainter, layer: LayerPresentation, top: int) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = MainRowLayout.create(
            top=top,
            width=self.width(),
            header_width=self._header_width,
            row_height=self._main_row_height,
        )
        row_bg = QColor(
            self._style.canvas.selected_row_fill_hex
            if layer.is_selected
            else self._style.canvas.row_fill_hex
        )
        if dimmed:
            row_bg = QColor(self._style.canvas.dimmed_row_fill_hex)
        painter.fillRect(layout.row_rect, row_bg)
        if self._push_outline_active_for_layer(layer):
            outline_rect = layout.row_rect.adjusted(1.0, 1.0, -1.0, -1.0)
            painter.save()
            painter.setPen(QPen(QColor("#8fd0ff"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(outline_rect, 8.0, 8.0)
            painter.restore()
        self._draw_region_overlay_band(painter, top=top, row_height=self._main_row_height)
        self._draw_time_grid_band(painter, top=top, row_height=self._main_row_height)
        painter.fillRect(
            0,
            top + self._main_row_height - 1,
            self.width(),
            1,
            QColor(self._style.canvas.row_divider_hex),
        )

        slots = self._header_block_slots_factory(layout)
        if layer.takes:
            self._toggle_rects.append((slots.toggle_rect, layer.layer_id))
        self._header_select_rects.append((layout.header_rect, layer.layer_id))
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        self._header_hover_rects.append((layout.header_rect, layer))
        if layer.kind.name == "EVENT":
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
            self._event_lane_rects.append((layout.content_rect, layer.layer_id, layer.main_take_id))
        hit_targets = self._header_block.paint(painter, slots, layer, dimmed=dimmed)
        for control_id, rect in hit_targets.control_rects:
            if control_id == "set_active_playback_target":
                self._active_rects.append((rect, layer.layer_id))
            elif control_id == "layer_pipeline_actions":
                self._pipeline_action_rects.append((rect, layer.layer_id))
            elif control_id in {"push_to_ma3", "send_to_ma3"}:
                self._push_rects.append((rect, layer.layer_id))
            elif control_id == "pull_from_ma3":
                self._pull_rects.append((rect, layer.layer_id))

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if layer.kind.name == "AUDIO":
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or self._style.fixture.fallback_audio_lane_hex,
                        row_height=self._main_row_height,
                        pixels_per_second=self.presentation.pixels_per_second,
                        scroll_x=self.presentation.scroll_x,
                        header_width=self._header_width,
                        width=self.width(),
                        dimmed=dimmed,
                        waveform_key=layer.waveform_key,
                    ),
                )
            else:
                self._event_rects.extend(
                    cast(
                        list[EventRect],
                        self._event_lane_block.paint(
                            painter,
                            top + 24,
                            EventLanePresentation(
                                layer_id=layer.layer_id,
                                take_id=layer.main_take_id,
                                events=layer.events,
                                default_fill_hex=layer.color,
                                pixels_per_second=self.presentation.pixels_per_second,
                                scroll_x=self.presentation.scroll_x,
                                header_width=self._header_width,
                                event_height=self._event_height,
                                dimmed=dimmed,
                                viewport_width=self.width(),
                            ),
                        ),
                    )
                )

            if not layer.takes:
                hint_color = (
                    self._style.canvas.no_takes_hint_dimmed_hex
                    if dimmed
                    else self._style.canvas.no_takes_hint_hex
                )
                painter.setPen(QColor(hint_color))
                painter.drawText(
                    layout.content_rect.adjusted(10, 0, -10, 0),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    "No takes yet",
                )
        finally:
            painter.restore()

    def _is_take_options_open(self: Any, layer_id: LayerId, take_id: TakeId) -> bool:
        return (layer_id, take_id) in self._open_take_options

    def _draw_take_row(
        self: Any,
        painter: QPainter,
        layer: LayerPresentation,
        take: TakeLanePresentation,
        top: int,
    ) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = TakeRowLayout.create(
            top=top,
            width=self.width(),
            header_width=self._header_width,
            row_height=self._take_row_height,
        )
        options_open = self._is_take_options_open(layer.layer_id, take.take_id)
        hit_targets = self._take_row_block.paint_header(
            painter,
            layout,
            layer,
            take,
            options_open=options_open,
            dimmed=dimmed,
        )
        self._draw_region_overlay_band(painter, top=top, row_height=self._take_row_height)
        self._draw_time_grid_band(painter, top=top, row_height=self._take_row_height)
        self._take_rects.append(cast(TakeRect, hit_targets.take_rect))
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        if take.kind.name == "EVENT":
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
            self._event_lane_rects.append((layout.content_rect, layer.layer_id, take.take_id))
        if hit_targets.options_toggle_rect is not None:
            self._take_option_rects.append(cast(TakeRect, hit_targets.options_toggle_rect))
        self._take_action_rects.extend(cast(list[TakeActionRect], hit_targets.action_rects))

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if take.kind.name == "AUDIO":
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or self._style.fixture.fallback_audio_lane_hex,
                        row_height=self._take_row_height,
                        pixels_per_second=self.presentation.pixels_per_second,
                        scroll_x=self.presentation.scroll_x,
                        header_width=self._header_width,
                        width=self.width(),
                        dimmed=True or dimmed,
                        waveform_key=take.waveform_key,
                    ),
                )
            else:
                self._event_rects.extend(
                    cast(
                        list[EventRect],
                        self._event_lane_block.paint(
                            painter,
                            top + 10,
                            EventLanePresentation(
                                layer_id=layer.layer_id,
                                take_id=take.take_id,
                                events=take.events,
                                default_fill_hex=layer.color,
                                pixels_per_second=self.presentation.pixels_per_second,
                                scroll_x=self.presentation.scroll_x,
                                header_width=self._header_width,
                                event_height=self._event_height,
                                dimmed=True or dimmed,
                                viewport_width=self.width(),
                            ),
                        ),
                    )
                )
        finally:
            painter.restore()

    def _draw_playhead(self: Any, painter: QPainter) -> None:
        x = timeline_x_for_time(
            self.presentation.playhead,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        if x < self._header_width or x > self.width():
            return

        painter.setPen(
            QPen(QColor(self._style.playhead.color_hex), self._style.playhead.line_width_px)
        )
        painter.drawLine(int(x), 0, int(x), self.height())
        painter.setBrush(QColor(self._style.playhead.color_hex))
        painter.setPen(
            QPen(
                QColor(self._style.playhead.color_hex),
                self._style.playhead.head_outline_width_px,
            )
        )
        painter.drawPolygon(playhead_head_polygon(x, float(self._top_padding)))

    def _draw_interaction_overlays(self: Any, painter: QPainter) -> None:
        if self._snap_indicator_time is not None:
            x = timeline_x_for_time(
                self._snap_indicator_time,
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=self._header_width,
            )
            if self._header_width <= x <= self.width():
                snap_color = QColor(EVENT_SELECTION_COLOR)
                snap_color.setAlpha(110)
                painter.save()
                painter.setPen(QPen(snap_color, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(x), int(self._top_padding), int(x), self.height())
                painter.restore()

        if self._preview_event_rect is not None:
            preview_color = QColor(EVENT_SELECTION_COLOR)
            preview_color.setAlpha(52)
            painter.save()
            painter.setPen(QPen(QColor(EVENT_SELECTION_COLOR), 1, Qt.PenStyle.DashLine))
            painter.setBrush(preview_color)
            painter.drawRoundedRect(self._preview_event_rect, 6.0, 6.0)
            painter.restore()

        if self._marquee_rect is not None:
            marquee_color = QColor(EVENT_SELECTION_COLOR)
            marquee_fill = QColor(EVENT_SELECTION_COLOR)
            marquee_fill.setAlpha(36)
            painter.save()
            painter.setPen(QPen(marquee_color, 1, Qt.PenStyle.DashLine))
            painter.setBrush(marquee_fill)
            painter.drawRect(self._marquee_rect.normalized())
            painter.restore()
