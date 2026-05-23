"""Timeline canvas paint helpers.
Exists to keep timeline drawing and tooltip formatting out of the canvas root.
Connects presentation rows and FEEL-backed geometry to the canvas render path.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, cast

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPaintEvent, QPainter, QPen

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
)
from echozero.application.shared.cue_numbers import cue_number_from_ref_text, cue_number_text
from echozero.application.shared.enums import LayerKind, PlaybackMode
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import LayerId, TakeId
from echozero.application.timeline.video_placement import VideoPlacement
from echozero.perf import timed
from echozero.ui.FEEL import (
    EVENT_MIN_HIT_WIDTH_PX,
    EVENT_MIN_VISIBLE_WIDTH_PX,
    EVENT_SELECTION_BORDER_PX,
    EVENT_SELECTION_COLOR,
    EVENT_SELECTION_OUTLINE_EXPAND_PX,
    EVENT_SELECTION_TINY_WIDTH_EXTRA_PX,
    EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX,
    GRID_BAR_LINE_ALPHA,
    GRID_BEAT_LINE_ALPHA,
    GRID_LINE_ALPHA,
    GRID_LINE_COLOR,
    MOVE_DRAG_PREVIEW_LINE_ALPHA,
    MOVE_DRAG_PREVIEW_LINE_WIDTH_PX,
    NOTE_CONTOUR_ALPHA,
    NOTE_CONTOUR_PEN_WIDTH_PX,
    SECTION_MOVE_EVENT_HIT_MIN_WIDTH_PX,
)
from echozero.ui.qt.timeline.blocks.event_lane import EventLanePresentation
from echozero.ui.qt.timeline.blocks.layouts import MainRowLayout, TakeRowLayout
from echozero.ui.qt.timeline.blocks.ruler import timeline_x_for_time
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLanePresentation
from echozero.ui.qt.timeline.note_contour_overlay import (
    build_note_contour_path,
    contour_samples_from_events,
)
from echozero.ui.qt.timeline.time_grid import GridLine, visible_grid_lines
from echozero.ui.qt.timeline.widget_canvas_types import EventRect, TakeActionRect, TakeRect

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FixCandidateLane:
    layer: LayerPresentation
    take_id: TakeId | None
    events: list[EventPresentation]


def badge_tooltip_labels(badges: list[str]) -> list[str]:
    mapping = {
        "main": "Main take",
        "stem": "Stem output",
        "audio": "Audio lane",
        "event": "Event lane",
        "section": "Sections lane",
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


def section_region_label_text(region: Any) -> str:
    raw_cue_ref = str(getattr(region, "cue_ref", "") or "").strip()
    cue_ref = raw_cue_ref
    if cue_ref.casefold() == "none":
        cue_ref = ""
    elif cue_ref:
        cue_ref = cue_number_text(cue_number_from_ref_text(cue_ref)) or cue_ref
    name = str(getattr(region, "name", "") or "").strip()
    if raw_cue_ref and name.casefold() == raw_cue_ref.casefold():
        return cue_ref
    if cue_ref and name and name.casefold() != cue_ref.casefold():
        return f"{cue_ref} {name}"
    return cue_ref or name


class _TimelineCanvasPaintMixin:
    def paintEvent(self: Any, event: QPaintEvent | None) -> None:
        if event is None:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.fillRect(self.rect(), QColor(self._style.canvas.background_hex))
            content_left = float(self._header_width)
            content_width = max(1.0, float(self.width()) - content_left)
            self._frame_visible_grid_lines: list[GridLine] = visible_grid_lines(
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_width=content_width,
                mode=self._grid_mode,
                bpm=self.presentation.bpm,
                beat_anchor_seconds=self.presentation.beat_anchor_seconds,
            )
            self._take_rects.clear()
            self._take_option_rects.clear()
            self._take_action_rects.clear()
            self._toggle_rects.clear()
            self._mute_rects.clear()
            self._solo_rects.clear()
            self._pipeline_action_rects.clear()
            self._push_rects.clear()
            self._pull_rects.clear()
            self._section_manager_rects.clear()
            self._event_rects.clear()
            self._section_label_rects.clear()
            self._section_boundary_rects.clear()
            self._section_marker_rects.clear()
            self._fix_event_rects.clear()
            self._event_lane_rects.clear()
            self._header_select_rects.clear()
            self._row_body_select_rects.clear()
            self._header_hover_rects.clear()
            self._event_drop_rects.clear()
            self._video_clip_rects.clear()
            self._layer_row_resize_hit_rects.clear()
            with timed("timeline.paint.layers"):
                self._draw_layers(painter)
            self._draw_header_content_divider(painter)
            self._draw_playback_start_marker(painter)
            with timed("timeline.paint.playhead"):
                self._draw_playhead(painter)
            self._draw_interaction_overlays(painter)
            self._frame_visible_grid_lines = []
        except Exception:
            logger.exception("Timeline canvas paint failed")
        finally:
            if painter.isActive():
                painter.end()

    def _draw_time_grid_band(self: Any, painter: QPainter, *, top: int, row_height: int) -> None:
        content_left = float(self._header_width)
        lines = list(getattr(self, "_frame_visible_grid_lines", []))
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

    def _draw_section_backdrop_band(
        self: Any, painter: QPainter, *, top: int, row_height: int
    ) -> None:
        if not self.presentation.section_regions:
            return
        content_left = float(self._header_width)
        content_right = float(self.width())
        if content_right <= content_left:
            return
        for index, region in enumerate(self.presentation.section_regions):
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
            fill_hex = region.color or (
                self._style.canvas.section_even_hex
                if index % 2 == 0
                else self._style.canvas.section_odd_hex
            )
            fill_color = QColor(fill_hex)
            fill_color.setAlpha(max(0, min(255, int(self._style.canvas.section_alpha))))
            rect = QRectF(left, float(top), width, float(max(1, row_height) - 1))
            painter.fillRect(rect, fill_color)

    def _draw_section_overlay_band(
        self: Any, painter: QPainter, *, top: int, row_height: int
    ) -> None:
        if not self.presentation.section_regions:
            return
        content_left = float(self._header_width)
        content_right = float(self.width())
        if content_right <= content_left:
            return
        for index, region in enumerate(self.presentation.section_regions):
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
            fill_hex = region.color or (
                self._style.canvas.section_even_hex
                if index % 2 == 0
                else self._style.canvas.section_odd_hex
            )
            fill_color = QColor(fill_hex)
            fill_color.setAlpha(max(0, min(255, int(self._style.canvas.section_alpha))))
            rect = QRectF(left, float(top), width, float(max(1, row_height) - 1))
            painter.fillRect(rect, fill_color)
            painter.save()
            painter.setPen(QPen(QColor(self._style.canvas.section_boundary_hex), 2))
            painter.drawLine(
                int(round(left)),
                int(round(top)),
                int(round(left)),
                int(round(top + max(1, row_height) - 1)),
            )
            painter.drawLine(
                int(round(right)),
                int(round(top)),
                int(round(right)),
                int(round(top + max(1, row_height) - 1)),
            )
            painter.restore()
            label_text = section_region_label_text(region)
            boundary_rect = QRectF(
                left - 6.0,
                float(top),
                12.0,
                float(max(1, row_height)),
            )
            marker_rect = boundary_rect
            if label_text:
                metrics = painter.fontMetrics()
                label_width = min(
                    max(36.0, float(metrics.horizontalAdvance(label_text)) + 16.0),
                    max(36.0, width - 10.0),
                )
                label_rect = QRectF(
                    left + 6.0,
                    float(top),
                    label_width,
                    float(max(1, row_height) - 1),
                )
                painter.setPen(QColor(self._style.canvas.section_boundary_hex))
                painter.drawText(
                    label_rect,
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    label_text,
                )
                self._section_label_rects.append((label_rect, region.cue_id))
                marker_rect = marker_rect.united(label_rect)
            self._section_boundary_rects.append(
                (
                    boundary_rect,
                    region.cue_id,
                )
            )
            self._section_marker_rects.append(
                (
                    marker_rect,
                    region.cue_id,
                )
            )

    @staticmethod
    def _shows_section_overlay_for_layer(layer: LayerPresentation) -> bool:
        return layer.kind is LayerKind.SECTION

    def _draw_layers(self: Any, painter: QPainter) -> None:
        y = self._top_padding
        for row in self._layer_rows():
            layer = row.layer
            row_height = self._main_row_height_for_layer(layer)
            self._draw_main_row(
                painter,
                layer,
                y,
                row_height=row_height,
                hierarchy_depth=row.depth,
                has_child_layers=row.has_child_layers,
            )
            y += row_height
            if layer.is_expanded and not layer.is_fully_collapsed:
                for take in layer.takes:
                    self._draw_take_row(painter, layer, take, y)
                    y += self._take_row_height

    @staticmethod
    def _header_tooltip_text(layer: LayerPresentation) -> str:
        labels = badge_tooltip_labels(layer.badges)
        parts: list[str] = []
        if layer.takes:
            parts.append("Toggle takes: click chevron | Full collapse: Shift-click chevron")
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

    def _draw_main_row(
        self: Any,
        painter: QPainter,
        layer: LayerPresentation,
        top: int,
        *,
        row_height: int,
        hierarchy_depth: int = 0,
        has_child_layers: bool = False,
    ) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = MainRowLayout.create(
            top=top,
            width=self.width(),
            header_width=self._header_width,
            row_height=row_height,
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
            painter.setPen(QPen(QColor("#8f8a84"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(outline_rect, 3.0, 3.0)
            painter.restore()
        if hierarchy_depth > 0:
            hierarchy_accent = QColor("#93A0B1")
            hierarchy_accent.setAlpha(84)
            indent_px = min(34.0, 14.0 + (float(hierarchy_depth - 1) * 10.0))
            painter.fillRect(
                QRectF(0.0, float(top), indent_px, float(max(1, row_height) - 1)),
                hierarchy_accent,
            )
            painter.save()
            painter.setPen(QPen(QColor("#8f8a84"), 1))
            branch_x = indent_px + 5.0
            branch_top = float(top + 8)
            branch_mid = float(top + max(12, row_height // 2))
            painter.drawLine(int(branch_x), int(branch_top), int(branch_x), int(branch_mid))
            painter.drawLine(int(branch_x), int(branch_mid), int(branch_x + 8), int(branch_mid))
            painter.restore()
        if self._shows_section_overlay_for_layer(layer) and not layer.is_fully_collapsed:
            self._draw_section_overlay_band(painter, top=top, row_height=row_height)
        if layer.kind is not LayerKind.SECTION and not layer.is_fully_collapsed:
            self._draw_section_backdrop_band(painter, top=top, row_height=row_height)
        if not layer.is_fully_collapsed:
            self._draw_time_grid_band(painter, top=top, row_height=row_height)
        painter.fillRect(
            0,
            top + row_height - 1,
            self.width(),
            1,
            QColor(self._style.canvas.row_divider_hex),
        )
        if not layer.is_fully_collapsed:
            resize_hit_padding = max(2, int(self._resize_handle_hit_padding))
            self._layer_row_resize_hit_rects.append(
                (
                    QRectF(
                        0.0,
                        float(top + row_height - resize_hit_padding),
                        float(self.width()),
                        float(resize_hit_padding * 2),
                    ),
                    layer.layer_id,
                )
            )

        slots = self._header_block_slots_factory(layout)
        if hierarchy_depth > 0:
            indent_px = min(44.0, 18.0 + (float(hierarchy_depth - 1) * 10.0))
            slots.title_rect.adjust(indent_px, 0.0, 0.0, 0.0)
            slots.status_rect.adjust(indent_px, 0.0, 0.0, 0.0)
            slots.metadata_rect.adjust(indent_px, 0.0, 0.0, 0.0)
        if layer.takes or layer.is_fully_collapsed or has_child_layers:
            self._toggle_rects.append((slots.toggle_rect, layer.layer_id))
        self._header_select_rects.append((layout.header_rect, layer.layer_id))
        if not layer.is_fully_collapsed:
            self._row_body_select_rects.append((layout.content_rect, layer.layer_id, None))
        self._header_hover_rects.append((layout.header_rect, layer))
        if is_event_like_layer_kind(layer.kind) and not layer.is_fully_collapsed:
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
            self._event_lane_rects.append(
                (layout.content_rect, layer.layer_id, layer.main_take_id)
            )
        hit_targets = self._header_block.paint(
            painter,
            slots,
            layer,
            dimmed=dimmed,
            has_child_layers=has_child_layers,
        )
        for control_id, rect in hit_targets.control_rects:
            if control_id == "set_layer_mute":
                self._mute_rects.append((rect, layer.layer_id))
            elif control_id == "set_layer_solo":
                self._solo_rects.append((rect, layer.layer_id))
            elif control_id == "layer_pipeline_actions":
                self._pipeline_action_rects.append((rect, layer.layer_id))
            elif control_id in {"push_to_ma3", "send_to_ma3"}:
                self._push_rects.append((rect, layer.layer_id))
            elif control_id == "pull_from_ma3":
                self._pull_rects.append((rect, layer.layer_id))
            elif control_id == "open_section_layer_manager":
                self._section_manager_rects.append((rect, layer.layer_id))
        if layer.is_fully_collapsed:
            return

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if layer.kind.name == "AUDIO":
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or self._style.fixture.fallback_audio_lane_hex,
                        row_height=row_height,
                        pixels_per_second=self.presentation.pixels_per_second,
                        scroll_x=self.presentation.scroll_x,
                        header_width=self._header_width,
                        width=self.width(),
                        dimmed=dimmed,
                        waveform_key=layer.waveform_key,
                        source_audio_path=layer.source_audio_path,
                        unavailable_reason="Waveform unavailable",
                        repaint_target=self,
                    ),
                )
                self._draw_note_contour_overlay(
                    painter,
                    layer=layer,
                    top=float(top),
                    row_height=float(row_height),
                    dimmed=dimmed,
                )
            elif layer.kind is LayerKind.REFERENCE and layer.reference_kind == "video":
                self._draw_video_reference_lane(
                    painter,
                    layer=layer,
                    top=top,
                    row_height=row_height,
                    dimmed=dimmed,
                )
            else:
                if layer.kind is not LayerKind.SECTION:
                    visible_events = self._visible_lane_events(layer.events)
                    event_lane_top = float(top + max(0.0, (row_height - self._event_height) * 0.5))
                    self._draw_fix_overlay_events(
                        painter,
                        layer=layer,
                        take_id=layer.main_take_id,
                        lane_events=visible_events,
                        top=event_lane_top,
                    )
                    self._event_rects.extend(
                        cast(
                            list[EventRect],
                            self._event_lane_block.paint(
                                painter,
                                int(round(event_lane_top)),
                                EventLanePresentation(
                                    layer_id=layer.layer_id,
                                    take_id=layer.main_take_id,
                                    events=visible_events,
                                    layer_kind=layer.kind,
                                    event_hit_min_width_px=(
                                        float(SECTION_MOVE_EVENT_HIT_MIN_WIDTH_PX)
                                        if self._edit_mode in {"move", "select"}
                                        and layer.kind is LayerKind.SECTION
                                        else float(EVENT_MIN_HIT_WIDTH_PX)
                                    ),
                                    default_fill_hex=layer.color,
                                    waveform_key=layer.waveform_key
                                    or (
                                        f"event-audio:{layer.playback_source_ref}"
                                        if layer.playback_source_ref
                                        else None
                                    ),
                                    source_audio_path=layer.source_audio_path
                                    or layer.playback_source_ref,
                                    render_audio_shape=bool(
                                        self._edit_mode != "fix"
                                        and
                                        layer.playback_enabled
                                        and layer.playback_mode is PlaybackMode.EVENT_SLICE
                                    ),
                                    pixels_per_second=self.presentation.pixels_per_second,
                                    scroll_x=self.presentation.scroll_x,
                                    header_width=self._header_width,
                                    event_height=self._event_height,
                                    dimmed=dimmed,
                                    viewport_width=self.width(),
                                    repaint_target=self,
                                ),
                            ),
                        )
                    )

        finally:
            painter.restore()

    def _draw_video_reference_lane(
        self: Any,
        painter: QPainter,
        *,
        layer: LayerPresentation,
        top: int,
        row_height: int,
        dimmed: bool,
    ) -> None:
        start = float(layer.video_start_seconds)
        trim_start = max(0.0, float(layer.video_trim_start_seconds))
        source_duration = max(0.0, float(layer.video_duration_seconds))
        placement = VideoPlacement(
            start_seconds=start,
            trim_start_seconds=trim_start,
            visible_duration_seconds=float(layer.video_visible_duration_seconds),
            source_duration_seconds=source_duration,
            loop_enabled=bool(layer.video_loop_enabled),
        ).normalized()
        start = placement.start_seconds
        trim_start = placement.trim_start_seconds
        duration = placement.visible_duration_seconds
        pps = max(1.0, float(self.presentation.pixels_per_second))
        content_left = float(self._header_width)
        content_right = float(max(self._header_width + 1, self.width()))
        clip_x = content_left + (start * pps) - float(self.presentation.scroll_x)
        clip_w = max(2.0, duration * pps)
        clip_rect = QRectF(clip_x, float(top + 4), clip_w, float(max(1, row_height - 8)))
        visible_rect = clip_rect.intersected(
            QRectF(content_left, float(top), content_right - content_left, float(row_height))
        )
        fill = QColor(layer.color or self._style.fixture.fallback_audio_lane_hex)
        fill.setAlpha(58 if not dimmed else 34)
        outline = QColor(layer.color or self._style.fixture.fallback_audio_lane_hex)
        outline.setAlpha(190 if not dimmed else 120)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(outline, 1))
        painter.setBrush(QBrush(fill))
        if not visible_rect.isEmpty():
            painter.drawRoundedRect(visible_rect, 3.0, 3.0)
        self._waveform_block.paint(
            painter,
            top,
            WaveformLanePresentation(
                color_hex=layer.color or self._style.fixture.fallback_audio_lane_hex,
                row_height=row_height,
                pixels_per_second=self.presentation.pixels_per_second,
                scroll_x=self.presentation.scroll_x,
                header_width=self._header_width,
                width=self.width(),
                dimmed=dimmed,
                waveform_key=layer.waveform_key,
                source_audio_path=layer.source_audio_path,
                unavailable_reason="Video audio unavailable",
                repaint_target=self,
                time_offset_seconds=start - trim_start,
            ),
        )
        if not visible_rect.isEmpty():
            self._draw_video_loop_affordances(
                painter,
                rect=visible_rect,
                placement=placement,
                color=outline,
                dimmed=dimmed,
            )
        painter.restore()
        self._video_clip_rects.append(
            (
                visible_rect if not visible_rect.isEmpty() else clip_rect,
                layer.layer_id,
                start,
                trim_start,
                duration,
                source_duration,
                bool(layer.video_loop_enabled),
            )
        )

    def _draw_video_loop_affordances(
        self: Any,
        painter: QPainter,
        *,
        rect: QRectF,
        placement: VideoPlacement,
        color: QColor,
        dimmed: bool,
    ) -> None:
        if rect.width() < 18.0 or rect.height() < 14.0:
            return
        icon_color = QColor(color)
        icon_color.setAlpha(170 if placement.loop_enabled and not dimmed else 96)
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(icon_color, 1))
        self._draw_video_loop_icon(painter, rect.left() + 5.0, rect.top() + 4.0)
        self._draw_video_loop_icon(painter, rect.right() - 15.0, rect.top() + 4.0)
        if placement.loop_enabled:
            self._draw_video_loop_repeats(painter, rect=rect, placement=placement)
        painter.restore()

    def _draw_video_loop_icon(self: Any, painter: QPainter, x: float, y: float) -> None:
        icon_rect = QRectF(float(x), float(y), 10.0, 8.0)
        painter.drawArc(icon_rect, 40 * 16, 285 * 16)
        painter.drawLine(int(x + 8), int(y + 1), int(x + 10), int(y + 1))
        painter.drawLine(int(x + 8), int(y + 1), int(x + 9), int(y + 3))

    def _draw_video_loop_repeats(
        self: Any,
        painter: QPainter,
        *,
        rect: QRectF,
        placement: VideoPlacement,
    ) -> None:
        cycle_seconds = placement.loop_cycle_seconds
        if cycle_seconds <= 0.0 or placement.visible_duration_seconds <= cycle_seconds:
            return
        pps = max(1.0, float(self.presentation.pixels_per_second))
        seam_color = QColor(EVENT_SELECTION_COLOR)
        seam_color.setAlpha(95)
        painter.setPen(QPen(seam_color, 1, Qt.PenStyle.DashLine))
        seam_offset = cycle_seconds * pps
        x = float(rect.left()) + seam_offset
        while x < float(rect.right()) - 2.0:
            painter.drawLine(
                int(round(x)),
                int(rect.top() + 3),
                int(round(x)),
                int(rect.bottom() - 3),
            )
            x += seam_offset

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
        if self._shows_section_overlay_for_layer(layer):
            self._draw_section_overlay_band(painter, top=top, row_height=self._take_row_height)
        if layer.kind is not LayerKind.SECTION:
            self._draw_section_backdrop_band(painter, top=top, row_height=self._take_row_height)
        self._draw_time_grid_band(painter, top=top, row_height=self._take_row_height)
        self._take_rects.append(cast(TakeRect, hit_targets.take_rect))
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id, take.take_id))
        if is_event_like_layer_kind(take.kind):
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
                        source_audio_path=take.source_audio_path,
                        unavailable_reason="Waveform unavailable",
                        repaint_target=self,
                    ),
                )
            else:
                if take.kind is not LayerKind.SECTION:
                    visible_events = self._visible_lane_events(take.events)
                    event_lane_top = float(
                        top
                        + max(
                            0.0,
                            (self._take_row_height - self._event_height) * 0.5,
                        )
                    )
                    self._draw_fix_overlay_events(
                        painter,
                        layer=layer,
                        take_id=take.take_id,
                        lane_events=visible_events,
                        top=event_lane_top,
                    )
                    self._event_rects.extend(
                        cast(
                            list[EventRect],
                            self._event_lane_block.paint(
                                painter,
                                int(round(event_lane_top)),
                                EventLanePresentation(
                                    layer_id=layer.layer_id,
                                    take_id=take.take_id,
                                    events=visible_events,
                                    layer_kind=take.kind,
                                    event_hit_min_width_px=(
                                        float(SECTION_MOVE_EVENT_HIT_MIN_WIDTH_PX)
                                        if self._edit_mode in {"move", "select"}
                                        and take.kind is LayerKind.SECTION
                                        else float(EVENT_MIN_HIT_WIDTH_PX)
                                    ),
                                    default_fill_hex=layer.color,
                                    waveform_key=take.waveform_key
                                    or layer.waveform_key
                                    or (
                                        f"event-audio:{take.playback_source_ref or layer.playback_source_ref}"
                                        if (take.playback_source_ref or layer.playback_source_ref)
                                        else None
                                    ),
                                    source_audio_path=take.source_audio_path
                                    or take.playback_source_ref
                                    or layer.source_audio_path
                                    or layer.playback_source_ref,
                                    render_audio_shape=bool(
                                        self._edit_mode != "fix"
                                        and
                                        layer.playback_enabled
                                        and layer.playback_mode is PlaybackMode.EVENT_SLICE
                                    ),
                                    pixels_per_second=self.presentation.pixels_per_second,
                                    scroll_x=self.presentation.scroll_x,
                                    header_width=self._header_width,
                                    event_height=self._event_height,
                                    dimmed=True or dimmed,
                                    viewport_width=self.width(),
                                    repaint_target=self,
                                ),
                            ),
                        )
                    )
        finally:
            painter.restore()

    def _draw_fix_overlay_events(
        self: Any,
        painter: QPainter,
        *,
        layer: LayerPresentation,
        take_id: TakeId | None,
        lane_events: list[EventPresentation],
        top: float,
    ) -> None:
        if self._edit_mode != "fix":
            return
        source_events = self._resolve_fix_overlay_source_events(
            layer=layer,
            take_id=take_id,
            lane_events=lane_events,
        )
        if not source_events:
            return

        pps = max(1.0, float(self.presentation.pixels_per_second))
        content_left = float(self._header_width)
        content_right = float(max(self._header_width + 1, self.width()))
        visible_start_t = max(0.0, float(self.presentation.scroll_x) / pps)
        visible_end_t = max(
            visible_start_t,
            (float(self.presentation.scroll_x) + max(1.0, content_right - content_left)) / pps,
        )
        matched_source_ids = self._fix_overlay_matched_source_ids(lane_events=lane_events)
        include_unmatched = str(self._fix_action or "").strip().lower() == "promote"

        for source_event in source_events:
            start = float(source_event.start)
            end = float(source_event.end)
            if end < visible_start_t:
                continue
            if start > visible_end_t:
                break

            x = self._header_width + (start * pps) - float(self.presentation.scroll_x)
            width = max(float(EVENT_MIN_VISIBLE_WIDTH_PX), (max(0.0, end - start) * pps))
            if x + width < content_left - 2.0 or x > content_right + 2.0:
                continue

            source_event_id = str(source_event.event_id)
            matched = source_event_id in matched_source_ids
            if not matched and not include_unmatched:
                continue
            rect = QRectF(float(x), top, float(width), float(self._event_height))
            self._fix_event_rects.append(
                (
                    rect,
                    layer.layer_id,
                    take_id,
                    source_event_id,
                    start,
                    max(start + 0.01, end),
                    matched,
                )
            )

            fill = QColor("#93A0B1")
            fill.setAlpha(68 if matched else 112)
            border = QColor("#c0bab4")
            border.setAlpha(96 if matched else 176)
            painter.setPen(
                QPen(
                    border,
                    1,
                    Qt.PenStyle.DashLine if matched else Qt.PenStyle.SolidLine,
                )
            )
            painter.setBrush(QBrush(fill))
            painter.drawRoundedRect(rect, 3.0, 3.0)

    def _resolve_fix_overlay_source_events(
        self: Any,
        *,
        layer: LayerPresentation,
        take_id: TakeId | None,
        lane_events: list[EventPresentation],
    ) -> list[EventPresentation]:
        source_lane = self._resolve_fix_overlay_source_lane(
            layer=layer,
            take_id=take_id,
            lane_events=lane_events,
        )
        if source_lane is None:
            return []
        return sorted(
            source_lane.events,
            key=lambda event: (float(event.start), float(event.end)),
        )

    def _resolve_fix_overlay_source_lane(
        self: Any,
        *,
        layer: LayerPresentation,
        take_id: TakeId | None,
        lane_events: list[EventPresentation],
    ) -> _FixCandidateLane | None:
        candidates = self._fix_candidate_lanes()
        if not candidates:
            return None

        target_is_onset = self._is_onset_layer(layer)
        lane_source_ids = {str(event.source_event_id or event.event_id) for event in lane_events}
        layer_source_id = str(layer.status.source_layer_id or "").strip()
        best_score = float("-inf")
        best_lane: _FixCandidateLane | None = None

        for candidate in candidates:
            if candidate.layer.layer_id == layer.layer_id and candidate.take_id == take_id:
                continue
            if not candidate.events:
                continue
            candidate_is_onset = self._is_onset_layer(candidate.layer)
            # Non-onset lanes should only borrow overlay previews from onset-like source lanes.
            # This prevents cross-class ghost boxes (for example kick events appearing on snare).
            if not target_is_onset and not candidate_is_onset:
                continue
            candidate_ids = {str(event.event_id) for event in candidate.events}
            overlap = len(lane_source_ids.intersection(candidate_ids))
            candidate_title = str(candidate.layer.title or "").strip().lower()
            candidate_source_id = str(candidate.layer.status.source_layer_id or "").strip()
            score = float(overlap * 100)
            if "onset" in candidate_title:
                score += 35.0
            if layer_source_id and candidate_source_id == layer_source_id:
                score += 20.0
            if candidate.layer.layer_id == layer.layer_id:
                score += 10.0
            if not lane_source_ids and "onset" in candidate_title:
                score += 25.0
            if score <= best_score:
                continue
            best_score = score
            best_lane = candidate

        if best_lane is not None and best_score > 0.0:
            return best_lane
        if target_is_onset:
            return _FixCandidateLane(
                layer=layer,
                take_id=take_id,
                events=lane_events,
            )
        return None

    def _fix_candidate_lanes(self: Any) -> list[_FixCandidateLane]:
        lanes: list[_FixCandidateLane] = []
        for layer in self.presentation.layers:
            if layer.kind.name != "EVENT":
                continue
            if layer.main_take_id is not None:
                lanes.append(
                    _FixCandidateLane(
                        layer=layer,
                        take_id=layer.main_take_id,
                        events=layer.events,
                    )
                )
            for take in layer.takes:
                if take.kind.name != "EVENT":
                    continue
                lanes.append(
                    _FixCandidateLane(
                        layer=layer,
                        take_id=take.take_id,
                        events=take.events,
                    )
                )
        return lanes

    def _fix_overlay_matched_source_ids(
        self: Any,
        *,
        lane_events: list[EventPresentation],
    ) -> set[str]:
        return {
            str(event.source_event_id or event.event_id)
            for event in lane_events
            if not self._event_is_demoted(event)
        }

    def _visible_lane_events(
        self: Any,
        lane_events: list[EventPresentation],
    ) -> list[EventPresentation]:
        if self._edit_mode == "fix":
            return list(lane_events)
        return [event for event in lane_events if not self._event_is_demoted(event)]

    @staticmethod
    def _event_is_demoted(event: EventPresentation) -> bool:
        return any(str(badge).strip().lower() == "demoted" for badge in (event.badges or []))

    @staticmethod
    def _is_onset_layer(layer: LayerPresentation) -> bool:
        title = str(layer.title or "").strip().lower()
        source_label = str(getattr(layer.status, "source_label", "") or "").strip().lower()
        output_name = str(getattr(layer.status, "output_name", "") or "").strip().lower()
        pipeline_id = str(getattr(layer.status, "pipeline_id", "") or "").strip().lower()
        return any("onset" in value for value in (title, source_label, output_name, pipeline_id))

    def _draw_note_contour_overlay(
        self: Any,
        painter: QPainter,
        *,
        layer: LayerPresentation,
        top: float,
        row_height: float,
        dimmed: bool,
    ) -> None:
        if layer.kind is not LayerKind.AUDIO:
            return
        contour_layer = self._resolve_note_contour_overlay_layer(layer)
        if contour_layer is None:
            return
        samples = contour_samples_from_events(contour_layer.events)
        if len(samples) < 2:
            return
        path = build_note_contour_path(
            samples,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=float(self._header_width),
            top=top,
            row_height=row_height,
        )
        if path is None:
            return
        overlay_color = QColor(layer.color or self._style.event_lane.default_fill_hex)
        overlay_color = overlay_color.lighter(165)
        overlay_color.setAlpha(120 if dimmed else NOTE_CONTOUR_ALPHA)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(overlay_color, NOTE_CONTOUR_PEN_WIDTH_PX))
        painter.drawPath(path)
        painter.restore()

    def _resolve_note_contour_overlay_layer(
        self: Any,
        layer: LayerPresentation,
    ) -> LayerPresentation | None:
        layer_id = str(layer.layer_id)
        candidates = [
            candidate
            for candidate in self.presentation.layers
            if (
                str(candidate.parent_layer_id or "") == layer_id
                or str(getattr(candidate.status, "source_layer_id", "") or "").strip() == layer_id
            )
            and str(getattr(candidate.status, "pipeline_id", "") or "").strip()
            == "extract_note_contour"
        ]
        if not candidates:
            return None
        return candidates[0]

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

    def _draw_playback_start_marker(self: Any, painter: QPainter) -> None:
        x = timeline_x_for_time(
            self.presentation.playback_start,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        if x < self._header_width or x > self.width():
            return

        marker = QColor("#9ca3af")
        marker.setAlpha(135)
        painter.save()
        try:
            pen = QPen(marker, 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(x), 0, int(x), self.height())
        finally:
            painter.restore()

    def _draw_header_content_divider(self: Any, painter: QPainter) -> None:
        divider_left = int(max(0, self._header_width - 1))
        divider_width = min(2, max(1, self.width() - divider_left))
        painter.fillRect(
            divider_left,
            0,
            divider_width,
            self.height(),
            QColor(self._style.canvas.split_divider_hex),
        )

    def _draw_interaction_overlays(self: Any, painter: QPainter) -> None:
        self._draw_fix_tool_overlay(painter)
        focused_fix_rect = self._focused_fix_overlay()
        if focused_fix_rect is not None:
            rect, _layer_id, _take_id, _source_event_id, _start, _end, matched = focused_fix_rect
            focus_outline = QColor(EVENT_SELECTION_COLOR)
            focus_outline.setAlpha(180 if matched else 220)
            focus_fill = QColor(EVENT_SELECTION_COLOR)
            focus_fill.setAlpha(26 if matched else 40)
            outline_width = EVENT_SELECTION_BORDER_PX
            if rect.width() <= EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX:
                outline_width += EVENT_SELECTION_TINY_WIDTH_EXTRA_PX
            outline_expand = float(EVENT_SELECTION_OUTLINE_EXPAND_PX)
            painter.save()
            painter.setPen(QPen(focus_outline, outline_width))
            painter.setBrush(QBrush(focus_fill))
            painter.drawRoundedRect(
                rect.adjusted(
                    -outline_expand,
                    -outline_expand,
                    outline_expand,
                    outline_expand,
                ),
                3.0 + outline_expand,
                3.0 + outline_expand,
            )
            painter.restore()

        if self._snap_indicator_time is not None:
            x = timeline_x_for_time(
                self._snap_indicator_time,
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=self._header_width,
            )
            if self._header_width <= x <= self.width():
                snap_color = QColor(EVENT_SELECTION_COLOR)
                drag_preview_active = bool(
                    self._drag_candidate is not None and self._dragging_events
                )
                if drag_preview_active:
                    snap_color.setAlpha(int(MOVE_DRAG_PREVIEW_LINE_ALPHA))
                else:
                    snap_color.setAlpha(110)
                painter.save()
                if drag_preview_active:
                    painter.setPen(
                        QPen(
                            snap_color,
                            int(MOVE_DRAG_PREVIEW_LINE_WIDTH_PX),
                            Qt.PenStyle.SolidLine,
                        )
                    )
                else:
                    painter.setPen(QPen(snap_color, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(x), int(self._top_padding), int(x), self.height())
                painter.restore()

        if self._preview_event_rect is not None:
            preview_color = QColor(EVENT_SELECTION_COLOR)
            preview_color.setAlpha(52)
            painter.save()
            painter.setPen(QPen(QColor(EVENT_SELECTION_COLOR), 1, Qt.PenStyle.DashLine))
            painter.setBrush(preview_color)
            painter.drawRoundedRect(self._preview_event_rect, 3.0, 3.0)
            painter.restore()

        if self._video_drag_candidate is not None:
            start = self._move_drag_preview_time
            if start is None:
                start = float(self._video_drag_candidate["anchor_start_seconds"])
            duration = max(
                0.05,
                float(self._video_drag_candidate["anchor_visible_duration_seconds"]),
            )
            if self._video_drag_preview_values is not None:
                start, _trim_start, duration, _loop_enabled = self._video_drag_preview_values
            x = timeline_x_for_time(
                float(start),
                scroll_x=self.presentation.scroll_x,
                pixels_per_second=self.presentation.pixels_per_second,
                content_start_x=self._header_width,
            )
            width = max(2.0, float(duration) * max(1.0, self.presentation.pixels_per_second))
            preview_rect = QRectF(
                x,
                float(self._video_drag_candidate["rect_top"]),
                width,
                float(self._video_drag_candidate["rect_height"]),
            )
            preview_color = QColor(EVENT_SELECTION_COLOR)
            preview_color.setAlpha(40)
            painter.save()
            painter.setPen(QPen(QColor(EVENT_SELECTION_COLOR), 1, Qt.PenStyle.DashLine))
            painter.setBrush(preview_color)
            painter.drawRoundedRect(preview_rect, 3.0, 3.0)
            if self._video_drag_preview_values is not None:
                _start, trim_start, preview_duration, loop_enabled = (
                    self._video_drag_preview_values
                )
                if loop_enabled:
                    self._draw_video_loop_affordances(
                        painter,
                        rect=preview_rect,
                        placement=VideoPlacement(
                            start_seconds=float(start),
                            trim_start_seconds=float(trim_start),
                            visible_duration_seconds=float(preview_duration),
                            source_duration_seconds=float(
                                self._video_drag_candidate["source_duration_seconds"]
                            ),
                            loop_enabled=True,
                        ).normalized(),
                        color=QColor(EVENT_SELECTION_COLOR),
                        dimmed=False,
                    )
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

        if self._layer_drag_target_y is not None:
            marker_color = QColor(EVENT_SELECTION_COLOR)
            marker_color.setAlpha(190)
            painter.save()
            painter.setPen(QPen(marker_color, 2))
            painter.drawLine(
                int(self._header_width),
                int(self._layer_drag_target_y),
                int(self.width()),
                int(self._layer_drag_target_y),
            )
            painter.restore()

    def _draw_fix_tool_overlay(self: Any, painter: QPainter) -> None:
        if self._edit_mode != "fix":
            return
        left = int(self._header_width + 12)
        top = int(max(6, self._top_padding + 4))
        width = max(200, self.width() - left - 12)
        current_tool = {
            "remove": "- Demote",
            "select": "Click",
            "promote": "+ Promote",
        }.get(str(self._fix_action).strip().lower(), "Click")
        demoted_nav_label = "on" if bool(self._fix_nav_include_demoted) else "off"
        hint_text = (
            f"Fix: {current_tool}  |  Z -  |  X Select  |  C +  |  +/- toggle  |  "
            f"Drag marquee applies current tool  |  Arrows navigate events (demoted {demoted_nav_label})  |  "
            "D toggles demoted nav  |  "
            ",/. navigate preview  |  Shift+Space/Enter preview"
        )
        painter.save()
        try:
            overlay_rect = QRectF(float(left), float(top), float(width), 22.0)
            overlay_fill = QColor("#101010")
            overlay_fill.setAlpha(210)
            painter.setBrush(QBrush(overlay_fill))
            painter.setPen(QPen(QColor("#685f67"), 1))
            painter.drawRoundedRect(overlay_rect, 2.0, 2.0)
            painter.setPen(QPen(QColor("#d8d2cb"), 1))
            painter.drawText(
                overlay_rect.adjusted(8.0, 0.0, -8.0, 0.0),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                hint_text,
            )
        finally:
            painter.restore()
