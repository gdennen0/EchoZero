"""Stage Zero timeline shell composed from reusable blocks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QScrollBar, QToolTip

from echozero.application.presentation.models import TimelinePresentation, LayerPresentation, TakeLanePresentation
from echozero.application.shared.enums import FollowMode
from echozero.application.timeline.intents import (
    ClearSelection,
    DuplicateSelectedEvents,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    Pause,
    Play,
    Seek,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectTake,
    Stop,
    ToggleMute,
    ToggleSolo,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.perf import timed
from echozero.ui.FEEL import (
    EVENT_BAR_HEIGHT_PX,
    GRID_LINE_ALPHA,
    GRID_LINE_COLOR,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
    LAYER_ROW_HEIGHT_PX,
    RULER_HEIGHT_PX,
    TAKE_ROW_HEIGHT_PX,
    TIMELINE_RIGHT_PADDING_PX,
)
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock, EventLanePresentation
from echozero.ui.qt.timeline.blocks.layer_header import HeaderSlots, LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.layouts import MainRowLayout, TakeRowLayout
from echozero.ui.qt.timeline.blocks.ruler import (
    RulerBlock,
    RulerLayout,
    playhead_head_polygon,
    seek_time_for_x,
    timeline_x_for_time,
    visible_ruler_seconds,
)
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.blocks.transport_bar_block import TransportBarBlock
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLaneBlock, WaveformLanePresentation


_SPAN_CACHE: dict[tuple, float] = {}
_MAX_SPAN_CACHE_ENTRIES = 24
_ZOOM_STEP_FACTOR = 1.12
_ZOOM_MIN_PPS = 40.0
_ZOOM_MAX_PPS = 720.0


def _span_signature(presentation: TimelinePresentation) -> tuple:
    layer_sig: list[tuple[int, tuple[int, ...]]] = []
    for layer in presentation.layers:
        take_sig = tuple(id(take.events) for take in layer.takes)
        layer_sig.append((id(layer.events), take_sig))
    return (id(presentation.layers), tuple(layer_sig), presentation.end_time_label)


def estimate_timeline_span_seconds(presentation: TimelinePresentation) -> float:
    """Best-effort duration estimate for viewport/scroll math (memoized by lane identity)."""
    key = _span_signature(presentation)
    cached = _SPAN_CACHE.get(key)
    if cached is not None:
        return cached

    span = max(0.0, presentation.playhead)
    for layer in presentation.layers:
        for event in layer.events:
            span = max(span, event.end)
        for take in layer.takes:
            for event in take.events:
                span = max(span, event.end)

    span = max(span, _parse_time_label_seconds(presentation.end_time_label))
    resolved = max(0.0, span)

    _SPAN_CACHE[key] = resolved
    if len(_SPAN_CACHE) > _MAX_SPAN_CACHE_ENTRIES:
        oldest = next(iter(_SPAN_CACHE.keys()))
        _SPAN_CACHE.pop(oldest, None)
    return resolved


def compute_scroll_bounds(
    presentation: TimelinePresentation,
    viewport_width: int,
    *,
    header_width: int = LAYER_HEADER_WIDTH_PX,
    right_padding_px: int = TIMELINE_RIGHT_PADDING_PX,
) -> tuple[int, int]:
    """Return (content_width, max_scroll_x) for horizontal timeline navigation."""
    viewport = max(1, int(viewport_width))
    span = estimate_timeline_span_seconds(presentation)
    content_width = max(viewport, int(header_width + (span * presentation.pixels_per_second) + right_padding_px))
    max_scroll = max(0, content_width - viewport)
    return content_width, max_scroll


def compute_follow_scroll_x(
    presentation: TimelinePresentation,
    viewport_width: int,
    *,
    header_width: int = LAYER_HEADER_WIDTH_PX,
    content_padding_px: int = 24,
) -> float:
    """Compute follow-mode adjusted scroll target for the current playhead."""
    if presentation.follow_mode == FollowMode.OFF or not presentation.is_playing:
        return presentation.scroll_x

    viewport = max(1, int(viewport_width))
    content_width = max(1.0, viewport - header_width)
    pps = max(1.0, presentation.pixels_per_second)
    timeline_x = presentation.playhead * pps
    current = presentation.scroll_x
    left_bound = current + content_padding_px
    right_bound = current + max(content_padding_px + 1.0, content_width - content_padding_px)

    target = current
    if presentation.follow_mode == FollowMode.PAGE:
        if timeline_x < left_bound:
            target = max(0.0, timeline_x - content_padding_px)
        elif timeline_x > right_bound:
            target = max(0.0, timeline_x - content_padding_px)
    elif presentation.follow_mode == FollowMode.CENTER:
        target = max(0.0, timeline_x - (content_width * 0.5))
    elif presentation.follow_mode == FollowMode.SMOOTH:
        # Match EZ1 semantics: keep playhead around 75% of the viewport
        # rather than centered, reducing forward-jump feel.
        target = max(0.0, timeline_x - (content_width * 0.75))

    _, max_scroll = compute_scroll_bounds(presentation, viewport, header_width=header_width)
    return float(max(0.0, min(target, max_scroll)))


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


def _parse_time_label_seconds(label: str | None) -> float:
    if not label:
        return 0.0
    text = label.strip()
    if not text:
        return 0.0
    try:
        if ':' in text:
            mins_txt, secs_txt = text.split(':', 1)
            return max(0.0, int(mins_txt) * 60 + float(secs_txt))
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return 0.0


class TimelineCanvas(QWidget):
    layer_clicked = pyqtSignal(object)
    mute_clicked = pyqtSignal(object)
    solo_clicked = pyqtSignal(object)
    take_toggle_clicked = pyqtSignal(object)
    take_selected = pyqtSignal(object, object)
    event_selected = pyqtSignal(object, object, object, str)
    move_selected_events_requested = pyqtSignal(float, object)
    take_action_selected = pyqtSignal(object, object, str)
    horizontal_scroll_requested = pyqtSignal(int)
    zoom_requested = pyqtSignal(int, float)
    playhead_drag_requested = pyqtSignal(float)
    clear_selection_requested = pyqtSignal()
    select_all_requested = pyqtSignal()
    nudge_requested = pyqtSignal(int, int)
    duplicate_requested = pyqtSignal(int)

    def __init__(self, presentation: TimelinePresentation, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._header_width = LAYER_HEADER_WIDTH_PX
        self._top_padding = LAYER_HEADER_TOP_PADDING_PX
        self._main_row_height = LAYER_ROW_HEIGHT_PX
        self._take_row_height = TAKE_ROW_HEIGHT_PX
        self._event_height = EVENT_BAR_HEIGHT_PX
        self._take_rects: list[tuple[QRectF, object, object]] = []
        self._take_option_rects: list[tuple[QRectF, object, object]] = []
        self._take_action_rects: list[tuple[QRectF, object, object, str]] = []
        self._open_take_options: set[tuple[object, object]] = set()
        self._toggle_rects: list[tuple[QRectF, object]] = []
        self._mute_rects: list[tuple[QRectF, object]] = []
        self._solo_rects: list[tuple[QRectF, object]] = []
        self._event_rects: list[tuple[QRectF, object, object | None, object]] = []
        self._header_select_rects: list[tuple[QRectF, object]] = []
        self._row_body_select_rects: list[tuple[QRectF, object]] = []
        self._header_hover_rects: list[tuple[QRectF, LayerPresentation]] = []
        self._event_drop_rects: list[tuple[QRectF, object]] = []
        self._hovered_layer_id: object | None = None
        self._dragging_playhead = False
        self._drag_candidate: dict[str, object] | None = None
        self._dragging_events = False
        self._header_block = LayerHeaderBlock()
        self._waveform_block = WaveformLaneBlock()
        self._event_lane_block = EventLaneBlock()
        self._take_row_block = TakeRowBlock()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(1440)
        self.setMouseTracking(True)
        self._recompute_height()

    def _recompute_height(self) -> None:
        height = self._top_padding
        for layer in self.presentation.layers:
            height += self._main_row_height
            if layer.is_expanded:
                height += len(layer.takes) * self._take_row_height
        self.setMinimumHeight(max(320, height + 12))

    def _any_solo(self) -> bool:
        return any(layer.soloed for layer in self.presentation.layers)

    def _layer_dimmed(self, layer: LayerPresentation) -> bool:
        if layer.muted:
            return True
        if self._any_solo() and not layer.soloed:
            return True
        return False

    def set_presentation(self, presentation: TimelinePresentation, *, recompute_layout: bool = True) -> None:
        self.presentation = presentation
        if recompute_layout:
            self._recompute_height()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#12151b'))
        self._take_rects.clear()
        self._take_option_rects.clear()
        self._take_action_rects.clear()
        self._toggle_rects.clear()
        self._mute_rects.clear()
        self._solo_rects.clear()
        self._event_rects.clear()
        self._header_select_rects.clear()
        self._row_body_select_rects.clear()
        self._header_hover_rects.clear()
        self._event_drop_rects.clear()
        with timed("timeline.paint.layers"):
            self._draw_layers(painter)
        with timed("timeline.paint.playhead"):
            self._draw_playhead(painter)

    def _draw_time_grid_band(self, painter: QPainter, *, top: int, row_height: int) -> None:
        content_left = float(self._header_width)
        content_width = max(1.0, float(self.width()) - content_left)
        marks = visible_ruler_seconds(
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_width=content_width,
            content_start_x=content_left,
        )

        grid_color = QColor(GRID_LINE_COLOR)
        grid_color.setAlpha(max(0, min(255, GRID_LINE_ALPHA)))
        painter.setPen(QPen(grid_color, 1))

        band_top = int(top)
        band_bottom = int(top + max(1, row_height) - 1)
        for _, x in marks:
            if x < content_left:
                continue
            painter.drawLine(int(x), band_top, int(x), band_bottom)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_playhead and event.buttons() & Qt.MouseButton.LeftButton:
            self.playhead_drag_requested.emit(self._seek_time_at_x(event.position().x()))
            event.accept()
            return

        if self._drag_candidate is not None and event.buttons() & Qt.MouseButton.LeftButton:
            dx = abs(event.position().x() - float(self._drag_candidate["anchor_x"]))
            dy = abs(event.position().y() - float(self._drag_candidate["anchor_y"]))
            if max(dx, dy) >= 4.0:
                self._dragging_events = True
                event.accept()
                return

        pos = event.position()
        hovered: LayerPresentation | None = None
        hovered_rect: QRectF | None = None
        for rect, layer in self._header_hover_rects:
            if rect.contains(pos):
                hovered = layer
                hovered_rect = rect
                break

        next_id = hovered.layer_id if hovered is not None else None
        if next_id != self._hovered_layer_id:
            self._hovered_layer_id = next_id
            if hovered is not None and hovered_rect is not None:
                tip = self._header_tooltip_text(hovered)
                if tip:
                    QToolTip.showText(
                        self.mapToGlobal(pos.toPoint()),
                        tip,
                        self,
                        hovered_rect.toRect(),
                        6000,
                    )
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered_layer_id = None
        self._dragging_playhead = False
        self._drag_candidate = None
        self._dragging_events = False
        QToolTip.hideText()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton and self._playhead_head_contains(pos):
            self._dragging_playhead = True
            self.playhead_drag_requested.emit(self._seek_time_at_x(pos.x()))
            event.accept()
            return
        for rect, layer_id in self._mute_rects:
            if rect.contains(pos):
                self.mute_clicked.emit(layer_id)
                return
        for rect, layer_id in self._solo_rects:
            if rect.contains(pos):
                self.solo_clicked.emit(layer_id)
                return
        for rect, layer_id, take_id, action_id in self._take_action_rects:
            if rect.contains(pos):
                self.take_action_selected.emit(layer_id, take_id, action_id)
                return
        for rect, layer_id, take_id in self._take_option_rects:
            if rect.contains(pos):
                key = (layer_id, take_id)
                if key in self._open_take_options:
                    self._open_take_options.remove(key)
                else:
                    self._open_take_options.add(key)
                self.update()
                return
        for rect, layer_id, take_id in self._take_rects:
            if rect.contains(pos):
                self.take_selected.emit(layer_id, take_id)
                return
        for rect, layer_id, take_id, event_id in self._event_rects:
            if rect.contains(pos):
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self._can_start_event_drag(event.modifiers(), event_id)
                ):
                    self._drag_candidate = {
                        "anchor_x": pos.x(),
                        "anchor_y": pos.y(),
                        "source_layer_id": layer_id,
                    }
                    self._dragging_events = False
                    event.accept()
                    return
                self.event_selected.emit(layer_id, take_id, event_id, self._selection_mode_for_modifiers(event.modifiers()))
                return
        for rect, layer_id in self._toggle_rects:
            if rect.contains(pos):
                self.take_toggle_clicked.emit(layer_id)
                return
        for rect, layer_id in self._header_select_rects:
            if rect.contains(pos):
                self.layer_clicked.emit(layer_id)
                return
        for rect, layer_id in self._row_body_select_rects:
            if rect.contains(pos):
                self.layer_clicked.emit(layer_id)
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_playhead = False
            if self._drag_candidate is not None:
                if self._dragging_events:
                    delta_seconds = (event.position().x() - float(self._drag_candidate["anchor_x"])) / max(1.0, self.presentation.pixels_per_second)
                    target_layer_id = self._event_drop_target_layer_id(event.position())
                    source_layer_id = self._drag_candidate["source_layer_id"]
                    if target_layer_id == source_layer_id:
                        target_layer_id = None
                    if abs(delta_seconds) >= 0.0001 or target_layer_id is not None:
                        self.move_selected_events_requested.emit(float(delta_seconds), target_layer_id)
                    event.accept()
                self._drag_candidate = None
                self._dragging_events = False
                if event.isAccepted():
                    return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta:
                self.zoom_requested.emit(int(delta), float(event.position().x()))
                event.accept()
                return

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta:
                self.horizontal_scroll_requested.emit(-delta)
                event.accept()
                return
        super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        modifiers = event.modifiers()
        has_primary = bool(
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        )
        steps = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
        if event.key() == Qt.Key.Key_Escape:
            self.clear_selection_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_A and has_primary:
            self.select_all_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_D and has_primary:
            self.duplicate_requested.emit(steps)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Left:
            self.nudge_requested.emit(-1, steps)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.nudge_requested.emit(1, steps)
            event.accept()
            return

        if has_primary and event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_requested.emit(120, float(self.width() * 0.5))
            event.accept()
            return
        if has_primary and event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_requested.emit(-120, float(self.width() * 0.5))
            event.accept()
            return

        super().keyPressEvent(event)

    @staticmethod
    def _selection_mode_for_modifiers(modifiers: Qt.KeyboardModifier) -> str:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return "additive"
        if modifiers & Qt.KeyboardModifier.ControlModifier or modifiers & Qt.KeyboardModifier.MetaModifier:
            return "toggle"
        return "replace"

    def _draw_layers(self, painter: QPainter) -> None:
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

    def _draw_main_row(self, painter: QPainter, layer: LayerPresentation, top: int) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = MainRowLayout.create(top=top, width=self.width(), header_width=self._header_width, row_height=self._main_row_height)
        row_bg = QColor('#1a212b' if layer.is_selected else '#161b22')
        if dimmed:
            row_bg = QColor('#12161c')
        painter.fillRect(layout.row_rect, row_bg)
        self._draw_time_grid_band(painter, top=top, row_height=self._main_row_height)
        painter.fillRect(0, top + self._main_row_height - 1, self.width(), 1, QColor('#252c38'))

        slots = HeaderSlots(
            rect=layout.header_rect,
            title_rect=layout.title_rect,
            subtitle_rect=layout.subtitle_rect,
            status_rect=layout.status_rect,
            controls_rect=layout.controls_rect,
            toggle_rect=layout.toggle_rect,
            metadata_rect=layout.metadata_rect,
        )
        self._toggle_rects.append((slots.toggle_rect, layer.layer_id))
        self._header_select_rects.append((layout.header_rect, layer.layer_id))
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        self._header_hover_rects.append((layout.header_rect, layer))
        if layer.kind.name == 'EVENT':
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
        hit_targets = self._header_block.paint(painter, slots, layer, dimmed=dimmed)
        self._mute_rects.append((hit_targets.mute_rect, layer.layer_id))
        self._solo_rects.append((hit_targets.solo_rect, layer.layer_id))

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if layer.kind.name == 'AUDIO':
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or '#9b87f5',
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
                    self._event_lane_block.paint(
                        painter,
                        top + 24,
                        EventLanePresentation(
                            layer_id=layer.layer_id,
                            take_id=layer.main_take_id,
                            events=layer.events,
                            pixels_per_second=self.presentation.pixels_per_second,
                            scroll_x=self.presentation.scroll_x,
                            header_width=self._header_width,
                            event_height=self._event_height,
                            dimmed=dimmed,
                            viewport_width=self.width(),
                        ),
                    )
                )
        finally:
            painter.restore()

    def _is_take_options_open(self, layer_id: object, take_id: object) -> bool:
        return (layer_id, take_id) in self._open_take_options

    def _draw_take_row(self, painter: QPainter, layer: LayerPresentation, take: TakeLanePresentation, top: int) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = TakeRowLayout.create(top=top, width=self.width(), header_width=self._header_width, row_height=self._take_row_height)
        options_open = self._is_take_options_open(layer.layer_id, take.take_id)
        hit_targets = self._take_row_block.paint_header(
            painter,
            layout,
            layer,
            take,
            options_open=options_open,
            dimmed=dimmed,
        )
        self._draw_time_grid_band(painter, top=top, row_height=self._take_row_height)
        self._take_rects.append(hit_targets.take_rect)
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        if take.kind.name == 'EVENT':
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
        if hit_targets.options_toggle_rect is not None:
            self._take_option_rects.append(hit_targets.options_toggle_rect)
        self._take_action_rects.extend(hit_targets.action_rects)

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if take.kind.name == 'AUDIO':
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or '#9b87f5',
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
                    self._event_lane_block.paint(
                        painter,
                        top + 10,
                        EventLanePresentation(
                            layer_id=layer.layer_id,
                            take_id=take.take_id,
                            events=take.events,
                            pixels_per_second=self.presentation.pixels_per_second,
                            scroll_x=self.presentation.scroll_x,
                            header_width=self._header_width,
                            event_height=self._event_height,
                            dimmed=True or dimmed,
                            viewport_width=self.width(),
                        ),
                    )
                )
        finally:
            painter.restore()

    def _draw_playhead(self, painter: QPainter) -> None:
        x = timeline_x_for_time(
            self.presentation.playhead,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        painter.setPen(QPen(QColor('#ff5f57'), 2))
        painter.drawLine(int(x), 0, int(x), self.height())
        if x >= self._header_width:
            painter.setBrush(QColor('#ff5f57'))
            painter.setPen(QPen(QColor('#ff5f57'), 1))
            painter.drawPolygon(playhead_head_polygon(x, float(self._top_padding)))

    def _playhead_head_contains(self, pos: QPointF) -> bool:
        x = timeline_x_for_time(
            self.presentation.playhead,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        return playhead_head_polygon(x, float(self._top_padding)).boundingRect().contains(pos)

    def _seek_time_at_x(self, x: float) -> float:
        return seek_time_for_x(
            x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )

    def _can_start_event_drag(self, modifiers: Qt.KeyboardModifier, event_id: object) -> bool:
        if modifiers & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False
        return event_id in set(self.presentation.selected_event_ids)

    def _event_drop_target_layer_id(self, pos: QPointF):
        for rect, layer_id in self._event_drop_rects:
            if rect.contains(pos):
                return layer_id
        return None


class TransportBar(QWidget):
    def __init__(self, presentation: TimelinePresentation, on_intent: Callable[[object], object | None] | None = None, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self._block = TransportBarBlock()
        self._control_rects: dict[str, QRectF] = {}
        self.setMinimumHeight(44)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._control_rects = self._block.paint(
            painter,
            TransportLayout.create(width=self.width(), height=self.height()),
            self.presentation,
        )

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        if (play_rect := self._control_rects.get('play')) is not None and play_rect.contains(pos):
            self._dispatch(Pause() if self.presentation.is_playing else Play())
            return
        if (stop_rect := self._control_rects.get('stop')) is not None and stop_rect.contains(pos):
            self._dispatch(Stop())
            return
        super().mousePressEvent(event)

    def _dispatch(self, intent: object) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            self.set_presentation(updated)


class TimelineRuler(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, presentation: TimelinePresentation, *, header_width: float = float(LAYER_HEADER_WIDTH_PX), parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._header_width = header_width
        self._block = RulerBlock()
        self._dragging = False
        self.setMinimumHeight(RULER_HEIGHT_PX)
        self.setMaximumHeight(RULER_HEIGHT_PX)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._block.paint(
            painter,
            RulerLayout(QRectF(0, 0, self.width(), self.height()), self._header_width),
            self.presentation,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().x() >= self._header_width:
            self._dragging = True
            self.seek_requested.emit(self._seek_time_at_x(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self._seek_time_at_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def _seek_time_at_x(self, x: float) -> float:
        return seek_time_for_x(
            x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )


class TimelineWidget(QWidget):
    def __init__(
        self,
        presentation: TimelinePresentation,
        on_intent: Callable[[object], TimelinePresentation] | None = None,
        *,
        runtime_audio: TimelineRuntimeAudioController | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self._runtime_audio = runtime_audio
        self._runtime_source_signature: tuple[tuple[str, str], ...] | None = None
        self._runtime_playhead_floor: float | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle('EchoZero Timeline Preview')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._transport = TransportBar(self.presentation, on_intent=self._dispatch)
        layout.addWidget(self._transport)

        self._canvas = TimelineCanvas(self.presentation)
        self._ruler = TimelineRuler(self.presentation, header_width=self._canvas._header_width)
        layout.addWidget(self._ruler)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet('background: #12151b; border: none;')
        self._canvas.layer_clicked.connect(self._select_layer)
        self._canvas.mute_clicked.connect(self._toggle_mute)
        self._canvas.solo_clicked.connect(self._toggle_solo)
        self._canvas.take_toggle_clicked.connect(self._toggle_take_selector)
        self._canvas.take_selected.connect(self._select_take)
        self._canvas.event_selected.connect(self._select_event)
        self._canvas.move_selected_events_requested.connect(self._move_selected_events)
        self._canvas.take_action_selected.connect(self._trigger_take_action)
        self._canvas.playhead_drag_requested.connect(self._seek)
        self._canvas.horizontal_scroll_requested.connect(self._scroll_horizontally_by_steps)
        self._canvas.zoom_requested.connect(self._zoom_from_input)
        self._canvas.clear_selection_requested.connect(self._clear_selection)
        self._canvas.select_all_requested.connect(self._select_all_events)
        self._canvas.nudge_requested.connect(self._nudge_selected_events)
        self._canvas.duplicate_requested.connect(self._duplicate_selected_events)
        self._ruler.seek_requested.connect(self._seek)
        self._scroll.setWidget(self._canvas)
        self.setFocusProxy(self._canvas)
        layout.addWidget(self._scroll)

        self._hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self._hscroll.setSingleStep(24)
        self._hscroll.setPageStep(200)
        self._hscroll.valueChanged.connect(self._on_horizontal_scroll_changed)
        layout.addWidget(self._hscroll)

        self._runtime_timer = QTimer(self)
        self._runtime_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._runtime_timer.setInterval(8)
        self._runtime_timer.timeout.connect(self._on_runtime_tick)
        self._runtime_timer.start()

        self.set_presentation(self.presentation)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        viewport = max(1, self._scroll.viewport().width())
        followed = compute_follow_scroll_x(
            presentation,
            viewport,
            header_width=self._canvas._header_width,
        )
        self.presentation = replace(presentation, scroll_x=followed)
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._transport.set_presentation(self.presentation)
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation)
        if self._runtime_audio is not None:
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_horizontal_scroll_bounds(sync_bar_value=False)

    def _update_horizontal_scroll_bounds(self, *, sync_bar_value: bool) -> None:
        viewport = max(1, self._scroll.viewport().width())
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

    def _on_horizontal_scroll_changed(self, value: int) -> None:
        next_scroll = float(max(0, value))
        if abs(next_scroll - self.presentation.scroll_x) < 0.5:
            return
        self.presentation = replace(self.presentation, scroll_x=next_scroll)
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)

    def _scroll_horizontally_by_steps(self, delta: int) -> None:
        if delta == 0:
            return
        notches = max(-6, min(6, int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1)))
        next_value = self._hscroll.value() + (notches * self._hscroll.singleStep())
        self._hscroll.setValue(max(self._hscroll.minimum(), min(self._hscroll.maximum(), next_value)))

    def _zoom_from_input(self, delta: int, anchor_x: float) -> None:
        if delta == 0:
            return
        factor = _ZOOM_STEP_FACTOR if delta > 0 else (1.0 / _ZOOM_STEP_FACTOR)
        self._apply_zoom_factor(factor, anchor_x=anchor_x)

    def _apply_zoom_factor(self, factor: float, *, anchor_x: float) -> None:
        current_pps = max(1.0, float(self.presentation.pixels_per_second))
        target_pps = max(_ZOOM_MIN_PPS, min(_ZOOM_MAX_PPS, current_pps * float(factor)))
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
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)

    def _dispatch(self, intent: object) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            # Viewport state (scroll/zoom) is owned by the widget, not the demo/app
            # transport intent responses. Preserve local viewport through dispatch.
            updated = replace(
                updated,
                scroll_x=self.presentation.scroll_x,
                scroll_y=self.presentation.scroll_y,
                pixels_per_second=self.presentation.pixels_per_second,
            )
            if self._runtime_audio is not None:
                runtime_time = self._runtime_audio.current_time_seconds()
                runtime_playing = self._runtime_audio.is_playing()
                if isinstance(intent, Seek):
                    runtime_time = max(0.0, float(intent.position))
                    self._runtime_playhead_floor = runtime_time if runtime_playing else None
                elif isinstance(intent, Stop):
                    runtime_time = 0.0
                    self._runtime_playhead_floor = None
                else:
                    runtime_time = self._stabilize_runtime_playhead(runtime_time, playing=runtime_playing)
                updated = replace(
                    updated,
                    playhead=runtime_time,
                    is_playing=runtime_playing,
                    current_time_label=_format_time_label(runtime_time),
                )
            self.set_presentation(updated)

    def _on_runtime_tick(self) -> None:
        if self._runtime_audio is None:
            return

        playing = self._runtime_audio.is_playing()
        current_time = self._stabilize_runtime_playhead(
            self._runtime_audio.current_time_seconds(),
            playing=playing,
        )
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
        followed = compute_follow_scroll_x(
            next_presentation,
            max(1, self._scroll.viewport().width()),
            header_width=self._canvas._header_width,
        )
        self.presentation = replace(next_presentation, scroll_x=followed)
        self._update_horizontal_scroll_bounds(sync_bar_value=True)
        self._transport.set_presentation(self.presentation)
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)

    def _seek(self, position: float) -> None:
        self._dispatch(Seek(position))

    def _select_layer(self, layer_id) -> None:
        self._dispatch(SelectLayer(layer_id))

    def _toggle_take_selector(self, layer_id) -> None:
        self._dispatch(ToggleLayerExpanded(layer_id))

    def _select_take(self, layer_id, take_id) -> None:
        if take_id is None:
            return
        self._dispatch(SelectTake(layer_id, take_id))

    def _select_event(self, layer_id, take_id, event_id, mode: str) -> None:
        if event_id is None:
            return
        self._dispatch(SelectEvent(layer_id, take_id, event_id, mode=mode))

    def _clear_selection(self) -> None:
        self._dispatch(ClearSelection())

    def _select_all_events(self) -> None:
        self._dispatch(SelectAllEvents())

    def _nudge_selected_events(self, direction: int, steps: int) -> None:
        self._dispatch(NudgeSelectedEvents(direction=direction, steps=steps))

    def _duplicate_selected_events(self, steps: int) -> None:
        self._dispatch(DuplicateSelectedEvents(steps=steps))

    def _trigger_take_action(self, layer_id, take_id, action_id: str) -> None:
        if take_id is None or not action_id:
            return
        self._dispatch(TriggerTakeAction(layer_id, take_id, action_id))

    def _move_selected_events(self, delta_seconds: float, target_layer_id) -> None:
        self._dispatch(MoveSelectedEvents(delta_seconds=delta_seconds, target_layer_id=target_layer_id))

    def _toggle_mute(self, layer_id) -> None:
        self._dispatch(ToggleMute(layer_id))

    def _toggle_solo(self, layer_id) -> None:
        self._dispatch(ToggleSolo(layer_id))

    def _stabilize_runtime_playhead(self, current_time: float, *, playing: bool) -> float:
        resolved = max(0.0, float(current_time))
        if not playing:
            self._runtime_playhead_floor = None
            return resolved

        floor = self._runtime_playhead_floor
        if floor is None:
            floor = max(0.0, float(self.presentation.playhead))
            self._runtime_playhead_floor = floor

        if resolved + 0.001 < floor:
            return floor

        self._runtime_playhead_floor = resolved
        return resolved


def _format_time_label(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


