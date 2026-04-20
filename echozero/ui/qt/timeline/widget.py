"""Stage Zero timeline shell composed from reusable blocks."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QContextMenuEvent, QCursor, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSplitter,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    InspectorContract,
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
    render_inspector_contract_text,
)
from echozero.application.presentation.models import (
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import FollowMode
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import (
    ApplyTransferPlan,
    ClearSelection,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    Pause,
    Play,
    Seek,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectTake,
    SetActivePlaybackTarget,
    SetSelectedEvents,
    Stop,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.models.paths import ensure_installed_models_dir
from echozero.perf import timed
from echozero.ui.FEEL import (
    DRAG_THRESHOLD_PX,
    EVENT_BAR_HEIGHT_PX,
    EVENT_SELECTION_COLOR,
    GRID_BAR_LINE_ALPHA,
    GRID_BEAT_LINE_ALPHA,
    GRID_LINE_ALPHA,
    GRID_LINE_COLOR,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
    LAYER_ROW_HEIGHT_PX,
    RULER_HEIGHT_PX,
    SNAP_MAGNETISM_RADIUS_PX,
    TAKE_ROW_HEIGHT_PX,
    TIMELINE_RIGHT_PADDING_PX,
    TIMELINE_ZOOM_MAX_PPS,
    TIMELINE_ZOOM_MIN_PPS,
    TIMELINE_ZOOM_STEP_FACTOR,
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
from echozero.ui.qt.timeline.blocks.waveform_lane import (
    WaveformLaneBlock,
    WaveformLanePresentation,
)
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullTimelineDialog,
    ManualPullTimelineSelectionResult,
)
from echozero.ui.qt.timeline.manual_pull import format_manual_pull_seconds as _format_seconds
from echozero.ui.qt.timeline.object_info_panel import ObjectInfoPanel
from echozero.ui.qt.timeline.runtime_audio import (
    RuntimeAudioTimingSnapshot,
    TimelineRuntimeAudioController,
)
from echozero.ui.qt.timeline.style import (
    TIMELINE_STYLE,
    TimelineShellStyle,
    build_timeline_scroll_area_stylesheet,
)
from echozero.ui.qt.timeline.time_grid import (
    TimelineGridMode,
    resolve_snap_time,
    visible_grid_lines,
)
from echozero.ui.qt.timeline.widget_actions import TimelineWidgetActionRouter
from echozero.ui.style.tokens import SHELL_TOKENS

_SPAN_CACHE: dict[tuple, float] = {}
_MAX_SPAN_CACHE_ENTRIES = 24


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
    content_width = max(
        viewport, int(header_width + (span * presentation.pixels_per_second) + right_padding_px)
    )
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
        # Keep the playhead around 75% of the viewport rather than centered
        # to reduce forward-jump feel during follow mode.
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
        if ":" in text:
            mins_txt, secs_txt = text.split(":", 1)
            return max(0.0, int(mins_txt) * 60 + float(secs_txt))
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return 0.0


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


@dataclass(slots=True)
class _ContextMenuEntry:
    label: str
    enabled: bool
    callback: Callable[[], None] | None = None


class TimelineEditorModeBar(QWidget):
    """Compact editor-mode strip for event-layer timeline interactions."""

    edit_mode_changed = pyqtSignal(str)
    snap_toggled = pyqtSignal(bool)
    grid_mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        for mode, label in (
            ("select", "Select"),
            ("draw", "Draw"),
            ("erase", "Erase"),
        ):
            button = QPushButton(label, self)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked=False, mode_name=mode: self.edit_mode_changed.emit(mode_name)
            )
            layout.addWidget(button)
            self._button_group.addButton(button)
            self._mode_buttons[mode] = button

        layout.addSpacing(12)
        self._snap_button = QPushButton("Snap", self)
        self._snap_button.setCheckable(True)
        self._snap_button.clicked.connect(lambda checked: self.snap_toggled.emit(bool(checked)))
        layout.addWidget(self._snap_button)

        self._grid_button = QPushButton("Grid: Auto", self)
        self._grid_button.clicked.connect(self._cycle_grid_mode)
        layout.addWidget(self._grid_button)
        layout.addStretch(1)

        self._grid_modes: tuple[TimelineGridMode, ...] = (TimelineGridMode.AUTO,)
        self._grid_mode = TimelineGridMode.AUTO

    def set_state(
        self,
        *,
        edit_mode: str,
        snap_enabled: bool,
        grid_mode: str,
        beat_available: bool,
    ) -> None:
        self._grid_modes = (
            (TimelineGridMode.AUTO, TimelineGridMode.BEAT, TimelineGridMode.OFF)
            if beat_available
            else (TimelineGridMode.AUTO, TimelineGridMode.OFF)
        )
        self._grid_mode = TimelineGridMode(str(grid_mode))
        if self._grid_mode not in self._grid_modes:
            self._grid_mode = self._grid_modes[0]

        for mode_name, button in self._mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(mode_name == edit_mode)
            button.blockSignals(False)
        self._snap_button.blockSignals(True)
        self._snap_button.setChecked(bool(snap_enabled))
        self._snap_button.blockSignals(False)
        self._grid_button.setText(f"Grid: {self._grid_mode.value.title()}")

    def _cycle_grid_mode(self) -> None:
        if not self._grid_modes:
            return
        try:
            current_index = self._grid_modes.index(self._grid_mode)
        except ValueError:
            current_index = 0
        next_mode = self._grid_modes[(current_index + 1) % len(self._grid_modes)]
        self._grid_mode = next_mode
        self._grid_button.setText(f"Grid: {next_mode.value.title()}")
        self.grid_mode_changed.emit(next_mode.value)


class TimelineCanvas(QWidget):
    layer_clicked = pyqtSignal(object, str)
    active_clicked = pyqtSignal(object)
    pipeline_actions_clicked = pyqtSignal(object)
    push_clicked = pyqtSignal(object)
    pull_clicked = pyqtSignal(object)
    take_toggle_clicked = pyqtSignal(object)
    take_selected = pyqtSignal(object, object)
    event_selected = pyqtSignal(object, object, object, str)
    move_selected_events_requested = pyqtSignal(float, object)
    take_action_selected = pyqtSignal(object, object, str)
    contract_action_selected = pyqtSignal(object)
    horizontal_scroll_requested = pyqtSignal(float)
    zoom_requested = pyqtSignal(int, float)
    playhead_drag_requested = pyqtSignal(float)
    clear_selection_requested = pyqtSignal()
    select_all_requested = pyqtSignal()
    set_selected_events_requested = pyqtSignal(object, object, object, object)
    create_event_requested = pyqtSignal(object, object, float, float)
    delete_events_requested = pyqtSignal(object)
    nudge_requested = pyqtSignal(int, int)
    duplicate_requested = pyqtSignal(int)
    edit_mode_requested = pyqtSignal(str)
    snap_toggle_requested = pyqtSignal()
    grid_mode_cycle_requested = pyqtSignal()
    preview_transfer_plan_requested = pyqtSignal()
    apply_transfer_plan_requested = pyqtSignal()
    cancel_transfer_plan_requested = pyqtSignal()

    def __init__(self, presentation: TimelinePresentation, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._style = TIMELINE_STYLE
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
        self._active_rects: list[tuple[QRectF, object]] = []
        self._pipeline_action_rects: list[tuple[QRectF, object]] = []
        self._push_rects: list[tuple[QRectF, object]] = []
        self._pull_rects: list[tuple[QRectF, object]] = []
        self._event_rects: list[tuple[QRectF, object, object | None, object]] = []
        self._event_lane_rects: list[tuple[QRectF, object, object | None]] = []
        self._header_select_rects: list[tuple[QRectF, object]] = []
        self._row_body_select_rects: list[tuple[QRectF, object]] = []
        self._header_hover_rects: list[tuple[QRectF, LayerPresentation]] = []
        self._event_drop_rects: list[tuple[QRectF, object]] = []
        self._hovered_layer_id: object | None = None
        self._dragging_playhead = False
        self._drag_candidate: dict[str, object] | None = None
        self._dragging_events = False
        self._selection_drag_candidate: dict[str, object] | None = None
        self._drawing_candidate: dict[str, object] | None = None
        self._marquee_rect: QRectF | None = None
        self._preview_event_rect: QRectF | None = None
        self._snap_indicator_time: float | None = None
        self._edit_mode = "select"
        self._snap_enabled = True
        self._grid_mode = TimelineGridMode.AUTO.value
        self._header_block = LayerHeaderBlock()
        self._waveform_block = WaveformLaneBlock()
        self._event_lane_block = EventLaneBlock()
        self._take_row_block = TakeRowBlock()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(1440)
        self.setMouseTracking(True)
        self._recompute_height()
        self._sync_cursor()

    def _recompute_height(self) -> None:
        height = self._top_padding
        for layer in self.presentation.layers:
            height += self._main_row_height
            if layer.is_expanded:
                height += len(layer.takes) * self._take_row_height
        self.setMinimumHeight(max(320, height + 12))

    def _layer_dimmed(self, layer: LayerPresentation) -> bool:
        return False

    def _push_outline_active_for_layer(self, layer: LayerPresentation) -> bool:
        if not self.presentation.manual_push_flow.push_mode_active:
            return False
        if layer.layer_id in set(self.presentation.manual_push_flow.selected_layer_ids):
            return True
        plan = self.presentation.batch_transfer_plan
        if plan is None or plan.operation_type not in {"push", "mixed"}:
            return False
        return any(
            row.direction == "push" and row.source_layer_id == layer.layer_id for row in plan.rows
        )

    def set_presentation(
        self, presentation: TimelinePresentation, *, recompute_layout: bool = True
    ) -> None:
        self.presentation = presentation
        if recompute_layout:
            self._recompute_height()
        self.update()

    def set_editor_state(
        self,
        *,
        edit_mode: str,
        snap_enabled: bool,
        grid_mode: str,
    ) -> None:
        self._edit_mode = edit_mode
        self._snap_enabled = bool(snap_enabled)
        self._grid_mode = grid_mode
        self._sync_cursor()
        self.update()

    def _sync_cursor(self) -> None:
        if self._edit_mode == "draw":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._edit_mode == "erase":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def paintEvent(self, event):
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

    def _draw_time_grid_band(self, painter: QPainter, *, top: int, row_height: int) -> None:
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

    def mouseMoveEvent(self, event) -> None:
        if self._drawing_candidate is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._update_draw_preview(event.position(), modifiers=event.modifiers())
            event.accept()
            return

        if (
            self._selection_drag_candidate is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            anchor = self._selection_drag_candidate["anchor_pos"]
            self._marquee_rect = QRectF(anchor, event.position()).normalized()
            event.accept()
            self.update()
            return

        if self._dragging_playhead and event.buttons() & Qt.MouseButton.LeftButton:
            self.playhead_drag_requested.emit(self._seek_time_at_x(event.position().x()))
            event.accept()
            return

        if self._drag_candidate is not None and event.buttons() & Qt.MouseButton.LeftButton:
            dx = abs(event.position().x() - float(self._drag_candidate["anchor_x"]))
            dy = abs(event.position().y() - float(self._drag_candidate["anchor_y"]))
            if max(dx, dy) >= DRAG_THRESHOLD_PX:
                self._dragging_events = True
                raw_delta = (event.position().x() - float(self._drag_candidate["anchor_x"])) / max(
                    1.0, self.presentation.pixels_per_second
                )
                anchor_time = float(self._drag_candidate["anchor_event_start"]) + raw_delta
                snapped = self._resolve_snap_target_time(
                    anchor_time,
                    modifiers=event.modifiers(),
                    exclude_event_ids=tuple(self.presentation.selected_event_ids),
                )
                self._snap_indicator_time = snapped
                self.update()
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
        self._selection_drag_candidate = None
        self._drawing_candidate = None
        self._marquee_rect = None
        self._preview_event_rect = None
        self._snap_indicator_time = None
        QToolTip.hideText()
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = event.position()

        if event.button() == Qt.MouseButton.RightButton:
            if self._show_context_menu(pos):
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton and self._playhead_head_contains(pos):
            self._dragging_playhead = True
            self.playhead_drag_requested.emit(self._seek_time_at_x(pos.x()))
            event.accept()
            return
        for rect, layer_id in self._active_rects:
            if rect.contains(pos):
                self.active_clicked.emit(layer_id)
                return
        for rect, layer_id in self._pipeline_action_rects:
            if rect.contains(pos):
                self.pipeline_actions_clicked.emit(layer_id)
                return
        for rect, layer_id in self._push_rects:
            if rect.contains(pos):
                self.push_clicked.emit(layer_id)
                return
        for rect, layer_id in self._pull_rects:
            if rect.contains(pos):
                self.pull_clicked.emit(layer_id)
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
                if self._edit_mode == "erase":
                    self.delete_events_requested.emit([event_id])
                    return
                if (
                    self._edit_mode == "select"
                    and event.button() == Qt.MouseButton.LeftButton
                    and self._can_start_event_drag(event.modifiers(), event_id)
                ):
                    self._drag_candidate = {
                        "anchor_x": pos.x(),
                        "anchor_y": pos.y(),
                        "source_layer_id": layer_id,
                        "anchor_event_id": event_id,
                        "anchor_event_start": self._event_start_for_id(event_id),
                    }
                    self._dragging_events = False
                    event.accept()
                    return
                self.event_selected.emit(
                    layer_id,
                    take_id,
                    event_id,
                    self._selection_mode_for_modifiers(event.modifiers()),
                )
                return
        lane_hit = self._event_lane_hit(pos)
        if lane_hit is not None and event.button() == Qt.MouseButton.LeftButton:
            lane_rect, lane_layer_id, lane_take_id = lane_hit
            del lane_rect
            if self._edit_mode == "draw":
                anchor_time = self._resolve_draw_time(pos.x(), modifiers=event.modifiers())
                self._drawing_candidate = {
                    "layer_id": lane_layer_id,
                    "take_id": lane_take_id,
                    "anchor_time": anchor_time,
                }
                self._preview_event_rect = None
                self._update_draw_preview(pos, modifiers=event.modifiers())
                event.accept()
                return
            if self._edit_mode == "select":
                self._selection_drag_candidate = {
                    "anchor_pos": pos,
                    "origin_layer_id": lane_layer_id,
                    "origin_take_id": lane_take_id,
                    "modifiers": event.modifiers(),
                }
                self._marquee_rect = None
                self._snap_indicator_time = None
                event.accept()
                return
        for rect, layer_id in self._toggle_rects:
            if rect.contains(pos):
                self.take_toggle_clicked.emit(layer_id)
                return
        for rect, layer_id in self._header_select_rects:
            if rect.contains(pos):
                self.layer_clicked.emit(
                    layer_id, self._layer_selection_mode_for_modifiers(event.modifiers())
                )
                return
        for rect, layer_id in self._row_body_select_rects:
            if rect.contains(pos):
                self.layer_clicked.emit(
                    layer_id, self._layer_selection_mode_for_modifiers(event.modifiers())
                )
                return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        if not self._show_context_menu(event.position(), global_pos=event.globalPos()):
            event.ignore()
            return
        event.accept()

    def _show_context_menu(self, pos: QPointF, *, global_pos=None) -> bool:
        hit_target = self._hit_target_for_position(pos)
        contract = build_timeline_inspector_contract(self.presentation, hit_target=hit_target)
        menu = self._build_context_menu(contract, hit_kind=hit_target.kind)
        if menu.isEmpty():
            return False
        if global_pos is None:
            global_pos = self.mapToGlobal(pos.toPoint())
        chosen = menu.exec(global_pos)
        if chosen is None:
            return True
        payload = chosen.data()
        if isinstance(payload, InspectorAction):
            self.contract_action_selected.emit(payload)
        return True

    def _build_context_menu(
        self, contract: InspectorContract, *, hit_kind: str | None = None
    ) -> QMenu:
        menu = QMenu(self)
        first_section = True
        seen_action_ids: set[str] = set()
        for section in contract.context_sections:
            visible_actions: list[InspectorAction] = []
            for action in section.actions:
                if hit_kind is not None and not self._context_action_visible_for_hit_kind(
                    action, hit_kind
                ):
                    continue
                if hit_kind is not None and action.action_id in seen_action_ids:
                    continue
                visible_actions.append(action)
                if hit_kind is not None:
                    seen_action_ids.add(action.action_id)
            if not visible_actions:
                continue
            if not first_section:
                menu.addSeparator()
            first_section = False
            for action in visible_actions:
                qt_action = menu.addAction(action.label)
                qt_action.setEnabled(action.enabled)
                qt_action.setData(action)
        return menu

    @staticmethod
    def _context_action_visible_for_hit_kind(action: InspectorAction, hit_kind: str) -> bool:
        group = (action.group or "").strip().lower()
        kind = (hit_kind or "").strip().lower()
        allowed_groups_by_kind = {
            "timeline": {"tools", "transport"},
            "layer": {"layer", "gain", "pipeline", "transfer", "live_sync", "transport"},
            "take": {"take", "transport"},
            "event": {"selection", "take", "transport"},
        }
        allowed = allowed_groups_by_kind.get(kind)
        if allowed is None:
            return True
        return group in allowed

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_playhead = False
            if self._drawing_candidate is not None:
                self._commit_draw_preview(event.position(), modifiers=event.modifiers())
                event.accept()
                return
            if self._selection_drag_candidate is not None:
                self._commit_selection_drag()
                event.accept()
                return
            if self._drag_candidate is not None:
                if self._dragging_events:
                    delta_seconds = (
                        event.position().x() - float(self._drag_candidate["anchor_x"])
                    ) / max(1.0, self.presentation.pixels_per_second)
                    anchor_start = float(self._drag_candidate["anchor_event_start"])
                    snapped_time = self._resolve_snap_target_time(
                        anchor_start + delta_seconds,
                        modifiers=event.modifiers(),
                        exclude_event_ids=tuple(self.presentation.selected_event_ids),
                    )
                    if snapped_time is not None:
                        delta_seconds = snapped_time - anchor_start
                    target_layer_id = self._event_drop_target_layer_id(event.position())
                    source_layer_id = self._drag_candidate["source_layer_id"]
                    if target_layer_id == source_layer_id:
                        target_layer_id = None
                    if abs(delta_seconds) >= 0.0001 or target_layer_id is not None:
                        self.move_selected_events_requested.emit(
                            float(delta_seconds), target_layer_id
                        )
                    event.accept()
                self._drag_candidate = None
                self._dragging_events = False
                self._snap_indicator_time = None
                self.update()
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

        horizontal_delta = self._horizontal_pan_delta(event)
        if horizontal_delta:
            self.horizontal_scroll_requested.emit(horizontal_delta)
            event.accept()
            return
        super().wheelEvent(event)

    @staticmethod
    def _horizontal_pan_delta(event: QWheelEvent) -> float:
        pixel_delta = event.pixelDelta()
        if pixel_delta.x():
            return float(-pixel_delta.x())

        angle_delta = event.angleDelta()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta = angle_delta.y() or angle_delta.x()
            return float(-delta) if delta else 0.0

        if abs(angle_delta.x()) > abs(angle_delta.y()) and angle_delta.x():
            return float(-angle_delta.x())
        return 0.0

    def keyPressEvent(self, event) -> None:
        modifiers = event.modifiers()
        has_primary = bool(
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        )
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        steps = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
        if event.key() == Qt.Key.Key_Escape:
            self.clear_selection_requested.emit()
            event.accept()
            return
        if has_primary and has_shift and event.key() == Qt.Key.Key_P:
            self.preview_transfer_plan_requested.emit()
            event.accept()
            return
        if has_primary and has_shift and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.apply_transfer_plan_requested.emit()
            event.accept()
            return
        if has_primary and has_shift and event.key() == Qt.Key.Key_Backspace:
            self.cancel_transfer_plan_requested.emit()
            event.accept()
            return
        if (
            event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete)
            and self.presentation.selected_event_ids
        ):
            self.delete_events_requested.emit(list(self.presentation.selected_event_ids))
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
        if not has_primary and not has_shift and event.key() == Qt.Key.Key_V:
            self.edit_mode_requested.emit("select")
            event.accept()
            return
        if not has_primary and not has_shift and event.key() == Qt.Key.Key_B:
            self.edit_mode_requested.emit("draw")
            event.accept()
            return
        if not has_primary and not has_shift and event.key() == Qt.Key.Key_E:
            self.edit_mode_requested.emit("erase")
            event.accept()
            return
        if not has_primary and event.key() == Qt.Key.Key_S:
            self.snap_toggle_requested.emit()
            event.accept()
            return
        if not has_primary and event.key() == Qt.Key.Key_G:
            self.grid_mode_cycle_requested.emit()
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
        if (
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        ):
            return "toggle"
        return "replace"

    @staticmethod
    def _layer_selection_mode_for_modifiers(modifiers: Qt.KeyboardModifier) -> str:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return "range"
        if (
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        ):
            return "toggle"
        return "replace"

    def _event_lane_hit(self, pos: QPointF) -> tuple[QRectF, object, object | None] | None:
        for rect, layer_id, take_id in self._event_lane_rects:
            if rect.contains(pos):
                return rect, layer_id, take_id
        return None

    def _resolve_draw_time(self, x: float, *, modifiers: Qt.KeyboardModifier) -> float:
        time_seconds = self._seek_time_at_x(x)
        snapped = self._resolve_snap_target_time(
            time_seconds, modifiers=modifiers, exclude_event_ids=()
        )
        return snapped if snapped is not None else time_seconds

    def _event_start_for_id(self, event_id: object) -> float:
        for layer in self.presentation.layers:
            for event in layer.events:
                if event.event_id == event_id:
                    return float(event.start)
            for take in layer.takes:
                for event in take.events:
                    if event.event_id == event_id:
                        return float(event.start)
        return 0.0

    def _event_times(self, *, exclude_event_ids: tuple[object, ...]) -> tuple[float, ...]:
        excluded = set(exclude_event_ids)
        times: list[float] = []
        for layer in self.presentation.layers:
            for event in layer.events:
                if event.event_id in excluded:
                    continue
                times.extend((float(event.start), float(event.end)))
            for take in layer.takes:
                for event in take.events:
                    if event.event_id in excluded:
                        continue
                    times.extend((float(event.start), float(event.end)))
        return tuple(times)

    def _resolve_snap_target_time(
        self,
        time_seconds: float,
        *,
        modifiers: Qt.KeyboardModifier,
        exclude_event_ids: tuple[object, ...],
    ) -> float | None:
        if not self._snap_enabled or modifiers & Qt.KeyboardModifier.AltModifier:
            return None
        resolved = resolve_snap_time(
            time_seconds,
            pixels_per_second=self.presentation.pixels_per_second,
            mode=self._grid_mode,
            bpm=self.presentation.bpm,
            threshold_px=float(SNAP_MAGNETISM_RADIUS_PX),
            event_times=self._event_times(exclude_event_ids=exclude_event_ids),
            playhead_time=self.presentation.playhead,
        )
        return resolved.time_seconds if resolved is not None else None

    def _update_draw_preview(self, pos: QPointF, *, modifiers: Qt.KeyboardModifier) -> None:
        if self._drawing_candidate is None:
            return
        anchor_time = float(self._drawing_candidate["anchor_time"])
        current_time = self._resolve_draw_time(pos.x(), modifiers=modifiers)
        if abs(current_time - anchor_time) < 1e-6:
            default_duration = 0.25
            current_time = anchor_time + default_duration
        start_time = max(0.0, min(anchor_time, current_time))
        end_time = max(start_time + 0.01, max(anchor_time, current_time))
        self._snap_indicator_time = current_time
        x = timeline_x_for_time(
            start_time,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        width = max(2.0, (end_time - start_time) * max(1.0, self.presentation.pixels_per_second))
        lane_hit = self._event_lane_hit(pos)
        if lane_hit is None:
            lane_hit = next(
                (
                    candidate
                    for candidate in self._event_lane_rects
                    if candidate[1] == self._drawing_candidate["layer_id"]
                    and candidate[2] == self._drawing_candidate["take_id"]
                ),
                None,
            )
        if lane_hit is not None:
            lane_rect, _layer_id, _take_id = lane_hit
            top = lane_rect.top() + max(0.0, (lane_rect.height() - self._event_height) * 0.5)
            self._preview_event_rect = QRectF(x, top, width, self._event_height)
        self.update()

    def _commit_draw_preview(self, pos: QPointF, *, modifiers: Qt.KeyboardModifier) -> None:
        if self._drawing_candidate is None:
            return
        anchor_time = float(self._drawing_candidate["anchor_time"])
        current_time = self._resolve_draw_time(pos.x(), modifiers=modifiers)
        if abs(current_time - anchor_time) < 1e-6:
            current_time = anchor_time + 0.25
        start_time = max(0.0, min(anchor_time, current_time))
        end_time = max(start_time + 0.01, max(anchor_time, current_time))
        self.create_event_requested.emit(
            self._drawing_candidate["layer_id"],
            self._drawing_candidate["take_id"],
            float(start_time),
            float(end_time),
        )
        self._drawing_candidate = None
        self._preview_event_rect = None
        self._snap_indicator_time = None
        self.update()

    def _commit_selection_drag(self) -> None:
        if self._selection_drag_candidate is None:
            return
        candidate = self._selection_drag_candidate
        rect = self._marquee_rect.normalized() if self._marquee_rect is not None else None
        self._selection_drag_candidate = None
        self._marquee_rect = None

        if rect is None or rect.width() < DRAG_THRESHOLD_PX and rect.height() < DRAG_THRESHOLD_PX:
            self.layer_clicked.emit(
                candidate["origin_layer_id"],
                self._layer_selection_mode_for_modifiers(candidate["modifiers"]),
            )
            self.update()
            return

        mode = self._selection_mode_for_modifiers(candidate["modifiers"])
        intersected_ids: list[object] = []
        for event_rect, _layer_id, _take_id, event_id in self._event_rects:
            if rect.intersects(event_rect):
                intersected_ids.append(event_id)

        next_event_ids = list(self.presentation.selected_event_ids)
        if mode == "replace":
            next_event_ids = intersected_ids
        elif mode == "additive":
            for event_id in intersected_ids:
                if event_id not in next_event_ids:
                    next_event_ids.append(event_id)
        else:
            for event_id in intersected_ids:
                if event_id in next_event_ids:
                    next_event_ids = [
                        candidate_id for candidate_id in next_event_ids if candidate_id != event_id
                    ]
                else:
                    next_event_ids.append(event_id)

        anchor_layer_id, anchor_take_id, selected_layer_ids = (
            self._selection_context_for_event_ids(next_event_ids)
        )
        self.set_selected_events_requested.emit(
            next_event_ids,
            anchor_layer_id,
            anchor_take_id,
            selected_layer_ids,
        )
        self.update()

    def _selection_context_for_event_ids(
        self, event_ids: list[object]
    ) -> tuple[object | None, object | None, list[object]]:
        lookup = set(event_ids)
        selected_layer_ids: list[object] = []
        anchor_layer_id = None
        anchor_take_id = None
        for layer in self.presentation.layers:
            layer_has_selected = False
            for event in layer.events:
                if event.event_id not in lookup:
                    continue
                layer_has_selected = True
                anchor_layer_id = layer.layer_id
                anchor_take_id = layer.main_take_id
            for take in layer.takes:
                for event in take.events:
                    if event.event_id not in lookup:
                        continue
                    layer_has_selected = True
                    anchor_layer_id = layer.layer_id
                    anchor_take_id = take.take_id
            if layer_has_selected and layer.layer_id not in selected_layer_ids:
                selected_layer_ids.append(layer.layer_id)
        return anchor_layer_id, anchor_take_id, selected_layer_ids

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
            outline_color = QColor("#8fd0ff")
            painter.save()
            painter.setPen(QPen(outline_color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(outline_rect, 8.0, 8.0)
            painter.restore()
        self._draw_time_grid_band(painter, top=top, row_height=self._main_row_height)
        painter.fillRect(
            0,
            top + self._main_row_height - 1,
            self.width(),
            1,
            QColor(self._style.canvas.row_divider_hex),
        )

        slots = HeaderSlots(
            rect=layout.header_rect,
            title_rect=layout.title_rect,
            subtitle_rect=layout.subtitle_rect,
            status_rect=layout.status_rect,
            controls_rect=layout.controls_rect,
            toggle_rect=layout.toggle_rect,
            metadata_rect=layout.metadata_rect,
        )
        if layer.takes:
            self._toggle_rects.append((slots.toggle_rect, layer.layer_id))
        self._header_select_rects.append((layout.header_rect, layer.layer_id))
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        self._header_hover_rects.append((layout.header_rect, layer))
        if layer.kind.name == "EVENT":
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
            self._event_lane_rects.append(
                (layout.content_rect, layer.layer_id, layer.main_take_id)
            )
        hit_targets = self._header_block.paint(painter, slots, layer, dimmed=dimmed)
        for control_id, rect in hit_targets.control_rects:
            if control_id == "set_active_playback_target":
                self._active_rects.append((rect, layer.layer_id))
            elif control_id == "layer_pipeline_actions":
                self._pipeline_action_rects.append((rect, layer.layer_id))
            elif control_id == "push_to_ma3":
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

    def _is_take_options_open(self, layer_id: object, take_id: object) -> bool:
        return (layer_id, take_id) in self._open_take_options

    def _draw_take_row(
        self, painter: QPainter, layer: LayerPresentation, take: TakeLanePresentation, top: int
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
        self._draw_time_grid_band(painter, top=top, row_height=self._take_row_height)
        self._take_rects.append(hit_targets.take_rect)
        self._row_body_select_rects.append((layout.content_rect, layer.layer_id))
        if take.kind.name == "EVENT":
            self._event_drop_rects.append((layout.content_rect, layer.layer_id))
            self._event_lane_rects.append((layout.content_rect, layer.layer_id, take.take_id))
        if hit_targets.options_toggle_rect is not None:
            self._take_option_rects.append(hit_targets.options_toggle_rect)
        self._take_action_rects.extend(hit_targets.action_rects)

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
        # Hide playhead when it is outside timeline content band.
        if x < self._header_width or x > self.width():
            return

        painter.setPen(
            QPen(QColor(self._style.playhead.color_hex), self._style.playhead.line_width_px)
        )
        painter.drawLine(int(x), 0, int(x), self.height())
        painter.setBrush(QColor(self._style.playhead.color_hex))
        painter.setPen(
            QPen(
                QColor(self._style.playhead.color_hex), self._style.playhead.head_outline_width_px
            )
        )
        painter.drawPolygon(playhead_head_polygon(x, float(self._top_padding)))

    def _draw_interaction_overlays(self, painter: QPainter) -> None:
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

    def _hit_target_for_position(self, pos: QPointF) -> TimelineInspectorHitTarget:
        for rect, layer_id, take_id, event_id in self._event_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(
                    kind="event",
                    layer_id=layer_id,
                    take_id=take_id,
                    event_id=event_id,
                    time_seconds=(
                        self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None
                    ),
                )
        for rect, layer_id, take_id in self._take_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(
                    kind="take",
                    layer_id=layer_id,
                    take_id=take_id,
                    time_seconds=(
                        self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None
                    ),
                )
        for rect, layer_id in self._header_select_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(kind="layer", layer_id=layer_id)
        for rect, layer_id in self._row_body_select_rects:
            if rect.contains(pos):
                return TimelineInspectorHitTarget(
                    kind="layer",
                    layer_id=layer_id,
                    time_seconds=(
                        self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None
                    ),
                )
        return TimelineInspectorHitTarget(
            kind="timeline",
            time_seconds=self._seek_time_at_x(pos.x()) if pos.x() >= self._header_width else None,
        )


class TransportBar(QWidget):
    def __init__(
        self,
        presentation: TimelinePresentation,
        on_intent: Callable[[object], object | None] | None = None,
        parent=None,
    ):
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
        if (play_rect := self._control_rects.get("play")) is not None and play_rect.contains(pos):
            self._dispatch(Pause() if self.presentation.is_playing else Play())
            return
        if (stop_rect := self._control_rects.get("stop")) is not None and stop_rect.contains(pos):
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

    def __init__(
        self,
        presentation: TimelinePresentation,
        *,
        header_width: float = float(LAYER_HEADER_WIDTH_PX),
        parent=None,
    ):
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
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().x() >= self._header_width
        ):
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
        self._style: TimelineShellStyle = TIMELINE_STYLE
        self.presentation = presentation
        self._on_intent = on_intent
        self._runtime_audio = runtime_audio
        self._runtime_source_signature: tuple[tuple[str, str], ...] | None = None
        self._runtime_playhead_floor: float | None = None
        self._runtime_timing_snapshot: RuntimeAudioTimingSnapshot | None = None
        self._edit_mode = "select"
        self._snap_enabled = True
        self._grid_mode = TimelineGridMode.AUTO.value
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle(self._style.window_title)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        left_pane = QWidget(self)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._transport = TransportBar(self.presentation, on_intent=self._dispatch)
        left_layout.addWidget(self._transport)

        self._editor_bar = TimelineEditorModeBar(self)
        self._editor_bar.edit_mode_changed.connect(self._set_edit_mode)
        self._editor_bar.snap_toggled.connect(self._set_snap_enabled)
        self._editor_bar.grid_mode_changed.connect(self._set_grid_mode)
        left_layout.addWidget(self._editor_bar)

        self._canvas = TimelineCanvas(self.presentation)
        self._ruler = TimelineRuler(self.presentation, header_width=self._canvas._header_width)
        left_layout.addWidget(self._ruler)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"background: {SHELL_TOKENS.canvas_bg}; border: none;")
        self._canvas.layer_clicked.connect(self._select_layer)
        self._canvas.active_clicked.connect(self._set_active_playback_target)
        self._canvas.pipeline_actions_clicked.connect(self._open_layer_pipeline_actions)
        self._canvas.push_clicked.connect(self._open_push_from_layer_action)
        self._canvas.pull_clicked.connect(self._open_pull_from_layer_action)
        self._canvas.take_toggle_clicked.connect(self._toggle_take_selector)
        self._canvas.take_selected.connect(self._select_take)
        self._canvas.event_selected.connect(self._select_event)
        self._canvas.move_selected_events_requested.connect(self._move_selected_events)
        self._canvas.take_action_selected.connect(self._trigger_take_action)
        self._canvas.contract_action_selected.connect(self._handle_contract_action)
        self._canvas.playhead_drag_requested.connect(self._seek)
        self._canvas.horizontal_scroll_requested.connect(self._scroll_horizontally_by_steps)
        self._canvas.zoom_requested.connect(self._zoom_from_input)
        self._canvas.clear_selection_requested.connect(self._clear_selection)
        self._canvas.select_all_requested.connect(self._select_all_events)
        self._canvas.set_selected_events_requested.connect(self._set_selected_events)
        self._canvas.create_event_requested.connect(self._create_event)
        self._canvas.delete_events_requested.connect(self._delete_events)
        self._canvas.nudge_requested.connect(self._nudge_selected_events)
        self._canvas.duplicate_requested.connect(self._duplicate_selected_events)
        self._canvas.edit_mode_requested.connect(self._set_edit_mode)
        self._canvas.snap_toggle_requested.connect(self._toggle_snap_enabled)
        self._canvas.grid_mode_cycle_requested.connect(self._cycle_grid_mode)
        self._canvas.preview_transfer_plan_requested.connect(self._preview_active_transfer_plan)
        self._canvas.apply_transfer_plan_requested.connect(self._apply_active_transfer_plan)
        self._canvas.cancel_transfer_plan_requested.connect(self._cancel_active_transfer_plan)
        self._ruler.seek_requested.connect(self._seek)
        self._scroll.setWidget(self._canvas)
        self.setFocusProxy(self._canvas)
        left_layout.addWidget(self._scroll)

        self._hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self._hscroll.setSingleStep(24)
        self._hscroll.setPageStep(200)
        self._hscroll.valueChanged.connect(self._on_horizontal_scroll_changed)
        left_layout.addWidget(self._hscroll)

        self._object_info = ObjectInfoPanel(self)
        self._object_info.action_requested.connect(self._handle_contract_action)
        self._object_info.settings_requested.connect(self._open_action_settings_dialog)
        self._object_info_panel = self._object_info
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(left_pane)
        self._main_splitter.addWidget(self._object_info)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.setSizes([1080, 320])
        root_layout.addWidget(self._main_splitter)

        self._runtime_timer = QTimer(self)
        self._runtime_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._runtime_timer.setInterval(8)
        self._runtime_timer.timeout.connect(self._on_runtime_tick)
        self._runtime_timer.start()
        self._action_router = TimelineWidgetActionRouter(
            widget=self,
            dispatch=self._dispatch,
            get_presentation=lambda: self.presentation,
            set_presentation=self.set_presentation,
            resolve_runtime_shell=self._resolve_runtime_shell,
            selected_event_ids_for_selected_layers=self._selected_event_ids_for_selected_layers,
            open_manual_pull_timeline_popup=lambda flow: self._open_manual_pull_timeline_popup(
                flow
            ),
            input_dialog=QInputDialog,
            file_dialog=QFileDialog,
            message_box=QMessageBox,
            resolve_models_dir=ensure_installed_models_dir,
        )

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
        self._reset_scroll_area_horizontal_offset()
        self._transport.set_presentation(self.presentation)
        self._sync_editor_state()
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
        self._reset_scroll_area_horizontal_offset()
        self._ruler.set_presentation(self.presentation)
        self._canvas.set_presentation(self.presentation, recompute_layout=False)

    def _scroll_horizontally_by_steps(self, delta: float) -> None:
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

    def _reset_scroll_area_horizontal_offset(self) -> None:
        bar = self._scroll.horizontalScrollBar()
        if bar.value() == 0:
            return
        bar.blockSignals(True)
        bar.setValue(0)
        bar.blockSignals(False)

    def _sync_editor_state(self) -> None:
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

    def _set_edit_mode(self, mode: str) -> None:
        normalized = (mode or "select").strip().lower()
        if normalized not in {"select", "draw", "erase"}:
            return
        self._edit_mode = normalized
        self._sync_editor_state()

    def _set_snap_enabled(self, enabled: bool) -> None:
        self._snap_enabled = bool(enabled)
        self._sync_editor_state()

    def _toggle_snap_enabled(self) -> None:
        self._set_snap_enabled(not self._snap_enabled)

    def _set_grid_mode(self, mode: str) -> None:
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

    def _cycle_grid_mode(self) -> None:
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

    def _zoom_from_input(self, delta: int, anchor_x: float) -> None:
        if delta == 0:
            return
        factor = TIMELINE_ZOOM_STEP_FACTOR if delta > 0 else (1.0 / TIMELINE_ZOOM_STEP_FACTOR)
        self._apply_zoom_factor(factor, anchor_x=anchor_x)

    def _apply_zoom_factor(self, factor: float, *, anchor_x: float) -> None:
        current_pps = max(1.0, float(self.presentation.pixels_per_second))
        target_pps = max(
            TIMELINE_ZOOM_MIN_PPS, min(TIMELINE_ZOOM_MAX_PPS, current_pps * float(factor))
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
                        runtime_time, playing=runtime_playing
                    )
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

    def _sample_runtime_playhead(self) -> tuple[float, bool]:
        snapshot = self._current_runtime_timing_snapshot()
        playing = snapshot.is_playing if snapshot is not None else self._runtime_audio.is_playing()
        current_time = self._resolve_runtime_time(snapshot)
        if snapshot is None:
            current_time = max(0.0, float(self._runtime_audio.current_time_seconds()))
        return current_time, playing

    def _current_runtime_timing_snapshot(self) -> RuntimeAudioTimingSnapshot | None:
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

    def _stabilize_runtime_playhead(self, runtime_time: float, *, playing: bool) -> float:
        next_time = max(0.0, float(runtime_time))
        if not playing:
            self._runtime_playhead_floor = None
            return next_time

        if self._runtime_playhead_floor is None:
            self._runtime_playhead_floor = next_time
            return next_time

        # Guard against stale backward clock samples during churn.
        if next_time + 1e-6 < self._runtime_playhead_floor:
            return self._runtime_playhead_floor

        self._runtime_playhead_floor = next_time
        return next_time

    def _seek(self, position: float) -> None:
        self._dispatch(Seek(position))

    def _select_layer(self, layer_id, mode: str = "replace") -> None:
        self._dispatch(SelectLayer(layer_id, mode=mode))

    def _toggle_take_selector(self, layer_id) -> None:
        self._dispatch(ToggleLayerExpanded(layer_id))

    def _select_take(self, layer_id, take_id) -> None:
        if take_id is None:
            return
        self._dispatch(SelectTake(layer_id, take_id))

    def _select_event(self, layer_id, take_id, event_id, mode: str) -> None:
        self._dispatch(SelectEvent(layer_id, take_id, event_id, mode=mode))

    def _set_selected_events(
        self, event_ids, anchor_layer_id, anchor_take_id, selected_layer_ids
    ) -> None:
        self._dispatch(
            SetSelectedEvents(
                event_ids=list(event_ids),
                anchor_layer_id=anchor_layer_id,
                anchor_take_id=anchor_take_id,
                selected_layer_ids=list(selected_layer_ids),
            )
        )

    def _create_event(self, layer_id, take_id, start_seconds: float, end_seconds: float) -> None:
        self._dispatch(
            CreateEvent(
                layer_id=layer_id,
                take_id=take_id,
                time_range=TimeRange(
                    start=max(0.0, min(float(start_seconds), float(end_seconds))),
                    end=max(float(start_seconds), float(end_seconds)),
                ),
            )
        )

    def _delete_events(self, event_ids) -> None:
        ids = list(event_ids)
        if not ids:
            return
        self._dispatch(DeleteEvents(event_ids=ids))

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
        self._dispatch(
            MoveSelectedEvents(delta_seconds=delta_seconds, target_layer_id=target_layer_id)
        )

    def _set_active_playback_target(self, layer_id) -> None:
        self._dispatch(SetActivePlaybackTarget(layer_id=layer_id, take_id=None))

    def _open_push_from_layer_action(self, layer_id) -> None:
        self._focus_layer_for_header_action(layer_id)
        scoped_event_ids = self._selected_event_ids_for_selected_layers()
        self._dispatch(OpenPushToMA3Dialog(selection_event_ids=scoped_event_ids))

    def _open_pull_from_layer_action(self, layer_id) -> None:
        self._focus_layer_for_header_action(layer_id)
        self._dispatch(OpenPullFromMA3Dialog())

    def _focus_layer_for_header_action(self, layer_id) -> None:
        if self.presentation.selected_layer_id != layer_id:
            self._dispatch(SelectLayer(layer_id))

    def _selected_event_ids_for_selected_layers(self) -> list:
        selected_layer_ids = set(self.presentation.selected_layer_ids)
        if not selected_layer_ids and self.presentation.selected_layer_id is not None:
            selected_layer_ids = {self.presentation.selected_layer_id}
        if not selected_layer_ids:
            return list(self.presentation.selected_event_ids)

        allowed_event_ids: set = set()
        for layer in self.presentation.layers:
            if layer.layer_id not in selected_layer_ids:
                continue
            for event in layer.events:
                allowed_event_ids.add(event.event_id)
        return [
            event_id
            for event_id in self.presentation.selected_event_ids
            if event_id in allowed_event_ids
        ]

    def _preview_active_transfer_plan(self) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.plan_preview",
                label=_preview_transfer_plan_label(plan),
                params={"plan_id": plan.plan_id},
            )
        )

    def _apply_active_transfer_plan(self) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.plan_apply",
                label=_apply_transfer_plan_label(plan),
                params={"plan_id": plan.plan_id},
            )
        )

    def _cancel_active_transfer_plan(self) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.plan_cancel",
                label="Cancel Transfer Plan",
                params={"plan_id": plan.plan_id},
            )
        )

    def _handle_contract_action(self, action: InspectorAction) -> None:
        if action.kind == "settings":
            self._action_router.open_object_action_settings(action)
            return
        self._action_router.trigger_contract_action(action)

    def _open_action_settings_dialog(self, action: InspectorAction) -> None:
        self._action_router.open_object_action_settings(action)

    def _open_layer_pipeline_actions(self, layer_id) -> None:
        self._focus_layer_for_header_action(layer_id)
        contract = build_timeline_inspector_contract(
            self.presentation,
            hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=layer_id),
        )
        pipeline_actions = [
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id.startswith("timeline.")
        ]
        if not pipeline_actions:
            return

        runtime = self._resolve_runtime_shell()
        describe = (
            getattr(runtime, "describe_object_action", None) if runtime is not None else None
        )
        menu = QMenu(self)
        for index, action in enumerate(pipeline_actions):
            if index:
                menu.addSeparator()
            plan = None
            if callable(describe):
                try:
                    plan = describe(
                        action.action_id,
                        action.params,
                        object_id=layer_id,
                        object_type="layer",
                    )
                except Exception:
                    plan = None
            settings_entry = menu.addAction(f"Open {action.label} Settings")
            settings_entry.setData(
                InspectorAction(
                    action_id=action.action_id,
                    label=action.label,
                    kind="settings",
                    params=dict(action.params),
                )
            )
            run_entry = menu.addAction(
                f"{plan.run_label} {action.label}" if plan is not None else f"Run {action.label}"
            )
            run_entry.setData(
                InspectorAction(
                    action_id=action.action_id,
                    label=action.label,
                    params=dict(action.params),
                )
            )

        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return
        payload = chosen.data()
        if isinstance(payload, InspectorAction):
            self._handle_contract_action(payload)

    def _trigger_contract_action(self, action: InspectorAction) -> None:
        self._handle_contract_action(action)

    def _handle_runtime_pipeline_action(self, action_id: str, params: dict[str, object]) -> bool:
        return self._action_router._handle_runtime_pipeline_action(action_id, params)

    def _open_manual_pull_timeline_popup(self, flow) -> ManualPullTimelineSelectionResult | None:
        return self._action_router._default_open_manual_pull_timeline_popup(flow)

    def _resolve_runtime_shell(self):
        owner = getattr(self._on_intent, "__self__", None)
        if owner is not None and all(
            hasattr(owner, method_name) for method_name in ("presentation",)
        ):
            return owner
        runtime = getattr(owner, "runtime", None)
        if runtime is not None and hasattr(runtime, "presentation"):
            return runtime
        return None

    def _refresh_object_info_panel(self) -> None:
        contract = build_timeline_inspector_contract(self.presentation)
        self._object_info.set_contract(self.presentation, contract)
        self._object_info.set_action_settings_plans(
            self._resolve_object_action_settings_plans(contract)
        )

    def _resolve_object_action_settings_plans(self, contract: InspectorContract) -> tuple:
        runtime = self._resolve_runtime_shell()
        describe = (
            getattr(runtime, "describe_object_action", None) if runtime is not None else None
        )
        if not callable(describe):
            return ()
        plans = []
        object_identity = contract.identity
        for section in contract.context_sections:
            for action in section.actions:
                if not action.action_id.startswith("timeline."):
                    continue
                try:
                    plan = describe(
                        action.action_id,
                        action.params,
                        object_id=(
                            object_identity.object_id if object_identity is not None else None
                        ),
                        object_type=(
                            object_identity.object_type if object_identity is not None else None
                        ),
                    )
                except Exception:
                    continue
                plans.append(plan)
        return tuple(plans)


def _format_time_label(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def _ready_count_label(count: int) -> str:
    noun = "ready row" if count == 1 else "ready rows"
    return f"{count} {noun}"


def _preview_transfer_plan_label(plan) -> str:
    return f"Preview Transfer Plan ({_ready_count_label(plan.ready_count)})"


def _apply_transfer_plan_label(plan) -> str:
    return f"Apply Transfer Plan ({_ready_count_label(plan.ready_count)})"
