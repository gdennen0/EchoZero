"""Timeline canvas rendering and interaction surface.
Exists to render timeline lanes and translate direct operator input into widget-level signals.
Connects timeline presentation, inspector contracts, and reusable row blocks to the main shell.
"""

from __future__ import annotations

from typing import Final

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.ui.FEEL import (
    EVENT_BAR_HEIGHT_PX,
    LAYER_HEADER_MAX_WIDTH_PX,
    LAYER_HEADER_MIN_WIDTH_PX,
    LAYER_HEADER_RESIZE_HANDLE_HALF_WIDTH_PX,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
)
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock
from echozero.ui.qt.timeline.blocks.layer_header import HeaderSlots, LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLaneBlock
from echozero.ui.qt.timeline.layer_height_config import timeline_layer_height_config
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.time_grid import TimelineGridMode
from echozero.ui.qt.timeline.widget_canvas_interaction_mixin import (
    _TimelineCanvasInteractionMixin,
)
from echozero.ui.qt.timeline.widget_canvas_paint_mixin import (
    _TimelineCanvasPaintMixin,
    badge_tooltip_labels,
)
from echozero.ui.qt.timeline.widget_canvas_types import (
    DrawCandidate as _DrawCandidate,
    EventDragCandidate as _EventDragCandidate,
    EventLaneRect as _EventLaneRect,
    EventRect as _EventRect,
    FixEventRect as _FixEventRect,
    SectionBoundaryRect as _SectionBoundaryRect,
    SectionLabelRect as _SectionLabelRect,
    LayerDragCandidate as _LayerDragCandidate,
    LayerResizeCandidate as _LayerResizeCandidate,
    SelectionDragCandidate as _SelectionDragCandidate,
    TakeActionRect as _TakeActionRect,
    TakeRect as _TakeRect,
)

_FIX_CURSOR_SIZE_PX: Final[int] = 24
_FIX_CURSOR_HOTSPOT_PX: Final[int] = 12
_FIX_CURSOR_STROKE_PX: Final[int] = 2
_FIX_CURSOR_BG_STROKE_HEX: Final[str] = "#d8dde5"
_FIX_CURSOR_PLUS_HEX: Final[str] = "#59d080"
_FIX_CURSOR_MINUS_HEX: Final[str] = "#f07373"
_FIX_CURSOR_SELECT_HEX: Final[str] = "#67b5ff"
_FIX_CURSOR_CACHE: dict[str, QCursor] = {}


def _build_fix_cursor(action: str) -> QCursor:
    cached = _FIX_CURSOR_CACHE.get(action)
    if cached is not None:
        return cached

    pixmap = QPixmap(_FIX_CURSOR_SIZE_PX, _FIX_CURSOR_SIZE_PX)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = float(_FIX_CURSOR_HOTSPOT_PX)
        radius = 7.5
        painter.setPen(QPen(QColor(_FIX_CURSOR_BG_STROKE_HEX), _FIX_CURSOR_STROKE_PX))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(
            int(round(center - radius)),
            int(round(center - radius)),
            int(round(radius * 2.0)),
            int(round(radius * 2.0)),
        )
        accent_hex = {
            "promote": _FIX_CURSOR_PLUS_HEX,
            "remove": _FIX_CURSOR_MINUS_HEX,
            "select": _FIX_CURSOR_SELECT_HEX,
        }.get(action, _FIX_CURSOR_SELECT_HEX)
        accent = QColor(accent_hex)
        painter.setPen(QPen(accent, _FIX_CURSOR_STROKE_PX))
        if action in {"promote", "remove"}:
            painter.drawLine(8, _FIX_CURSOR_HOTSPOT_PX, 16, _FIX_CURSOR_HOTSPOT_PX)
        if action == "promote":
            painter.drawLine(_FIX_CURSOR_HOTSPOT_PX, 8, _FIX_CURSOR_HOTSPOT_PX, 16)
        if action == "select":
            painter.setBrush(accent)
            painter.drawEllipse(_FIX_CURSOR_HOTSPOT_PX - 2, _FIX_CURSOR_HOTSPOT_PX - 2, 4, 4)
    finally:
        painter.end()

    cursor = QCursor(pixmap, _FIX_CURSOR_HOTSPOT_PX, _FIX_CURSOR_HOTSPOT_PX)
    _FIX_CURSOR_CACHE[action] = cursor
    return cursor


class TimelineCanvas(_TimelineCanvasPaintMixin, _TimelineCanvasInteractionMixin, QWidget):
    layer_clicked = pyqtSignal(object, str)
    layer_reorder_requested = pyqtSignal(object, object, bool)
    select_adjacent_layer_requested = pyqtSignal(int)
    mute_clicked = pyqtSignal(object)
    solo_clicked = pyqtSignal(object)
    pipeline_actions_clicked = pyqtSignal(object)
    push_clicked = pyqtSignal(object)
    pull_clicked = pyqtSignal(object)
    section_manager_clicked = pyqtSignal(object)
    take_toggle_clicked = pyqtSignal(object)
    take_selected = pyqtSignal(object, object)
    event_selected = pyqtSignal(object, object, object, str)
    select_adjacent_event_requested = pyqtSignal(int, bool)
    move_selected_events_requested = pyqtSignal(float, object, bool)
    move_selected_events_to_adjacent_layer_requested = pyqtSignal(int)
    take_action_selected = pyqtSignal(object, object, str)
    contract_action_selected = pyqtSignal(object)
    horizontal_scroll_requested = pyqtSignal(float)
    zoom_requested = pyqtSignal(int, float)
    playhead_drag_requested = pyqtSignal(float)
    clear_selection_requested = pyqtSignal()
    select_all_requested = pyqtSignal()
    set_selected_events_requested = pyqtSignal(object, object, object, object, object)
    create_event_requested = pyqtSignal(object, object, float, float)
    delete_events_requested = pyqtSignal(object)
    nudge_requested = pyqtSignal(int, int)
    duplicate_requested = pyqtSignal(int)
    edit_mode_requested = pyqtSignal(str)
    fix_action_requested = pyqtSignal(str)
    fix_nav_include_demoted_toggle_requested = pyqtSignal()
    fix_promote_requested = pyqtSignal(object, object, float, float, str)
    fix_promote_batch_requested = pyqtSignal(object)
    fix_demote_selected_requested = pyqtSignal(object)
    fix_promote_selected_requested = pyqtSignal(object)
    snap_toggle_requested = pyqtSignal()
    grid_mode_cycle_requested = pyqtSignal()
    add_event_at_playhead_requested = pyqtSignal()
    preview_transfer_plan_requested = pyqtSignal()
    apply_transfer_plan_requested = pyqtSignal()
    cancel_transfer_plan_requested = pyqtSignal()
    preview_selected_event_clip_requested = pyqtSignal()
    section_label_double_clicked = pyqtSignal(object)
    section_boundary_double_clicked = pyqtSignal(object)
    header_width_changed = pyqtSignal(int)

    def __init__(
        self,
        presentation: TimelinePresentation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.presentation = presentation
        self._style = TIMELINE_STYLE
        self._layer_height_config = timeline_layer_height_config()
        self._header_width = LAYER_HEADER_WIDTH_PX
        self._header_min_width = int(LAYER_HEADER_MIN_WIDTH_PX)
        self._header_max_width = int(LAYER_HEADER_MAX_WIDTH_PX)
        self._header_resize_handle_half_width = int(LAYER_HEADER_RESIZE_HANDLE_HALF_WIDTH_PX)
        self._top_padding = LAYER_HEADER_TOP_PADDING_PX
        self._main_row_height = self._layer_height_config.default_main_row_height_px
        self._take_row_height = self._layer_height_config.take_row_height_px
        self._main_row_min_height = self._layer_height_config.min_main_row_height_px
        self._main_row_max_height = self._layer_height_config.max_main_row_height_px
        self._resize_handle_hit_padding = self._layer_height_config.resize_handle_hit_padding_px
        self._main_row_height_by_kind = dict(
            self._layer_height_config.layer_kind_main_row_height_px
        )
        self._custom_main_row_height_by_layer: dict[LayerId, int] = {}
        self._event_height = EVENT_BAR_HEIGHT_PX
        self._take_rects: list[_TakeRect] = []
        self._take_option_rects: list[_TakeRect] = []
        self._take_action_rects: list[_TakeActionRect] = []
        self._layer_row_resize_hit_rects: list[tuple[object, LayerId]] = []
        self._open_take_options: set[tuple[LayerId, TakeId]] = set()
        self._toggle_rects: list[tuple[object, LayerId]] = []
        self._mute_rects: list[tuple[object, LayerId]] = []
        self._solo_rects: list[tuple[object, LayerId]] = []
        self._pipeline_action_rects: list[tuple[object, LayerId]] = []
        self._push_rects: list[tuple[object, LayerId]] = []
        self._pull_rects: list[tuple[object, LayerId]] = []
        self._section_manager_rects: list[tuple[object, LayerId]] = []
        self._event_rects: list[_EventRect] = []
        self._section_label_rects: list[_SectionLabelRect] = []
        self._section_boundary_rects: list[_SectionBoundaryRect] = []
        self._fix_event_rects: list[_FixEventRect] = []
        self._focused_fix_overlay_key: tuple[LayerId, TakeId | None, str, float, float] | None = None
        self._event_lane_rects: list[_EventLaneRect] = []
        self._header_select_rects: list[tuple[object, LayerId]] = []
        self._row_body_select_rects: list[tuple[object, LayerId, TakeId | None]] = []
        self._header_hover_rects: list[tuple[object, LayerPresentation]] = []
        self._event_drop_rects: list[tuple[object, LayerId]] = []
        self._layer_drag_candidate: _LayerDragCandidate | None = None
        self._dragging_layer_reorder = False
        self._layer_drag_target_y: float | None = None
        self._hovered_layer_id: LayerId | None = None
        self._dragging_playhead = False
        self._drag_candidate: _EventDragCandidate | None = None
        self._dragging_events = False
        self._layer_row_resize_candidate: _LayerResizeCandidate | None = None
        self._header_resize_candidate: tuple[float, int] | None = None
        self._selection_drag_candidate: _SelectionDragCandidate | None = None
        self._drawing_candidate: _DrawCandidate | None = None
        self._marquee_rect = None
        self._preview_event_rect = None
        self._snap_indicator_time: float | None = None
        self._edit_mode = "select"
        self._fix_action = "select"
        self._fix_nav_include_demoted = False
        self._snap_enabled = True
        self._grid_mode = TimelineGridMode.AUTO.value
        self._suppress_next_context_menu_event = False
        self._showing_context_menu = False
        self._header_block = LayerHeaderBlock()
        self._waveform_block = WaveformLaneBlock()
        self._event_lane_block = EventLaneBlock()
        self._take_row_block = TakeRowBlock()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Keep enough horizontal room for the fixed layer header while still allowing
        # the main shell window to condense on smaller displays.
        self.setMinimumWidth(max(440, self._header_width + 120))
        self.setMouseTracking(True)
        self._recompute_height()
        self._sync_cursor()

    def _header_block_slots_factory(self, layout) -> HeaderSlots:
        return HeaderSlots(
            rect=layout.header_rect,
            title_rect=layout.title_rect,
            subtitle_rect=layout.subtitle_rect,
            status_rect=layout.status_rect,
            controls_rect=layout.controls_rect,
            active_rect=layout.active_rect,
            toggle_rect=layout.toggle_rect,
            metadata_rect=layout.metadata_rect,
        )

    def _recompute_height(self) -> None:
        self._prune_row_height_overrides()
        height = self._top_padding
        for layer in self.presentation.layers:
            height += self._main_row_height_for_layer(layer)
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
        self,
        presentation: TimelinePresentation,
        *,
        recompute_layout: bool = True,
    ) -> None:
        self.presentation = presentation
        self._prune_row_height_overrides()
        if recompute_layout:
            self._recompute_height()
        self.update()

    def set_header_width(self, width: int) -> None:
        clamped = int(max(self._header_min_width, min(self._header_max_width, int(width))))
        if clamped == self._header_width:
            return
        self._header_width = clamped
        self.setMinimumWidth(max(440, self._header_width + 120))
        self.update()

    def _set_header_width_from_drag(self, x: float) -> None:
        if self._header_resize_candidate is None:
            return
        anchor_x, anchor_width = self._header_resize_candidate
        delta = int(round(float(x) - float(anchor_x)))
        self.set_header_width(anchor_width + delta)
        self.header_width_changed.emit(int(self._header_width))

    def set_editor_state(
        self,
        *,
        edit_mode: str,
        fix_action: str,
        fix_nav_include_demoted: bool,
        snap_enabled: bool,
        grid_mode: str,
    ) -> None:
        self._edit_mode = edit_mode
        normalized_fix_action = str(fix_action or "select").strip().lower()
        if normalized_fix_action not in {"promote", "remove", "select"}:
            normalized_fix_action = "select"
        self._fix_action = normalized_fix_action
        self._fix_nav_include_demoted = bool(fix_nav_include_demoted)
        self._snap_enabled = bool(snap_enabled)
        self._grid_mode = grid_mode
        if self._edit_mode != "fix":
            self._focused_fix_overlay_key = None
        self._sync_cursor()
        self.update()

    def _sync_cursor(self) -> None:
        if self._header_resize_candidate is not None:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            return
        if self._layer_row_resize_candidate is not None:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            return
        if self._edit_mode in {"draw", "region"}:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._edit_mode == "fix":
            self.setCursor(_build_fix_cursor(self._fix_action))
        elif self._edit_mode == "erase":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self._edit_mode == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()

    def _default_main_row_height_for_kind(self, kind: LayerKind) -> int:
        return int(self._main_row_height_by_kind.get(kind, self._main_row_height))

    def _main_row_height_for_layer(self, layer: LayerPresentation) -> int:
        return int(
            self._custom_main_row_height_by_layer.get(
                layer.layer_id,
                self._default_main_row_height_for_kind(layer.kind),
            )
        )

    def _main_row_height_for_layer_id(self, layer_id: LayerId) -> int:
        for layer in self.presentation.layers:
            if layer.layer_id == layer_id:
                return self._main_row_height_for_layer(layer)
        return int(self._main_row_height)

    def _set_main_row_height_for_layer(self, layer_id: LayerId, height: int) -> None:
        layer = next(
            (
                candidate
                for candidate in self.presentation.layers
                if candidate.layer_id == layer_id
            ),
            None,
        )
        if layer is None:
            return
        clamped = int(max(self._main_row_min_height, min(self._main_row_max_height, int(height))))
        default_height = self._default_main_row_height_for_kind(layer.kind)
        previous_height = self._main_row_height_for_layer(layer)
        if clamped == default_height:
            self._custom_main_row_height_by_layer.pop(layer.layer_id, None)
        else:
            self._custom_main_row_height_by_layer[layer.layer_id] = clamped
        if self._main_row_height_for_layer(layer) == previous_height:
            return
        self._recompute_height()
        self.update()

    def _prune_row_height_overrides(self) -> None:
        active_layer_ids = {layer.layer_id for layer in self.presentation.layers}
        stale_layer_ids = [
            layer_id
            for layer_id in self._custom_main_row_height_by_layer
            if layer_id not in active_layer_ids
        ]
        for layer_id in stale_layer_ids:
            self._custom_main_row_height_by_layer.pop(layer_id, None)


__all__ = ["TimelineCanvas", "badge_tooltip_labels"]
