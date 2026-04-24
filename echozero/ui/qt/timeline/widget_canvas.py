"""Timeline canvas rendering and interaction surface.
Exists to render timeline lanes and translate direct operator input into widget-level signals.
Connects timeline presentation, inspector contracts, and reusable row blocks to the main shell.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.ui.FEEL import (
    EVENT_BAR_HEIGHT_PX,
    LAYER_HEADER_TOP_PADDING_PX,
    LAYER_HEADER_WIDTH_PX,
    LAYER_ROW_HEIGHT_PX,
    TAKE_ROW_HEIGHT_PX,
)
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock
from echozero.ui.qt.timeline.blocks.layer_header import HeaderSlots, LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLaneBlock
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
    SelectionDragCandidate as _SelectionDragCandidate,
    TakeActionRect as _TakeActionRect,
    TakeRect as _TakeRect,
)


class TimelineCanvas(_TimelineCanvasPaintMixin, _TimelineCanvasInteractionMixin, QWidget):
    layer_clicked = pyqtSignal(object, str)
    select_adjacent_layer_requested = pyqtSignal(int)
    active_clicked = pyqtSignal(object)
    pipeline_actions_clicked = pyqtSignal(object)
    push_clicked = pyqtSignal(object)
    pull_clicked = pyqtSignal(object)
    take_toggle_clicked = pyqtSignal(object)
    take_selected = pyqtSignal(object, object)
    event_selected = pyqtSignal(object, object, object, str)
    select_adjacent_event_requested = pyqtSignal(int)
    move_selected_events_requested = pyqtSignal(float, object)
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
    snap_toggle_requested = pyqtSignal()
    grid_mode_cycle_requested = pyqtSignal()
    preview_transfer_plan_requested = pyqtSignal()
    apply_transfer_plan_requested = pyqtSignal()
    cancel_transfer_plan_requested = pyqtSignal()

    def __init__(
        self,
        presentation: TimelinePresentation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.presentation = presentation
        self._style = TIMELINE_STYLE
        self._header_width = LAYER_HEADER_WIDTH_PX
        self._top_padding = LAYER_HEADER_TOP_PADDING_PX
        self._main_row_height = LAYER_ROW_HEIGHT_PX
        self._take_row_height = TAKE_ROW_HEIGHT_PX
        self._event_height = EVENT_BAR_HEIGHT_PX
        self._take_rects: list[_TakeRect] = []
        self._take_option_rects: list[_TakeRect] = []
        self._take_action_rects: list[_TakeActionRect] = []
        self._open_take_options: set[tuple[LayerId, TakeId]] = set()
        self._toggle_rects: list[tuple[object, LayerId]] = []
        self._active_rects: list[tuple[object, LayerId]] = []
        self._pipeline_action_rects: list[tuple[object, LayerId]] = []
        self._push_rects: list[tuple[object, LayerId]] = []
        self._pull_rects: list[tuple[object, LayerId]] = []
        self._event_rects: list[_EventRect] = []
        self._event_lane_rects: list[_EventLaneRect] = []
        self._header_select_rects: list[tuple[object, LayerId]] = []
        self._row_body_select_rects: list[tuple[object, LayerId]] = []
        self._header_hover_rects: list[tuple[object, LayerPresentation]] = []
        self._event_drop_rects: list[tuple[object, LayerId]] = []
        self._hovered_layer_id: LayerId | None = None
        self._dragging_playhead = False
        self._drag_candidate: _EventDragCandidate | None = None
        self._dragging_events = False
        self._selection_drag_candidate: _SelectionDragCandidate | None = None
        self._drawing_candidate: _DrawCandidate | None = None
        self._marquee_rect = None
        self._preview_event_rect = None
        self._snap_indicator_time: float | None = None
        self._edit_mode = "select"
        self._snap_enabled = True
        self._grid_mode = TimelineGridMode.AUTO.value
        self._suppress_next_context_menu_event = False
        self._showing_context_menu = False
        self._header_block = LayerHeaderBlock()
        self._waveform_block = WaveformLaneBlock()
        self._event_lane_block = EventLaneBlock()
        self._take_row_block = TakeRowBlock()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(1440)
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
            toggle_rect=layout.toggle_rect,
            metadata_rect=layout.metadata_rect,
        )

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
        self,
        presentation: TimelinePresentation,
        *,
        recompute_layout: bool = True,
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
        if self._edit_mode in {"draw", "region"}:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._edit_mode == "erase":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self._edit_mode == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()


__all__ = ["TimelineCanvas", "badge_tooltip_labels"]
