# input.py — Mouse/keyboard input handling for EchoZero Timeline Canvas

from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QCursor
import math

from FEEL import (
    ZOOM_MIN, ZOOM_MAX, ZOOM_STEP,
    SNAP_THRESHOLD_PX, SNAP_GRID_SECONDS,
    LAYERS_PANEL_WIDTH
)
from model import TimelineState, TimelineEvent


class InputHandler:
    """
    Handles all mouse/keyboard events for the timeline canvas.
    Calls back into canvas via the interface methods.
    """

    def __init__(self, canvas):
        self._canvas = canvas
        self._pan_active = False
        self._pan_start_pos: QPointF = None
        self._pan_start_scroll_x: float = 0.0
        self._pan_start_scroll_y: float = 0.0

        self._drag_active = False
        self._drag_event_id: str = None
        self._drag_start_pos: QPointF = None
        self._drag_original_time: float = 0.0
        self._drag_original_layer: str = None
        self._snap_indicator_x: int = None   # pixel x for snap indicator, or None

        self._resize_active = False
        self._resize_event_id: str = None
        self._resize_start_pos: QPointF = None
        self._resize_original_duration: float = 0.0

    # ─── Coordinate helpers ──────────────────────────────────────────────────

    def _x_to_time(self, x: float) -> float:
        state = self._canvas.state
        return state.scroll_x + (x - LAYERS_PANEL_WIDTH) / state.zoom_level

    def _time_to_x(self, t: float) -> int:
        state = self._canvas.state
        return int(LAYERS_PANEL_WIDTH + (t - state.scroll_x) * state.zoom_level)

    def _y_to_layer_idx(self, y: float) -> int:
        state = self._canvas.state
        from FEEL import RULER_HEIGHT
        py = RULER_HEIGHT - state.scroll_y
        for i, layer in enumerate(state.layers):
            if py <= y < py + layer.height:
                return i
            py += layer.height
        return -1

    def _snap_time(self, t: float) -> tuple:
        """Snap t to grid. Returns (snapped_t, did_snap)."""
        grid = SNAP_GRID_SECONDS
        snapped = round(t / grid) * grid
        diff_px = abs(self._time_to_x(snapped) - self._time_to_x(t))
        if diff_px <= SNAP_THRESHOLD_PX:
            return snapped, True
        return t, False

    # ─── Event hit testing ───────────────────────────────────────────────────

    def _hit_event(self, x: float, y: float):
        """Return (event, edge) where edge is 'left', 'right', or None."""
        canvas = self._canvas
        state = canvas.state
        from FEEL import RULER_HEIGHT, EVENT_VERTICAL_PADDING

        t = self._x_to_time(x)
        layer_idx = self._y_to_layer_idx(y)
        if layer_idx < 0:
            return None, None

        layer = state.layers[layer_idx]
        edge_threshold_t = 6.0 / state.zoom_level  # 6px in time units

        # Check events on this layer
        for ev in state.events:
            if ev.layer_id != layer.id:
                continue
            ev_end = ev.time + ev.duration
            if t < ev.time - edge_threshold_t or t > ev_end + edge_threshold_t:
                continue

            ev_x1 = self._time_to_x(ev.time)
            ev_x2 = self._time_to_x(ev_end)

            # Check if y is within event row
            py = RULER_HEIGHT - state.scroll_y
            for i, lyr in enumerate(state.layers):
                if lyr.id == layer.id:
                    ev_y1 = py + EVENT_VERTICAL_PADDING
                    ev_y2 = py + lyr.height - EVENT_VERTICAL_PADDING
                    if not (ev_y1 <= y <= ev_y2):
                        continue
                    break
                py += lyr.height

            if ev_x1 <= x <= ev_x2:
                # Determine edge
                edge = None
                if x - ev_x1 < 6:
                    edge = 'left'
                elif ev_x2 - x < 6:
                    edge = 'right'
                return ev, edge

        return None, None

    # ─── Wheel (zoom) ────────────────────────────────────────────────────────

    def wheel_event(self, event):
        state = self._canvas.state
        delta = event.angleDelta().y()
        if delta == 0:
            return

        # Cursor-anchored zoom
        cursor_x = event.x()
        t_under_cursor = self._x_to_time(cursor_x)

        factor = 1.0 + ZOOM_STEP * (delta / 120.0)
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, state.zoom_level * factor))
        state.zoom_level = new_zoom

        # Adjust scroll_x so t_under_cursor stays under cursor
        state.scroll_x = t_under_cursor - (cursor_x - LAYERS_PANEL_WIDTH) / new_zoom
        state.scroll_x = max(0.0, state.scroll_x)

        self._canvas.update()
        self._canvas._ruler.set_zoom(new_zoom)
        self._canvas._ruler.set_scroll_x(state.scroll_x)

    # ─── Mouse Press ─────────────────────────────────────────────────────────

    def mouse_press(self, event):
        state = self._canvas.state
        pos = event.pos()
        btn = event.button()
        mods = event.modifiers()

        # Middle click or Alt+Left → pan
        if btn == Qt.MiddleButton or \
           (btn == Qt.LeftButton and mods & Qt.AltModifier):
            self._pan_active = True
            self._pan_start_pos = pos
            self._pan_start_scroll_x = state.scroll_x
            self._pan_start_scroll_y = state.scroll_y
            self._canvas.setCursor(Qt.ClosedHandCursor)
            return

        if btn == Qt.LeftButton:
            ev, edge = self._hit_event(pos.x(), pos.y())

            if ev is not None:
                # Selection
                if mods & Qt.ShiftModifier:
                    # Add to selection
                    state.selection.add(ev.id)
                else:
                    if ev.id not in state.selection:
                        state.selection = {ev.id}

                if edge in ('left', 'right'):
                    # Resize
                    self._resize_active = True
                    self._resize_event_id = ev.id
                    self._resize_edge = edge
                    self._resize_start_pos = pos
                    self._resize_original_time = ev.time
                    self._resize_original_duration = ev.duration
                else:
                    # Drag/move
                    self._drag_active = True
                    self._drag_event_id = ev.id
                    self._drag_start_pos = pos
                    self._drag_original_time = ev.time

            else:
                # Click on empty → deselect
                if not (mods & Qt.ShiftModifier):
                    state.selection = set()

            self._canvas.update()
            self._canvas._layers_panel.update()

    # ─── Mouse Move ──────────────────────────────────────────────────────────

    def mouse_move(self, event):
        state = self._canvas.state
        pos = event.pos()

        if self._pan_active:
            dx = pos.x() - self._pan_start_pos.x()
            dy = pos.y() - self._pan_start_pos.y()
            new_scroll_x = max(0.0, self._pan_start_scroll_x - dx / state.zoom_level)
            new_scroll_y = max(0.0, self._pan_start_scroll_y - dy)
            state.scroll_x = new_scroll_x
            state.scroll_y = new_scroll_y
            self._canvas._ruler.set_scroll_x(state.scroll_x)
            self._canvas._layers_panel.update()
            self._canvas.update()
            return

        if self._drag_active and self._drag_event_id:
            ev = state.get_event(self._drag_event_id)
            if ev:
                dx = pos.x() - self._drag_start_pos.x()
                dt = dx / state.zoom_level
                new_time = max(0.0, self._drag_original_time + dt)
                snapped_time, did_snap = self._snap_time(new_time)
                ev.time = snapped_time
                if did_snap:
                    self._snap_indicator_x = self._time_to_x(snapped_time)
                else:
                    self._snap_indicator_x = None
            self._canvas.update()
            return

        if self._resize_active and self._resize_event_id:
            ev = state.get_event(self._resize_event_id)
            if ev:
                dx = pos.x() - self._resize_start_pos.x()
                dt = dx / state.zoom_level
                if self._resize_edge == 'right':
                    new_dur = max(0.05, self._resize_original_duration + dt)
                    ev.duration = new_dur
                else:
                    new_time = max(0.0, self._resize_original_time + dt)
                    new_dur = max(0.05, self._resize_original_duration - dt)
                    ev.time = new_time
                    ev.duration = new_dur
            self._canvas.update()
            return

        # Cursor shape on hover
        ev, edge = self._hit_event(pos.x(), pos.y())
        if edge in ('left', 'right'):
            self._canvas.setCursor(Qt.SizeHorCursor)
        elif ev is not None:
            self._canvas.setCursor(Qt.OpenHandCursor)
        else:
            self._canvas.setCursor(Qt.ArrowCursor)

    # ─── Mouse Release ───────────────────────────────────────────────────────

    def mouse_release(self, event):
        if self._pan_active:
            self._pan_active = False
            self._canvas.setCursor(Qt.ArrowCursor)

        if self._drag_active:
            self._drag_active = False
            self._drag_event_id = None
            self._snap_indicator_x = None
            # Events are kept sorted internally by model.visible_events()
            self._canvas.update()

        if self._resize_active:
            self._resize_active = False
            self._resize_event_id = None
            self._canvas.update()

    # ─── Key Press ───────────────────────────────────────────────────────────

    def key_press(self, event):
        state = self._canvas.state
        key = event.key()

        if key == Qt.Key_Escape:
            state.selection = set()
            self._canvas.update()

        elif key == Qt.Key_Delete or key == Qt.Key_Backspace:
            if state.selection:
                state.events = [e for e in state.events if e.id not in state.selection]
                state.selection = set()
                state.rebuild_index()
                self._canvas.update()
