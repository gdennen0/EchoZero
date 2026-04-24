"""Timeline canvas interaction helpers.
Exists to keep hit-testing, drag state, and input routing out of the canvas root.
Connects pointer and keyboard input to the widget's signal-based edit contract.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QMenu, QToolTip

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    InspectorContract,
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.models import EventRef
from echozero.ui.FEEL import DRAG_THRESHOLD_PX, SNAP_MAGNETISM_RADIUS_PX
from echozero.ui.qt.timeline.blocks.ruler import (
    playhead_head_polygon,
    seek_time_for_x,
    timeline_x_for_time,
)
from echozero.ui.qt.timeline.time_grid import resolve_snap_time
from echozero.ui.qt.timeline.widget_canvas_types import (
    DrawCandidate,
    EventDragCandidate,
    EventLaneRect,
    SelectionDragCandidate,
)


class _TimelineCanvasInteractionMixin:
    def mouseMoveEvent(self: Any, event: QMouseEvent | None) -> None:
        if event is None:
            return
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
                    1.0,
                    self.presentation.pixels_per_second,
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
        hovered = None
        hovered_rect = None
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

    def leaveEvent(self: Any, event: QEvent | None) -> None:
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

    def mousePressEvent(self: Any, event: QMouseEvent | None) -> None:
        if event is None:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = event.position()

        if event.button() == Qt.MouseButton.RightButton:
            self._suppress_next_context_menu_event = True
            if self._show_context_menu(pos):
                event.accept()
                return
            self._suppress_next_context_menu_event = False

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
        for rect, layer_id, selected_take_id in self._take_rects:
            if rect.contains(pos):
                self.take_selected.emit(layer_id, selected_take_id)
                return
        for rect, layer_id, event_take_id, event_id in self._event_rects:
            if rect.contains(pos):
                if self._edit_mode == "erase":
                    if event_take_id is not None:
                        self.delete_events_requested.emit(
                            [
                                EventRef(
                                    layer_id=layer_id,
                                    take_id=event_take_id,
                                    event_id=event_id,
                                )
                            ]
                        )
                    return
                if (
                    self._edit_mode == "move"
                    and event.button() == Qt.MouseButton.LeftButton
                    and self._can_start_event_drag(
                        event.modifiers(),
                        layer_id,
                        event_take_id,
                        event_id,
                    )
                ):
                    self._drag_candidate = EventDragCandidate(
                        anchor_x=pos.x(),
                        anchor_y=pos.y(),
                        source_layer_id=layer_id,
                        source_take_id=event_take_id,
                        anchor_event_id=event_id,
                        anchor_event_start=self._event_start_for_event(
                            layer_id,
                            event_take_id,
                            event_id,
                        ),
                    )
                    self._dragging_events = False
                    event.accept()
                    return
                self.event_selected.emit(
                    layer_id,
                    event_take_id,
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
                self._drawing_candidate = DrawCandidate(
                    layer_id=lane_layer_id,
                    take_id=lane_take_id,
                    anchor_time=anchor_time,
                )
                self._preview_event_rect = None
                self._update_draw_preview(pos, modifiers=event.modifiers())
                event.accept()
                return
            if self._edit_mode == "select":
                self._selection_drag_candidate = SelectionDragCandidate(
                    anchor_pos=pos,
                    origin_layer_id=lane_layer_id,
                    origin_take_id=lane_take_id,
                    modifiers=event.modifiers(),
                )
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
                    layer_id,
                    self._layer_selection_mode_for_modifiers(event.modifiers()),
                )
                return
        for rect, layer_id in self._row_body_select_rects:
            if rect.contains(pos):
                self.layer_clicked.emit(
                    layer_id,
                    self._layer_selection_mode_for_modifiers(event.modifiers()),
                )
                return
        super().mousePressEvent(event)

    def contextMenuEvent(self: Any, event: QContextMenuEvent | None) -> None:
        if event is None:
            return
        if self._suppress_next_context_menu_event:
            self._suppress_next_context_menu_event = False
            event.ignore()
            return
        local_pos = QPointF(event.pos())
        if not self._show_context_menu(local_pos, global_pos=event.globalPos()):
            event.ignore()
            return
        event.accept()

    def _show_context_menu(self: Any, pos: QPointF, *, global_pos: QPoint | None = None) -> bool:
        if self._showing_context_menu:
            return False
        hit_target = self._hit_target_for_position(pos)
        contract = build_timeline_inspector_contract(self.presentation, hit_target=hit_target)
        menu = self._build_context_menu(contract, hit_kind=hit_target.kind)
        if menu.isEmpty():
            return False
        if global_pos is None:
            global_pos = self.mapToGlobal(pos.toPoint())
        self._showing_context_menu = True
        try:
            chosen = menu.exec(global_pos)
        finally:
            menu.hide()
            self._showing_context_menu = False
        if chosen is None:
            return True
        payload = chosen.data()
        if isinstance(payload, InspectorAction):
            QTimer.singleShot(
                0,
                lambda payload=payload: self.contract_action_selected.emit(payload),
            )
        return True

    def _build_context_menu(
        self: Any,
        contract: InspectorContract,
        *,
        hit_kind: str | None = None,
    ) -> QMenu:
        menu = QMenu(self)
        first_section = True
        seen_action_ids: set[str] = set()
        for section in contract.context_sections:
            visible_actions: list[InspectorAction] = []
            for action in section.actions:
                if hit_kind is not None and not self._context_action_visible_for_hit_kind(
                    action,
                    hit_kind,
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
                if qt_action is None:
                    continue
                qt_action.setEnabled(action.enabled)
                qt_action.setData(action)
        return menu

    @staticmethod
    def _context_action_visible_for_hit_kind(action: InspectorAction, hit_kind: str) -> bool:
        group = (action.group or "").strip().lower()
        kind = (hit_kind or "").strip().lower()
        if kind == "event" and group == "transfer":
            return action.action_id == "send_selected_events_to_ma3"

        allowed_groups_by_kind = {
            "timeline": {"tools", "transport"},
            "layer": {"batch", "layer", "gain", "pipeline", "transfer", "live_sync", "transport"},
            "take": {"batch", "take", "transport"},
            "event": {"batch", "selection", "take", "transport"},
        }
        allowed = allowed_groups_by_kind.get(kind)
        if allowed is None:
            return True
        return group in allowed

    def mouseReleaseEvent(self: Any, event: QMouseEvent | None) -> None:
        if event is None:
            return
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
                            float(delta_seconds),
                            target_layer_id,
                        )
                    event.accept()
                self._drag_candidate = None
                self._dragging_events = False
                self._snap_indicator_time = None
                self.update()
                if event.isAccepted():
                    return
        super().mouseReleaseEvent(event)

    def wheelEvent(self: Any, event: QWheelEvent | None) -> None:
        if event is None:
            return
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

    def keyPressEvent(self: Any, event: QKeyEvent | None) -> None:
        if event is None:
            return
        modifiers = event.modifiers()
        has_primary = bool(
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        )
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        steps = 10 if has_shift else 1
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
            self.delete_events_requested.emit(
                list(self.presentation.selected_event_refs)
                if self.presentation.selected_event_refs
                else list(self.presentation.selected_event_ids)
            )
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
        if not has_primary and not has_shift and event.key() == Qt.Key.Key_M:
            self.edit_mode_requested.emit("move")
            event.accept()
            return
        if not has_primary and not has_shift and event.key() == Qt.Key.Key_R:
            self.edit_mode_requested.emit("region")
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
        if not has_primary and self._handle_mode_specific_key_press(event, steps=steps):
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

    def _handle_mode_specific_key_press(self: Any, event: QKeyEvent, *, steps: int) -> bool:
        if self._edit_mode == "select":
            if event.key() in (Qt.Key.Key_Comma, Qt.Key.Key_Less):
                self.select_adjacent_event_requested.emit(-1)
                return True
            if event.key() in (Qt.Key.Key_Period, Qt.Key.Key_Greater):
                self.select_adjacent_event_requested.emit(1)
                return True
            if event.key() == Qt.Key.Key_Up:
                self.select_adjacent_layer_requested.emit(-1)
                return True
            if event.key() == Qt.Key.Key_Down:
                self.select_adjacent_layer_requested.emit(1)
                return True
            return False

        if self._edit_mode == "move":
            if event.key() == Qt.Key.Key_Left:
                self.nudge_requested.emit(-1, steps)
                return True
            if event.key() == Qt.Key.Key_Right:
                self.nudge_requested.emit(1, steps)
                return True
            if event.key() == Qt.Key.Key_Up:
                self.move_selected_events_to_adjacent_layer_requested.emit(-1)
                return True
            if event.key() == Qt.Key.Key_Down:
                self.move_selected_events_to_adjacent_layer_requested.emit(1)
                return True
        return False

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

    def _event_lane_hit(self: Any, pos: QPointF) -> EventLaneRect | None:
        for rect, layer_id, take_id in self._event_lane_rects:
            if rect.contains(pos):
                return rect, layer_id, take_id
        return None

    def _resolve_draw_time(self: Any, x: float, *, modifiers: Qt.KeyboardModifier) -> float:
        time_seconds = self._seek_time_at_x(x)
        snapped = self._resolve_snap_target_time(
            time_seconds,
            modifiers=modifiers,
            exclude_event_ids=(),
        )
        return snapped if snapped is not None else time_seconds

    def _event_start_for_event(
        self: Any,
        layer_id: LayerId,
        take_id: TakeId | None,
        event_id: EventId,
    ) -> float:
        for layer in self.presentation.layers:
            if layer.layer_id != layer_id:
                continue
            for event in layer.events:
                if layer.main_take_id == take_id and event.event_id == event_id:
                    return float(event.start)
            for take in layer.takes:
                if take.take_id != take_id:
                    continue
                for event in take.events:
                    if event.event_id == event_id:
                        return float(event.start)
        return 0.0

    def _event_times(self: Any, *, exclude_event_ids: tuple[EventId, ...]) -> tuple[float, ...]:
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
        self: Any,
        time_seconds: float,
        *,
        modifiers: Qt.KeyboardModifier,
        exclude_event_ids: tuple[EventId, ...],
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

    def _update_draw_preview(self: Any, pos: QPointF, *, modifiers: Qt.KeyboardModifier) -> None:
        if self._drawing_candidate is None:
            return
        anchor_time = float(self._drawing_candidate["anchor_time"])
        current_time = self._resolve_draw_time(pos.x(), modifiers=modifiers)
        if abs(current_time - anchor_time) < 1e-6:
            current_time = anchor_time + 0.25
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

    def _commit_draw_preview(self: Any, pos: QPointF, *, modifiers: Qt.KeyboardModifier) -> None:
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

    def _commit_selection_drag(self: Any) -> None:
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
        intersected_refs: list[EventRef] = []
        for event_rect, layer_id, take_id, event_id in self._event_rects:
            if rect.intersects(event_rect) and take_id is not None:
                intersected_refs.append(
                    EventRef(layer_id=layer_id, take_id=take_id, event_id=event_id)
                )

        next_event_refs = list(self.presentation.selected_event_refs)
        if mode == "replace":
            next_event_refs = intersected_refs
        elif mode == "additive":
            for event_ref in intersected_refs:
                if event_ref not in next_event_refs:
                    next_event_refs.append(event_ref)
        else:
            for event_ref in intersected_refs:
                if event_ref in next_event_refs:
                    next_event_refs = [
                        candidate_ref
                        for candidate_ref in next_event_refs
                        if candidate_ref != event_ref
                    ]
                else:
                    next_event_refs.append(event_ref)

        next_event_ids = [event_ref.event_id for event_ref in next_event_refs]
        anchor_layer_id, anchor_take_id, selected_layer_ids = (
            self._selection_context_for_event_refs(next_event_refs)
        )
        self.set_selected_events_requested.emit(
            next_event_ids,
            next_event_refs,
            anchor_layer_id,
            anchor_take_id,
            selected_layer_ids,
        )
        self.update()

    def _selection_context_for_event_refs(
        self: Any,
        event_refs: list[EventRef],
    ) -> tuple[LayerId | None, TakeId | None, list[LayerId]]:
        selected_layer_ids: list[LayerId] = []
        anchor_layer_id: LayerId | None = None
        anchor_take_id: TakeId | None = None
        for event_ref in event_refs:
            anchor_layer_id = event_ref.layer_id
            anchor_take_id = event_ref.take_id
            if event_ref.layer_id not in selected_layer_ids:
                selected_layer_ids.append(event_ref.layer_id)
        return anchor_layer_id, anchor_take_id, selected_layer_ids

    def _playhead_head_contains(self: Any, pos: QPointF) -> bool:
        x = timeline_x_for_time(
            self.presentation.playhead,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        return playhead_head_polygon(x, float(self._top_padding)).boundingRect().contains(pos)

    def _seek_time_at_x(self: Any, x: float) -> float:
        return seek_time_for_x(
            x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )

    def _can_start_event_drag(
        self: Any,
        modifiers: Qt.KeyboardModifier,
        layer_id: LayerId,
        take_id: TakeId | None,
        event_id: EventId,
    ) -> bool:
        if modifiers & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False
        if take_id is None:
            return event_id in set(self.presentation.selected_event_ids)
        event_ref = EventRef(layer_id=layer_id, take_id=take_id, event_id=event_id)
        if self.presentation.selected_event_refs:
            return event_ref in set(self.presentation.selected_event_refs)
        return event_id in set(self.presentation.selected_event_ids)

    def _event_drop_target_layer_id(self: Any, pos: QPointF) -> LayerId | None:
        for rect, layer_id in self._event_drop_rects:
            if rect.contains(pos):
                return layer_id
        return None

    def _hit_target_for_position(self: Any, pos: QPointF) -> TimelineInspectorHitTarget:
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
