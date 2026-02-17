"""
Movement Controller
====================

Handles all event drag/move operations with proper state management.

This controller centralizes movement logic to ensure:
- Atomic multi-select operations
- Proper state transitions
- Consistent coordinate mapping
- Clean signal emission

The controller uses a state machine pattern to manage drag operations.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from PyQt6.QtCore import QObject, pyqtSignal, QPointF

from ..types import (
    DragState, EditHandle,
    EventMoveResult, EventResizeResult
)
from ..logging import TimelineLog as Log


if TYPE_CHECKING:
    from .layer_manager import LayerManager
    from .items import BaseEventItem
    from ..core.scene import TimelineScene

# Minimum distance in pixels before drag starts (prevents accidental moves)
DRAG_THRESHOLD = 3

@dataclass
class ItemDragState:
    """State of a single item being dragged."""
    item: 'BaseEventItem'
    original_time: float
    original_layer_id: str
    time_offset: float  # Offset from primary item
    layer_offset: int   # Layer index offset from primary item
    
    # Current preview position (not committed yet)
    preview_time: float = 0.0
    preview_layer_id: str = ""

class MovementController(QObject):
    """
    Controls all event movement operations.
    
    Centralizes drag logic that was previously scattered across
    event items, scene, and widget.
    
    State Machine:
        IDLE -> (mouse_down) -> PENDING -> (threshold) -> DRAGGING -> (release) -> IDLE
                                  |                          |
                                  v                          v
                            (release w/o move)          (emit signals)
                                  |                          |
                                  v                          v
                                IDLE                       IDLE
    
    Signals:
        moves_completed(List[EventMoveResult]): Emitted when move is committed
        resizes_completed(List[EventResizeResult]): Emitted when resize is committed
        drag_started(): Emitted when drag begins (for visual feedback)
        drag_ended(): Emitted when drag ends (for visual feedback)
        drag_cancelled(): Emitted when drag is cancelled (Escape key)
    """
    
    moves_completed = pyqtSignal(list)    # List[EventMoveResult]
    resizes_completed = pyqtSignal(list)  # List[EventResizeResult]
    drag_started = pyqtSignal()
    drag_ended = pyqtSignal()
    drag_cancelled = pyqtSignal()
    status_message = pyqtSignal(str, bool)  # message, is_error
    
    def __init__(self, scene: 'TimelineScene', layer_manager: 'LayerManager', parent=None):
        super().__init__(parent)
        
        self._scene = scene
        self._layer_manager = layer_manager
        
        # State machine
        self._state = DragState.IDLE
        self._edit_handle = EditHandle.NONE
        
        # Drag context
        self._primary_item: Optional['BaseEventItem'] = None
        self._start_pos: Optional[QPointF] = None
        self._items: Dict[str, ItemDragState] = {}
        
        # For resize operations
        self._resize_original_time: float = 0.0
        self._resize_original_duration: float = 0.0
        
        # Native snapping mode (POC-verified infrastructure)
        # When True, snapping is done in item's itemChange() instead of here.
        # Default False for compatibility - proven multi-select flow continues to work.
        # Can be enabled incrementally for testing native snapping.
        self._use_native_snapping = False
    
    @property
    def is_dragging(self) -> bool:
        """Whether a drag operation is in progress."""
        return self._state in (DragState.PENDING, DragState.DRAGGING)
    
    @property
    def drag_target_layer_id(self) -> Optional[str]:
        """Layer ID being dragged to (for visual feedback)."""
        if not self.is_dragging or not self._primary_item:
            return None
        
        primary_state = self._items.get(self._primary_item.event_id)
        return primary_state.preview_layer_id if primary_state else None
    
    @property
    def use_native_snapping(self) -> bool:
        """Whether native itemChange() snapping is enabled."""
        return self._use_native_snapping
    
    def set_native_snapping(self, enabled: bool):
        """
        Enable/disable native snapping mode (POC-verified).
        
        When enabled:
        - Item's itemChange() handles snapping
        - Controller skips snap calculation
        - Smoother, more responsive feel
        
        When disabled (default):
        - Controller handles snapping (proven multi-select flow)
        - Better compatibility with existing code
        
        Args:
            enabled: Whether to use native itemChange() snapping
        """
        self._use_native_snapping = enabled
    
    # =========================================================================
    # Public API - Called by Event Items
    # =========================================================================
    
    def begin_move(self, item: 'BaseEventItem', pos: QPointF, selected_items: List['BaseEventItem']) -> None:
        """
        Begin a move operation.
        
        Called by event item on mouse press.
        
        Args:
            item: The primary item being dragged
            pos: Mouse position in scene coordinates
            selected_items: All selected items to move together
        """
        if self._state != DragState.IDLE:
            Log.warning("MovementController: begin_move called while not IDLE")
            return
        
        self._state = DragState.PENDING
        self._edit_handle = EditHandle.MOVE
        self._primary_item = item
        self._start_pos = pos
        
        # Capture state for all items being moved
        self._items.clear()
        
        primary_layer_index = self._layer_manager.get_layer_index(item.layer_id)
        
        for selected_item in selected_items:
            if not hasattr(selected_item, 'event_id'):
                continue
            
            layer_index = self._layer_manager.get_layer_index(selected_item.layer_id)
            
            self._items[selected_item.event_id] = ItemDragState(
                item=selected_item,
                original_time=selected_item.start_time,
                original_layer_id=selected_item.layer_id,
                time_offset=selected_item.start_time - item.start_time,
                layer_offset=layer_index - primary_layer_index,
                preview_time=selected_item.start_time,
                preview_layer_id=selected_item.layer_id,
            )
        
        Log.debug(f"MovementController: begin_move for {len(self._items)} items")
    
    def begin_resize(self, item: 'BaseEventItem', pos: QPointF, handle: EditHandle) -> None:
        """
        Begin a resize operation.
        
        Called by event item on mouse press on resize handle.
        
        Args:
            item: The item being resized
            pos: Mouse position in scene coordinates
            handle: Which resize handle (LEFT or RIGHT)
        """
        if self._state != DragState.IDLE:
            return
        
        if handle not in (EditHandle.RESIZE_LEFT, EditHandle.RESIZE_RIGHT):
            return
        
        self._state = DragState.PENDING
        self._edit_handle = handle
        self._primary_item = item
        self._start_pos = pos
        
        # Capture original state
        self._resize_original_time = item.start_time
        self._resize_original_duration = item.duration
        
        # Only the primary item is resized (no multi-select resize)
        self._items.clear()
        self._items[item.event_id] = ItemDragState(
            item=item,
            original_time=item.start_time,
            original_layer_id=item.layer_id,
            time_offset=0,
            layer_offset=0,
            preview_time=item.start_time,
            preview_layer_id=item.layer_id,
        )
        
        Log.debug(f"MovementController: begin_resize {handle.name}")
    
    def update_drag(self, pos: QPointF, snap_enabled: bool = False, snap_time_func=None, snap_calculator=None, unit_preference=None) -> None:
        """
        Update drag with new mouse position.
        
        Called by event item on mouse move.
        
        Args:
            pos: Current mouse position in scene coordinates
            snap_enabled: Whether to snap to grid
            snap_time_func: Function to snap time values (legacy)
            snap_calculator: SnapCalculator instance (new system)
            unit_preference: UnitPreference for snapping (new system)
        """
        if self._state == DragState.IDLE:
            return
        
        # Check drag threshold
        if self._state == DragState.PENDING:
            if self._start_pos:
                delta = pos - self._start_pos
                if delta.manhattanLength() < DRAG_THRESHOLD:
                    return
            
            # Threshold exceeded - start actual drag
            self._state = DragState.DRAGGING
            self.drag_started.emit()
            Log.debug("MovementController: drag threshold exceeded, DRAGGING")
        
        if self._edit_handle == EditHandle.MOVE:
            self._update_move(pos, snap_enabled, snap_time_func, snap_calculator, unit_preference)
        elif self._edit_handle in (EditHandle.RESIZE_LEFT, EditHandle.RESIZE_RIGHT):
            self._update_resize(pos, snap_enabled, snap_time_func, snap_calculator, unit_preference)
    
    def commit_drag(self) -> None:
        """
        Commit the current drag operation.
        
        Called by event item on mouse release.
        """
        if self._state == DragState.IDLE:
            return
        
        was_dragging = self._state == DragState.DRAGGING
        self._state = DragState.COMMITTING
        
        if was_dragging:
            if self._edit_handle == EditHandle.MOVE:
                self._commit_move()
            elif self._edit_handle in (EditHandle.RESIZE_LEFT, EditHandle.RESIZE_RIGHT):
                self._commit_resize()
        
        # Emit BEFORE cleanup so handlers can access _items
        self.drag_ended.emit()
        self._cleanup()
    
    def cancel_drag(self) -> None:
        """
        Cancel the current drag operation, restoring original positions.
        
        Called when Escape is pressed during drag.
        """
        if self._state == DragState.IDLE:
            return
        
        Log.debug("MovementController: cancelling drag")
        
        # Restore original positions
        for item_state in self._items.values():
            item = item_state.item
            item.start_time = item_state.original_time
            item.layer_id = item_state.original_layer_id
            item._update_geometry()
        
        # Restore resize state if needed
        if self._edit_handle in (EditHandle.RESIZE_LEFT, EditHandle.RESIZE_RIGHT):
            if self._primary_item:
                self._primary_item.start_time = self._resize_original_time
                self._primary_item.duration = self._resize_original_duration
                self._primary_item._update_geometry()
        
        # Emit BEFORE cleanup so handlers can access _items
        self.drag_cancelled.emit()
        self._cleanup()
    
    # =========================================================================
    # Internal - Move Operations
    # =========================================================================
    
    def _update_move(self, pos: QPointF, snap_enabled: bool, snap_time_func, snap_calculator=None, unit_preference=None) -> None:
        """Update positions during move drag.
        
        Args:
            pos: Current mouse position
            snap_enabled: Whether snapping is enabled
            snap_time_func: Legacy snap function
            snap_calculator: SnapCalculator instance (new system)
            unit_preference: UnitPreference for snapping (new system)
        """
        if not self._primary_item or not self._start_pos:
            return
        
        # Calculate time delta from mouse movement
        delta_x = pos.x() - self._start_pos.x()
        pps = self._primary_item._pixels_per_second
        delta_time = delta_x / pps
        
        # Calculate primary item's new time
        primary_state = self._items.get(self._primary_item.event_id)
        if not primary_state:
            return
        
        new_time = primary_state.original_time + delta_time
        new_time = max(0, new_time)
        
        # Apply snapping (unless native snapping is enabled in items)
        if snap_enabled and not self._use_native_snapping:
            if snap_calculator and unit_preference is not None:
                # Use new SnapCalculator
                new_time = snap_calculator.snap_time(new_time, pps, unit_preference)
            elif snap_time_func:
                # Fallback to legacy function
                new_time = snap_time_func(new_time, pps)
        # Note: When _use_native_snapping is True, snapping happens in item's itemChange()
        
        # Calculate new layer from Y position
        new_layer_id = self._layer_manager.get_layer_id_from_y(pos.y())
        if not new_layer_id:
            new_layer_id = primary_state.original_layer_id
        
        # Check if layer is locked
        layer = self._layer_manager.get_layer(new_layer_id)
        if layer and layer.locked:
            self.status_message.emit(f"Layer '{layer.name}' is locked", True)
            new_layer_id = primary_state.original_layer_id
        
        # Calculate primary's new layer index
        new_layer_index = self._layer_manager.get_layer_index(new_layer_id)
        
        # Update all dragged items
        for event_id, item_state in self._items.items():
            item = item_state.item
            if not item or not item.scene():
                continue
            
            # Calculate this item's position based on offset from primary
            item_new_time = new_time + item_state.time_offset
            item_new_time = max(0, item_new_time)
            
            # Calculate layer (clamped to valid range)
            item_layer_index = new_layer_index + item_state.layer_offset
            item_layer_index = max(0, min(item_layer_index, self._layer_manager.get_layer_count() - 1))
            
            item_layer = self._layer_manager.get_layer_at_index(item_layer_index)
            item_layer_id = item_layer.id if item_layer else new_layer_id
            
            # Check lock for this item's target layer
            if item_layer and item_layer.locked:
                item_layer_id = item_state.original_layer_id
            
            # Update preview state
            item_state.preview_time = item_new_time
            item_state.preview_layer_id = item_layer_id
            
            # Update item directly
            item.start_time = item_new_time
            item.layer_id = item_layer_id
            item._update_geometry()
        
        # Let Qt handle scene invalidation
        self._scene.update()
    
    def _commit_move(self) -> None:
        """Commit move operation and emit results."""
        results = []
        
        for event_id, item_state in self._items.items():
            # Check if anything changed
            time_changed = item_state.preview_time != item_state.original_time
            layer_changed = item_state.preview_layer_id != item_state.original_layer_id
            
            if time_changed or layer_changed:
                # Update visual item's user_data with current layer name (single source of truth)
                item = item_state.item
                layer = self._layer_manager.get_layer(item_state.preview_layer_id)
                if layer:
                    item.user_data['_visual_layer_name'] = layer.name
                    
                    # If layer changed, update audio source to match new layer's source
                    # Find audio source from an existing event on the destination layer
                    if layer_changed:
                        old_audio_id = item.audio_id
                        old_audio_name = item.audio_name
                        new_audio_id = None
                        new_audio_name = None
                        
                        # Find an existing event on the destination layer to get its audio source
                        if self._scene:
                            for existing_item in self._scene._event_items.values():
                                if (existing_item.layer_id == layer.id and 
                                    existing_item.event_id != event_id):
                                    candidate_audio_id = getattr(existing_item, 'audio_id', None)
                                    candidate_audio_name = getattr(existing_item, 'audio_name', None)
                                    if candidate_audio_id or candidate_audio_name:
                                        new_audio_id = candidate_audio_id
                                        new_audio_name = candidate_audio_name
                                        break
                        
                        if new_audio_id or new_audio_name:
                            item.audio_id = new_audio_id
                            item.audio_name = new_audio_name
                            Log.debug(f"MovementController: Updated audio source for {event_id}: {old_audio_id} -> {new_audio_id}")
                            
                            # Invalidate waveform cache to force reload with new audio
                            if hasattr(item, '_invalidate_waveform_cache'):
                                item._invalidate_waveform_cache()
                        else:
                            Log.warning(f"MovementController: No audio source found on layer {layer.name} for {event_id}")
                
                results.append(EventMoveResult(
                    event_id=event_id,
                    old_time=item_state.original_time,
                    new_time=item_state.preview_time,
                    old_layer_id=item_state.original_layer_id,
                    new_layer_id=item_state.preview_layer_id,
                ))
        
        if results:
            Log.info(f"MovementController: committed move of {len(results)} items")
            self.moves_completed.emit(results)
            
            # Status message
            if len(results) == 1:
                layer = self._layer_manager.get_layer(results[0].new_layer_id)
                layer_name = layer.name if layer else "unknown"
                self.status_message.emit(
                    f"Moved to '{layer_name}' at {results[0].new_time:.3f}s",
                    False
                )
            else:
                self.status_message.emit(
                    f"Moved {len(results)} events",
                    False
                )
    
    # =========================================================================
    # Internal - Resize Operations
    # =========================================================================
    
    def _update_resize(self, pos: QPointF, snap_enabled: bool, snap_time_func, snap_calculator=None, unit_preference=None) -> None:
        """Update during resize drag.
        
        Args:
            pos: Current mouse position
            snap_enabled: Whether snapping is enabled
            snap_time_func: Legacy snap function
            snap_calculator: SnapCalculator instance (new system)
            unit_preference: UnitPreference for snapping (new system)
        """
        if not self._primary_item or not self._start_pos:
            return
        
        pps = self._primary_item._pixels_per_second
        delta_x = pos.x() - self._start_pos.x()
        delta_time = delta_x / pps
        
        min_duration = 0.01  # Minimum 10ms
        
        if self._edit_handle == EditHandle.RESIZE_LEFT:
            # Resize from left - changes start time and duration
            new_time = self._resize_original_time + delta_time
            new_time = max(0, new_time)
            
            if snap_enabled:
                if snap_calculator and unit_preference is not None:
                    new_time = snap_calculator.snap_time(new_time, pps, unit_preference)
                elif snap_time_func:
                    new_time = snap_time_func(new_time, pps)
            
            # Calculate new duration (end time stays fixed)
            end_time = self._resize_original_time + self._resize_original_duration
            new_duration = end_time - new_time
            
            if new_duration >= min_duration:
                self._primary_item.start_time = new_time
                self._primary_item.duration = new_duration
                self._primary_item._update_geometry()
        
        elif self._edit_handle == EditHandle.RESIZE_RIGHT:
            # Resize from right - only changes duration
            end_time = self._resize_original_time + self._resize_original_duration + delta_time
            
            if snap_enabled:
                if snap_calculator and unit_preference is not None:
                    end_time = snap_calculator.snap_time(end_time, pps, unit_preference)
                elif snap_time_func:
                    end_time = snap_time_func(end_time, pps)
            
            new_duration = end_time - self._primary_item.start_time
            
            if new_duration >= min_duration:
                self._primary_item.duration = new_duration
                self._primary_item._update_geometry()
    
    def _commit_resize(self) -> None:
        """Commit resize operation and emit results."""
        if not self._primary_item:
            return
        
        item = self._primary_item
        
        time_changed = item.start_time != self._resize_original_time
        duration_changed = item.duration != self._resize_original_duration
        
        if time_changed or duration_changed:
            result = EventResizeResult(
                event_id=item.event_id,
                old_time=self._resize_original_time,
                new_time=item.start_time,
                old_duration=self._resize_original_duration,
                new_duration=item.duration,
            )
            
            Log.info(f"MovementController: committed resize of {item.event_id}")
            self.resizes_completed.emit([result])
    
    # =========================================================================
    # Internal - Cleanup
    # =========================================================================
    
    def _cleanup(self) -> None:
        """Clean up after drag operation."""
        self._state = DragState.IDLE
        self._edit_handle = EditHandle.NONE
        self._primary_item = None
        self._start_pos = None
        self._items.clear()
        self._resize_original_time = 0.0
        self._resize_original_duration = 0.0

