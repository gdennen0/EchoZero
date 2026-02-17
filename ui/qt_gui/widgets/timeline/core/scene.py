"""
Timeline Scene
===============

QGraphicsScene subclass containing events, tracks, and playhead.
Handles event management, grid rendering, and editing coordination.

Uses LayerManager for all layer operations (single source of truth).
Uses MovementController for all drag/move operations.
"""

import math
import uuid
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Callable, Tuple
from PyQt6.QtWidgets import QGraphicsScene, QMenu
from PyQt6.QtCore import Qt, QRectF, QLineF, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QPixmap

# Local imports
from .style import TimelineStyle as Colors
from ..logging import TimelineLog as Log

from ..types import TimelineEvent, TimelineLayer, EventMoveResult, EventResizeResult, EventCreateResult, EventDeleteResult, EventSliceResult
from ..constants import (
    DEFAULT_PIXELS_PER_SECOND, TRACK_HEIGHT, TRACK_SPACING,
    MIN_MINOR_LINE_SPACING_PX, MIN_MAJOR_LINE_SPACING_PX, MAX_GRID_LINES
)
from .style import TimelineStyle
from ..events.items import BlockEventItem, MarkerEventItem, BaseEventItem
from ..playback.playhead import PlayheadItem
from ..grid_system import GridSystem
from ..timing import (
    GridCalculator, SnapCalculator, GridRenderer,
    UnitPreference, timebase_to_unit_preference
)

if TYPE_CHECKING:
    from ..events.layer_manager import LayerManager
    from ..events.movement_controller import MovementController

class TimelineScene(QGraphicsScene):
    """
    Scene containing timeline events and tracks.
    
    Uses external LayerManager for all layer operations.
    Uses external MovementController for drag operations.
    
    Features:
    - Event management (add, remove, edit)
    - Track/layer visualization
    - Grid rendering
    - Playhead integration
    - DAW-standard selection
    
    Signals:
        events_moved(list): List[EventMoveResult] - events were moved
        events_resized(list): List[EventResizeResult] - events were resized
        event_deleted(str): event_id - deletion requested (single)
        events_deleted(list): List[EventDeleteResult] - batch deletion requested (for performance)
        event_created(object): EventCreateResult - new event created
        playhead_seeked(float): time - playhead was dragged
        selection_changed(list): List[str] - selected event IDs changed
    """
    
    # New signals using typed result objects
    events_moved = pyqtSignal(list)      # List[EventMoveResult]
    events_resized = pyqtSignal(list)    # List[EventResizeResult]
    event_deleted = pyqtSignal(str)      # event_id (single)
    events_deleted = pyqtSignal(list)    # List[EventDeleteResult] (batch - for performance)
    event_created = pyqtSignal(object)   # EventCreateResult
    event_sliced = pyqtSignal(object)    # EventSliceResult
    playhead_seeked = pyqtSignal(float)  # time
    selection_changed = pyqtSignal(list) # List[str] - event IDs
    
    # Status message signal for user feedback
    status_message = pyqtSignal(str, bool)  # message, is_error
    
    def __init__(self, layer_manager: 'LayerManager', parent=None):
        super().__init__(parent)
        
        # External dependencies (injected)
        self._layer_manager = layer_manager
        self._movement_controller: Optional['MovementController'] = None
        self._command_bus = None  # Set by widget when facade is available
        
        # Audio lookup callback for simple waveform module
        self._audio_lookup_callback: Optional[Callable] = None
        
        # Event update callback (for updating event metadata like render_as_marker)
        self._event_update_callback: Optional[Callable] = None
        
        # Event storage
        self._event_items: Dict[str, BaseEventItem] = {}
        
        # State
        self._pixels_per_second = DEFAULT_PIXELS_PER_SECOND
        self._duration = 60.0
        self._editable = True
        
        # Drag state tracking for visual feedback
        self._drag_active = False
        self._drag_target_layer_id: Optional[str] = None
        
        # Waveforms are now drawn directly in BlockEventItem.paint() for efficiency
        
        # Grid system (legacy - will be replaced)
        self.grid_system = GridSystem()
        
        # New timing system
        self._grid_calculator = GridCalculator(frame_rate=30.0)
        # Pass grid_system to SnapCalculator for explicit interval mode support
        self._snap_calculator = SnapCalculator(self._grid_calculator, self.grid_system)
        self._grid_renderer = GridRenderer(self._grid_calculator)
        self._unit_preference = UnitPreference.SECONDS  # Default
        self._sync_grid_calculator()
        
        # Playhead
        self._playhead = PlayheadItem()
        self.addItem(self._playhead)
        
        # Background
        self.setBackgroundBrush(QBrush(Colors.BG_DARK))
        
        # Connect to layer manager changes
        self._layer_manager.layers_changed.connect(self._on_layers_changed)
        if hasattr(self._layer_manager, 'layer_updated'):
            self._layer_manager.layer_updated.connect(self._on_layer_updated)
        
        # Connect selection changed
        self.selectionChanged.connect(self._on_selection_changed)
    
    def set_data_item_repo(self, repo):
        """
        Set data_item_repo for direct audio item lookup by ID.
        
        Used by waveform_simple module to resolve audio_id -> AudioDataItem.
        
        Args:
            repo: DataItemRepository instance
        """
        self._data_item_repo = repo
        # Set on simple waveform module
        from ..events.waveform_simple import set_data_item_repo
        set_data_item_repo(repo)
    
    def set_audio_lookup_callback(self, callback: Optional[Callable]):
        """Deprecated: Use set_data_item_repo() instead."""
        self._audio_lookup_callback = callback
    
    def set_event_update_callback(self, callback: Optional[Callable]):
        """
        Set event update callback for updating event metadata.
        
        Args:
            callback: Function that accepts (event_id: str, metadata: Dict[str, Any]) and returns bool
        """
        self._event_update_callback = callback
    
    def set_command_bus(self, command_bus):
        """Set command bus for undoable operations."""
        self._command_bus = command_bus
    
    def set_movement_controller(self, controller: 'MovementController'):
        """
        Set the movement controller (called by TimelineWidget during setup).
        
        Args:
            controller: The MovementController instance
        """
        self._movement_controller = controller
        
        # Connect controller signals
        controller.moves_completed.connect(self._on_moves_completed)
        controller.resizes_completed.connect(self._on_resizes_completed)
        controller.drag_started.connect(self._on_drag_started)
        controller.drag_ended.connect(self._on_drag_ended)
        controller.drag_cancelled.connect(self._on_drag_ended)  # Cancel also resets items
        controller.status_message.connect(self.status_message)
    
    @property
    def layer_manager(self) -> 'LayerManager':
        """Get the layer manager."""
        return self._layer_manager
    
    @property
    def movement_controller(self) -> Optional['MovementController']:
        """Get the movement controller."""
        return self._movement_controller
    
    @property
    def playhead(self) -> PlayheadItem:
        """Get the playhead item."""
        return self._playhead
    
    @property
    def editable(self) -> bool:
        """Check if events are editable."""
        return self._editable
    
    @editable.setter
    def editable(self, value: bool):
        """Set whether events are editable."""
        self._editable = value
        for item in self._event_items.values():
            item._editable = value
    
    @property
    def pixels_per_second(self) -> float:
        """Current zoom level."""
        return self._pixels_per_second
    
    # =========================================================================
    # Zoom and Duration
    # =========================================================================
    
    def set_pixels_per_second(self, pps: float, defer_scene_rect: bool = False):
        """Update zoom level for all items.
        
        SIMPLE DESIGN: Update all items during zoom.
        The previous "visible-only" optimization caused visual glitches where
        off-screen items appeared at wrong positions when scrolled into view.
        
        Args:
            pps: Pixels per second (zoom level)
            defer_scene_rect: If True, skip scene rect update (caller will update later)
        """
        # Only update if actually changed (prevents redundant work)
        if abs(self._pixels_per_second - pps) < 0.01:
            return
        
        self._pixels_per_second = pps
        
        # Update ALL items - positions must be correct at all times
        # The "visible-only" optimization was flawed: Qt doesn't auto-update positions
        # when items scroll into view, causing visual glitches
        for item in self._event_items.values():
            item.set_pixels_per_second(pps)
        
        self._playhead.set_pixels_per_second(pps)
        
        # Only invalidate visible region, not entire scene
        # Waveforms are drawn in drawForeground() which is called automatically on repaint
        # No need to explicitly invalidate - Qt handles it via view update batching
        if not defer_scene_rect:
            self._update_scene_rect()
    
    def set_duration(self, duration: float):
        """Set total timeline duration."""
        self._duration = max(duration, 1)
        self._update_scene_rect()
    
    def _update_scene_rect(self):
        """Update scene rect based on duration and layers."""
        width = self._duration * self._pixels_per_second + 200
        
        # Get content height from layer manager
        content_height = self._layer_manager.get_total_height()
        if content_height < TRACK_HEIGHT:
            content_height = TRACK_HEIGHT + TRACK_SPACING + 50
        else:
            content_height += 20  # Padding
        
        # Get viewport height
        viewport_height = 0
        views = self.views()
        if views:
            view = views[0]
            viewport = view.viewport()
            if viewport:
                viewport_height = viewport.height()
        
        # Scene height is max of content and viewport
        if viewport_height > 0:
            scene_height = max(content_height, viewport_height)
        else:
            scene_height = max(content_height, 600)
        
        self.setSceneRect(0, 0, width, scene_height)
        self._playhead.set_scene_height(scene_height)
    
    # =========================================================================
    # Event Management
    # =========================================================================
    
    def clear_events(self):
        """Remove all event items."""
        for item in list(self._event_items.values()):
            self.removeItem(item)
        self._event_items.clear()
        self._update_scene_rect()
    
    def clear_events_except_layers(self, layer_ids_to_keep: set) -> int:
        """
        Clear all events except those in specified layers.
        
        Used to preserve synced layer events during set_events() calls.
        Synced layers have a different data source (external MA3) and should
        not be cleared during execution pipeline updates.
        
        Args:
            layer_ids_to_keep: Set of layer IDs whose events should be preserved
            
        Returns:
            Number of events removed
        """
        items_to_remove = [
            item for item in self._event_items.values()
            if item.layer_id not in layer_ids_to_keep
        ]
        
        removed_count = 0
        for item in items_to_remove:
            self.removeItem(item)
            if item.event_id in self._event_items:
                del self._event_items[item.event_id]
                removed_count += 1
        
        self._update_scene_rect()
        return removed_count
    
    def clear_events_in_layer(self, layer_id: str) -> int:
        """
        Clear all events in a specific layer.
        
        Used for per-layer reload operations. Only removes events
        belonging to the specified layer, leaving other layers intact.
        
        Args:
            layer_id: ID of the layer to clear
            
        Returns:
            Number of events removed
        """
        items_to_remove = [
            item for item in self._event_items.values()
            if item.layer_id == layer_id
        ]
        
        removed_count = 0
        for item in items_to_remove:
            self.removeItem(item)
            if item.event_id in self._event_items:
                del self._event_items[item.event_id]
                removed_count += 1
        
        self._update_scene_rect()
        return removed_count
    
    def add_event(
        self,
        event_id: str,
        start_time: float,
        duration: float,
        classification: str,
        layer_id: Optional[str] = None,
        audio_id: Optional[str] = None,
        audio_name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
        editable: Optional[bool] = None
    ) -> BaseEventItem:
        """
        Add an event to the scene.
        
        Args:
            event_id: Unique identifier
            start_time: Start time in seconds
            duration: Duration in seconds (0 for markers)
            classification: Semantic type/category (does NOT determine layer)
            layer_id: Which layer to place on (None = first layer or auto-create)
            audio_id: Optional audio source ID (for waveform display)
            audio_name: Optional audio source name (fallback for waveform)
            user_data: Optional user/application data
            editable: Override default editability
            
        Returns:
            Created event item
        """
        # Determine layer - NO FALLBACK CREATION, THROW ERROR IF LAYER NOT FOUND
        if layer_id is None:
            # Auto-assign to first layer, but ONLY if layers exist
            if self._layer_manager.get_layer_count() == 0:
                from src.utils.message import Log
                import traceback
                stack = traceback.extract_stack()
                caller_info = f"{stack[-2].filename}:{stack[-2].lineno}" if len(stack) >= 2 else "unknown"
                error_msg = f"TimelineScene.add_event(): Cannot add event - no layers exist and layer_id is None. Event classification: '{classification}'. Caller: {caller_info}. Layers must be created explicitly before adding events."
                Log.error(f"[LAYER_CREATE] {error_msg}")
                raise ValueError(error_msg)
            else:
                layer_id = self._layer_manager.get_first_layer_id()
        
        # Verify layer exists - THROW ERROR IF NOT FOUND
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            from src.utils.message import Log
            import traceback
            stack = traceback.extract_stack()
            caller_info = f"{stack[-2].filename}:{stack[-2].lineno}" if len(stack) >= 2 else "unknown"
            error_msg = f"TimelineScene.add_event(): Layer '{layer_id}' not found. Event classification: '{classification}'. Caller: {caller_info}. Layer must exist before adding events."
            Log.error(f"[LAYER_CREATE] {error_msg}")
            raise ValueError(error_msg)
        
        # Determine editability
        if editable is None:
            editable = self._editable
        
        # Enforce minimum duration - all events must have valid start and end times
        from ..types import MIN_EVENT_DURATION
        if duration < MIN_EVENT_DURATION:
            duration = MIN_EVENT_DURATION
        
        # Determine if this should render as marker (check user_data for render_as_marker)
        render_as_marker = user_data.get('render_as_marker', False) if user_data else False
        
        # Always create BlockEventItem - visual distinction is handled by render_as_marker property
        item = BlockEventItem(
            event_id=event_id,
            start_time=start_time,
            duration=duration,
            classification=classification,
            layer_id=layer_id,
            layer_manager=self._layer_manager,
            pixels_per_second=self._pixels_per_second,
            audio_id=audio_id,
            audio_name=audio_name,
            user_data=user_data,
            editable=editable,
            render_as_marker=render_as_marker
        )
        
        self.addItem(item)
        self._event_items[event_id] = item
        
        # Set initial visibility based on layer visibility (single source of truth)
        if layer:
            item.setVisible(layer.visible)
        
        # Update duration if needed - always expand to accommodate new events
        end_time = start_time + duration
        if end_time > self._duration:
            # Always add 1 second padding after the last event
            self._duration = end_time + 1.0
            self._update_scene_rect()
        
        # Waveforms are drawn per-event, individual item will update itself
        
        return item
    
    def add_event_from_data(self, event: TimelineEvent, editable: Optional[bool] = None) -> BaseEventItem:
        """
        Add an event from a TimelineEvent object.
        
        Args:
            event: TimelineEvent data
            editable: Override default editability
            
        Returns:
            Created event item
        """
        return self.add_event(
            event_id=event.id,
            start_time=event.time,
            duration=event.duration,
            classification=event.classification,
            layer_id=event.layer_id,
            audio_id=event.audio_id,
            audio_name=event.audio_name,
            user_data=event.user_data,
            editable=editable
        )
    
    def remove_event(self, event_id: str) -> bool:
        """Remove an event from the scene."""
        if event_id in self._event_items:
            item = self._event_items[event_id]
            self.removeItem(item)
            del self._event_items[event_id]
            # Waveforms are drawn per-event, individual items will update themselves
            return True
        return False
    
    def remove_events_batch(self, event_ids: List[str]) -> List[str]:
        """Remove multiple events from the scene (batch operation).
        
        Args:
            event_ids: List of event IDs to remove
            
        Returns:
            List of event IDs that were actually removed
        """
        removed = []
        for event_id in event_ids:
            if self.remove_event(event_id):
                removed.append(event_id)
        return removed
    
    def get_event_item(self, event_id: str) -> Optional[BaseEventItem]:
        """Get event item by ID."""
        return self._event_items.get(event_id)
    
    def get_all_event_items(self) -> Dict[str, BaseEventItem]:
        """Get all event items."""
        return self._event_items.copy()
    
    def get_event_data(self, event_id: str) -> Optional[TimelineEvent]:
        """Get event as TimelineEvent data object."""
        item = self._event_items.get(event_id)
        if not item:
            return None
        
        return TimelineEvent(
            id=item.event_id,
            time=item.start_time,
            duration=item.duration,
            classification=item.classification,
            layer_id=item.layer_id,
            audio_id=item.audio_id,
            audio_name=item.audio_name,
            user_data=dict(item.user_data) if item.user_data else {},
        )
    
    def get_all_events_data(self) -> List[TimelineEvent]:
        """Get all events as TimelineEvent data objects."""
        return [self.get_event_data(eid) for eid in self._event_items]
    
    def get_events_in_layer(self, layer_id: str) -> List[TimelineEvent]:
        """
        Get all events in a specific layer.
        
        Args:
            layer_id: Layer ID to filter by
            
        Returns:
            List of TimelineEvent objects in the specified layer
        """
        events = []
        for event_id, item in self._event_items.items():
            if item.layer_id == layer_id:
                events.append(self.get_event_data(event_id))
        return events
    
    def update_event(
        self,
        event_id: str,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        layer_id: Optional[str] = None,
        audio_id: Optional[str] = None,
        audio_name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing event in place.
        
        Note: Classification is NOT updated - it's immutable during normal operations.
        
        Args:
            event_id: ID of event to update
            start_time: New start time (None to keep current)
            duration: New duration (None to keep current)
            layer_id: New layer ID (None to keep current)
            audio_id: New audio source ID (None to keep current)
            audio_name: New audio source name (None to keep current)
            user_data: New user data (None to keep current)
            
        Returns:
            True if event was found and updated
        """
        from ..logging import TimelineLog as Log
        
        if event_id not in self._event_items:
            Log.debug(f"TimelineScene.update_event: event {event_id} NOT FOUND")
            return False
        
        item = self._event_items[event_id]
        old_layer_id = item.layer_id
        
        if start_time is not None:
            item.start_time = start_time
        
        if duration is not None:
            item.duration = duration
        
        # Update duration if event extends beyond timeline (check after all updates)
        end_time = item.start_time + item.duration
        if end_time > self._duration:
            self._duration = end_time + 1.0
            self._update_scene_rect()
        
        if layer_id is not None and layer_id != item.layer_id:
            # Verify layer exists
            if self._layer_manager.has_layer(layer_id):
                item.layer_id = layer_id
                # Also update user_data with layer name (single source of truth sync)
                layer = self._layer_manager.get_layer(layer_id)
                if layer:
                    item.user_data['_visual_layer_name'] = layer.name
                Log.debug(f"TimelineScene.update_event: {event_id} layer changed {old_layer_id} -> {layer_id}")
            else:
                Log.debug(f"TimelineScene.update_event: {event_id} layer {layer_id} does NOT exist")
        # Note: layer_id=None means "keep current layer" - no log needed
        
        if audio_id is not None:
            item.audio_id = audio_id
        
        if audio_name is not None:
            item.audio_name = audio_name
        
        if user_data is not None:
            # Merge user_data instead of replacing to preserve existing fields
            if item.user_data is None:
                item.user_data = {}
            item.user_data.update(user_data)
            # Update render_as_marker property if it changed
            render_as_marker = item.user_data.get('render_as_marker', False)
            if item.render_as_marker != render_as_marker:
                item.render_as_marker = render_as_marker
                # Update styling based on render mode
                if render_as_marker:
                    item._apply_qt_styling("marker_event")
                else:
                    item._apply_qt_styling("block_event")
        
        item._update_geometry()
        item._update_tooltip()
        item.update()  # Force repaint to show visual changes
        
        # Waveforms are drawn per-event, individual item will update itself
        
        return True
    
    def request_event_delete(self, event_id: str):
        """Request deletion of an event (emits signal)."""
        self.event_deleted.emit(event_id)
    
    def request_events_delete_batch(self, event_ids: List[str]):
        """Request deletion of multiple events (emits batch signal for performance).
        
        This method emits events_deleted signal with EventDeleteResult objects
        for all events that exist in the scene. This is more efficient than
        emitting individual event_deleted signals.
        
        Args:
            event_ids: List of event IDs to delete
        """
        if not event_ids:
            return
        
        # Gather event data for all events that exist
        results = []
        for event_id in event_ids:
            event_data = self.get_event_data(event_id)
            if event_data:  # Only include events that actually exist
                results.append(EventDeleteResult(event_id=event_id, event_data=event_data))
        
        # Emit batch signal if we have any results
        if results:
            self.events_deleted.emit(results)
    
    # =========================================================================
    # Selection
    # =========================================================================
    
    def get_selected_event_ids(self) -> List[str]:
        """Get IDs of selected events."""
        selected_items = self.selectedItems()
        event_ids = []
        for item in selected_items:
            if isinstance(item, BaseEventItem):
                event_id = item.event_id
                event_ids.append(event_id)
        return event_ids
    
    def get_selected_items(self) -> List[BaseEventItem]:
        """Get selected event items."""
        return [
            item for item in self.selectedItems()
            if isinstance(item, BaseEventItem)
        ]
    
    def select_event(self, event_id: str, clear_others: bool = True):
        """Select an event by ID."""
        if clear_others:
            self.clearSelection()
        
        if event_id in self._event_items:
            self._event_items[event_id].setSelected(True)
    
    def select_events(self, event_ids: List[str], clear_others: bool = True):
        """Select multiple events by ID."""
        if clear_others:
            self.clearSelection()
        
        for event_id in event_ids:
            if event_id in self._event_items:
                self._event_items[event_id].setSelected(True)
    
    def select_all(self):
        """Select all events."""
        for item in self._event_items.values():
            item.setSelected(True)
    
    def deselect_all(self):
        """Deselect all events."""
        self.clearSelection()
    
    def delete_selected_events(self):
        """Delete all selected events (uses batch deletion for efficiency)."""
        event_ids = self.get_selected_event_ids()
        if event_ids:
            self.request_events_delete_batch(event_ids)
    
    def slice_selected_event(self, slice_time: float) -> bool:
        """
        Slice the selected event at the given time.
        
        Splits a single selected event into two events at the slice position.
        The original event is replaced by two new events.
        
        Args:
            slice_time: Time position to slice at (in seconds)
            
        Returns:
            True if slice was successful, False otherwise
        """
        selected_ids = self.get_selected_event_ids()
        if len(selected_ids) != 1:
            Log.warning("TimelineScene: Slice requires exactly one selected event")
            return False
        
        event_id = selected_ids[0]
        item = self._event_items.get(event_id)
        if not item:
            Log.warning(f"TimelineScene: Event {event_id} not found for slicing")
            return False
        
        # Can only slice events with duration (not markers)
        if item.duration <= 0:
            Log.warning("TimelineScene: Cannot slice marker events (no duration)")
            return False
        
        # Check if slice_time is within event bounds
        event_start = item.start_time
        event_end = item.start_time + item.duration
        if slice_time <= event_start or slice_time >= event_end:
            Log.warning(f"TimelineScene: Slice time {slice_time:.3f}s must be within event bounds [{event_start:.3f}s, {event_end:.3f}s]")
            return False
        
        # Get original event data
        original_event = self.get_event_data(event_id)
        if not original_event:
            Log.warning(f"TimelineScene: Could not get event data for {event_id}")
            return False
        
        # Calculate durations for the two new events
        first_duration = slice_time - event_start
        second_duration = event_end - slice_time
        
        # Create first event (from start to slice)
        first_event = TimelineEvent(
            id=str(uuid.uuid4()),
            time=event_start,
            duration=first_duration,
            classification=original_event.classification,
            layer_id=original_event.layer_id,
            audio_id=original_event.audio_id,
            audio_name=original_event.audio_name,
            user_data=original_event.user_data.copy() if original_event.user_data else {}
        )
        
        # Create second event (from slice to end)
        second_event = TimelineEvent(
            id=str(uuid.uuid4()),
            time=slice_time,
            duration=second_duration,
            classification=original_event.classification,
            layer_id=original_event.layer_id,
            audio_id=original_event.audio_id,
            audio_name=original_event.audio_name,
            user_data=original_event.user_data.copy() if original_event.user_data else {}
        )
        
        # Emit slice result
        slice_result = EventSliceResult(
            original_event_id=event_id,
            original_event_data=original_event,
            slice_time=slice_time,
            first_event_data=first_event,
            second_event_data=second_event
        )
        self.event_sliced.emit(slice_result)
        
        Log.debug(f"TimelineScene: Sliced event {event_id} at {slice_time:.3f}s into two events")
        return True
    
    def _on_selection_changed(self):
        """Handle selection changes."""
        selected_ids = self.get_selected_event_ids()
        self.selection_changed.emit(selected_ids)
    
    # =========================================================================
    # Playhead
    # =========================================================================
    
    def set_playhead_position(self, seconds: float):
        """Set playhead position."""
        self._playhead.set_position(seconds)
    
    def get_playhead_position(self) -> float:
        """Get current playhead position."""
        return self._playhead.position_seconds
    
    # =========================================================================
    # Layer Change Handlers
    # =========================================================================
    
    def _sync_grid_calculator(self):
        """Sync grid calculator with grid system settings."""
        if hasattr(self, '_grid_calculator'):
            self._grid_calculator.frame_rate = self.grid_system.frame_rate
            self._unit_preference = timebase_to_unit_preference(self.grid_system.timebase_mode)
            # Sync snap calculator
            if hasattr(self, '_snap_calculator'):
                self._snap_calculator.snap_enabled = self.grid_system.snap_enabled
            # Sync grid renderer
            if hasattr(self, '_grid_renderer'):
                self._grid_renderer.show_grid_lines = self.grid_system.settings.show_grid_lines
    
    def _on_layers_changed(self):
        """Handle layer structure changes (add/delete/reorder).
        
        Full refresh needed since layer ordering changed.
        """
        for item in self._event_items.values():
            item._update_geometry()
        
        self._update_scene_rect()
        self._force_visual_update()
    
    def _on_layer_updated(self, layer_id: str):
        """Handle single layer property change with TARGETED updates.
        
        Updates:
        - Events on the changed layer: visibility toggle + geometry
        - Events on layers BELOW: Y position update (they shift up/down)
        - Events on layers ABOVE: no change needed
        """
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            return
        
        changed_layer_index = layer.index
        
        # Targeted update: only events that need changes
        for item in self._event_items.values():
            if item.layer_id == layer_id:
                # This layer's events: toggle visibility, update geometry
                item.setVisible(layer.visible)
                item._update_geometry()
            else:
                # Check if this event's layer is BELOW the changed layer
                event_layer = self._layer_manager.get_layer(item.layer_id)
                if event_layer and event_layer.index > changed_layer_index:
                    # Below changed layer in visual order: needs Y update
                    item._update_geometry()
                # Events ABOVE the changed layer: no update needed
        
        self._update_scene_rect()
        self._force_visual_update()
    
    def _force_visual_update(self):
        """Force immediate visual repaint of the scene and all views.
        
        This is necessary because setPos() alone may not trigger an
        immediate repaint in some PyQt configurations.
        """
        # Invalidate the entire scene to force full redraw
        self.invalidate(self.sceneRect())
        
        # Force all attached views to repaint their viewports
        for view in self.views():
            view.viewport().update()
    
    # =========================================================================
    # Movement Controller Handlers
    # =========================================================================
    
    def _on_moves_completed(self, results: List[EventMoveResult]):
        """Forward move results to widget."""
        self.events_moved.emit(results)
    
    def _on_resizes_completed(self, results: List[EventResizeResult]):
        """Forward resize results to widget."""
        self.events_resized.emit(results)
    
    def _on_drag_started(self):
        """Handle drag start - clear waveforms for dragged items."""
        from ..logging import TimelineLog as Log
        
        self._drag_active = True
        if self._movement_controller:
            self._drag_target_layer_id = self._movement_controller.drag_target_layer_id
            
            # Get items being dragged from movement controller
            dragged_items = list(self._movement_controller._items.values())
            Log.debug(f"Scene._on_drag_started: {len(dragged_items)} items being dragged")
            
            for item_state in dragged_items:
                if hasattr(item_state.item, 'on_drag_start'):
                    item_state.item.on_drag_start()
        
        self.update()
    
    def _on_drag_ended(self):
        """Handle drag end - reload waveforms for dragged items."""
        from ..logging import TimelineLog as Log
        
        # Get items that were being dragged BEFORE cleanup
        dragged_items = []
        if self._movement_controller and hasattr(self._movement_controller, '_items'):
            dragged_items = [state.item for state in self._movement_controller._items.values()]
        
        Log.debug(f"Scene._on_drag_ended: {len(dragged_items)} items to update")
        
        self._drag_active = False
        self._drag_target_layer_id = None
        
        # Reload waveforms for items that were dragged
        for item in dragged_items:
            if hasattr(item, 'on_drag_end'):
                item.on_drag_end()
        
        self.update()
    
    # =========================================================================
    # Background Drawing
    # =========================================================================
    
    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw track backgrounds and grid lines."""
        scene_rect = self.sceneRect()
        
        # Draw scene background
        draw_rect = rect.intersected(scene_rect)
        if not draw_rect.isEmpty():
            painter.fillRect(draw_rect, Colors.BG_DARK)
        
        # Draw group headers first (behind tracks, header above first child)
        GROUP_HEADER_HEIGHT = 18  # Height of group header divider
        from collections import defaultdict
        
        processed_groups: set = set()
        
        # Build map of groups to render headers
        grouped_layers: Dict[Optional[str], List] = defaultdict(list)
        all_layers = self._layer_manager.get_all_layers()
        for layer in all_layers:
            if layer.visible and layer.group_id:
                grouped_layers[layer.group_id].append(layer)
        
        # Render group headers in event area (header above first child layer, takes up space)
        for layer in all_layers:
            if not layer.visible:
                continue
            
            if layer.group_id and layer.group_id not in processed_groups:
                processed_groups.add(layer.group_id)
                
                if not layer.group_name:
                    continue
                
                # Position header above the first layer (header space is calculated in get_layer_y_position)
                first_layer_y = self._layer_manager.get_layer_y_position(layer.id)
                header_y = first_layer_y - GROUP_HEADER_HEIGHT
                
                # Only draw if visible
                if header_y + GROUP_HEADER_HEIGHT < rect.top() or header_y > rect.bottom():
                    continue
                
                # Header background (extends into event area as a folder/separator)
                header_bg_color = QColor(Colors.BG_MEDIUM)
                header_bg_color = header_bg_color.darker(115)  # 15% darker
                header_rect = QRectF(rect.left(), header_y, rect.width(), GROUP_HEADER_HEIGHT)
                painter.fillRect(header_rect, header_bg_color)
                
                # Header top border line only (bottom is handled by layer separator)
                painter.setPen(QPen(Colors.BORDER, 2))
                painter.drawLine(
                    int(rect.left()), int(header_y),
                    int(rect.right()), int(header_y)
                )  # Top border only
        
        # Draw track backgrounds
        for layer in self._layer_manager.get_all_layers():
            if not layer.visible:
                continue
            
            y = self._layer_manager.get_layer_y_position(layer.id)
            height = layer.height
            
            # Only draw if visible
            if y + height < rect.top() or y > rect.bottom():
                continue
            
            track_rect = QRectF(rect.left(), y, rect.width(), height)
            
            # Update drag target from controller
            if self._drag_active and self._movement_controller:
                self._drag_target_layer_id = self._movement_controller.drag_target_layer_id
            
            # Visual feedback for drag target
            is_drag_target = self._drag_active and self._drag_target_layer_id == layer.id
            
            if is_drag_target:
                highlight_color = Colors.BG_MEDIUM.lighter(120)
                painter.fillRect(track_rect, highlight_color)
                painter.setPen(QPen(Colors.ACCENT_BLUE.lighter(150), 2))
                painter.drawRect(track_rect)
            else:
                # Alternating track colors
                if layer.index % 2 == 0:
                    painter.fillRect(track_rect, Colors.BG_DARK)
                else:
                    painter.fillRect(track_rect, Colors.BG_MEDIUM.darker(110))
            
            # Note: Track separator lines removed - group headers and layer backgrounds provide sufficient visual separation
        
        # Draw grid lines
        if self.grid_system.settings.show_grid_lines:
            self._draw_grid_lines(painter, rect)
    
    
    def _get_audio_item_for_event(self, event_item) -> Optional['AudioDataItem']:
        """Get AudioDataItem for event using lookup callback."""
        if not self._audio_lookup_callback:
            return None
        
        # Use direct audio fields instead of metadata lookup
        audio_id = getattr(event_item, 'audio_id', None)
        audio_name = getattr(event_item, 'audio_name', None)
        
        try:
            return self._audio_lookup_callback(audio_id, audio_name)
        except Exception:
            return None
    
    def drawForeground(self, painter: QPainter, rect: QRectF):
        """
        Draw foreground elements (overlays, etc.).
        
        Waveforms are now drawn directly in BlockEventItem.paint() for efficiency.
        """
        pass
    
    def _draw_grid_lines(self, painter: QPainter, rect: QRectF):
        """
        Draw vertical grid lines.
        
        Uses new GridRenderer for clean separation of concerns.
        Falls back to old GridSystem if needed for compatibility.
        """
        # Use new GridRenderer if available
        if hasattr(self, '_grid_renderer'):
            # Sync settings
            self._sync_grid_calculator()
            
            # Use grid renderer
            self._grid_renderer.draw_grid(
                painter,
                rect,
                self._pixels_per_second,
                self._unit_preference,
                self.sceneRect()
            )
        else:
            # Fallback to old system with POC-verified optimizations
            self._draw_grid_lines_fallback(painter, rect)
    
    def _draw_grid_lines_fallback(self, painter: QPainter, rect: QRectF):
        """
        Draw grid lines using POC-verified optimizations (fallback when GridRenderer unavailable).
        
        Uses:
        1. Cosmetic pens (consistent width at any zoom)
        2. Batch drawing with drawLines()
        3. Integer-based line counting (no floating point bugs)
        4. Adaptive density based on pixel spacing
        """
        # Get intervals from old grid system
        major_interval, minor_interval = self.grid_system.get_major_minor_intervals(
            self._pixels_per_second
        )
        
        scene_rect = self.sceneRect()
        
        # === STEP 1: Calculate adaptive grid density ===
        minor_px = minor_interval * self._pixels_per_second
        
        # Calculate line skip factor for minor lines
        if minor_px >= MIN_MINOR_LINE_SPACING_PX:
            minor_skip = 1
        else:
            minor_skip = math.ceil(MIN_MINOR_LINE_SPACING_PX / minor_px)
            minor_skip = self._snap_to_nice_divisor(minor_skip)
        
        display_minor = minor_interval * minor_skip
        display_minor_px = display_minor * self._pixels_per_second
        
        # Calculate line skip factor for major lines
        major_px = major_interval * self._pixels_per_second
        
        if major_px >= MIN_MAJOR_LINE_SPACING_PX:
            major_skip = 1
        else:
            major_skip = math.ceil(MIN_MAJOR_LINE_SPACING_PX / major_px)
            major_skip = self._snap_to_nice_divisor(major_skip)
        
        display_major = major_interval * major_skip
        
        # === STEP 2: Prepare cosmetic pens ===
        minor_pen = QPen(Colors.BORDER.darker(130), 1, Qt.PenStyle.DotLine)
        minor_pen.setCosmetic(True)  # Consistent width at any zoom
        
        major_pen = QPen(Colors.BORDER, 1)
        major_pen.setCosmetic(True)  # Consistent width at any zoom
        
        # === STEP 3: Calculate visible range using INTEGER line indices ===
        visible_left = max(rect.left(), scene_rect.left())
        visible_right = min(rect.right(), scene_rect.right())
        
        if display_minor_px <= 0:
            return
        
        first_line_idx = max(0, int(visible_left / display_minor_px))
        last_line_idx = int(visible_right / display_minor_px) + 1
        
        # Limit for performance
        if last_line_idx - first_line_idx > MAX_GRID_LINES:
            last_line_idx = first_line_idx + MAX_GRID_LINES
        
        # === STEP 4: Collect lines for batch drawing ===
        minor_lines: List[QLineF] = []
        major_lines: List[QLineF] = []
        
        lines_per_major = max(1, round(display_major / display_minor)) if display_minor > 0 else 1
        
        y_top = max(rect.top(), scene_rect.top())
        y_bottom = min(rect.bottom(), scene_rect.bottom())
        max_x = scene_rect.right()
        
        for i in range(first_line_idx, last_line_idx + 1):
            x = i * display_minor_px
            
            if x < 0 or x > max_x:
                continue
            
            line = QLineF(x, y_top, x, y_bottom)
            
            # Integer-based major line detection (no floating point modulo!)
            is_major = (lines_per_major > 0 and i % lines_per_major == 0)
            
            if is_major:
                major_lines.append(line)
            else:
                minor_lines.append(line)
        
        # === STEP 5: Batch draw with drawLines() ===
        if minor_lines:
            painter.setPen(minor_pen)
            painter.drawLines(minor_lines)
        
        if major_lines:
            painter.setPen(major_pen)
            painter.drawLines(major_lines)
    
    @staticmethod
    def _snap_to_nice_divisor(value: int) -> int:
        """
        Snap a skip factor to a 'nice' number for musical/temporal alignment.
        """
        nice_numbers = [1, 2, 4, 5, 8, 10, 12, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 64, 80, 100]
        for nice in nice_numbers:
            if nice >= value:
                return nice
        return ((value + 99) // 100) * 100
    
    # =========================================================================
    # Mouse Events
    # =========================================================================
    
    def mousePressEvent(self, event):
        """Handle mouse press - create events on double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else None)
            
            if item is None and self._editable:
                # Double-click detection
                if hasattr(self, '_last_click_pos') and hasattr(self, '_last_click_time'):
                    import time
                    current_time = time.time()
                    if (current_time - self._last_click_time < 0.3 and
                        (event.scenePos() - self._last_click_pos).manhattanLength() < 10):
                        self._create_event_at(event.scenePos())
                        return
                
                import time
                self._last_click_pos = event.scenePos()
                self._last_click_time = time.time()
        
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """Show context menu for empty timeline space."""
        if not self._editable:
            super().contextMenuEvent(event)
            return

        item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else None)
        if item is not None:
            super().contextMenuEvent(event)
            return

        self._context_menu_pos = event.scenePos()

        menu = QMenu()
        add_oneshot_cursor_action = menu.addAction("Add One-shot at Cursor")
        add_oneshot_playhead_action = menu.addAction("Add One-shot at Playhead")
        menu.addSeparator()
        add_clip_cursor_action = menu.addAction("Add Clip at Cursor")
        add_clip_playhead_action = menu.addAction("Add Clip at Playhead")

        selected = menu.exec(event.screenPos())
        if selected == add_oneshot_cursor_action:
            self._create_event_at(self._context_menu_pos, duration_override=0.0)
        elif selected == add_oneshot_playhead_action:
            self._create_event_at_playhead(self._context_menu_pos, duration_override=0.0)
        elif selected == add_clip_cursor_action:
            self._create_event_at(self._context_menu_pos)
        elif selected == add_clip_playhead_action:
            self._create_event_at_playhead(self._context_menu_pos)

        event.accept()
    
    def _create_event_at(self, pos, duration_override: Optional[float] = None):
        """Create a new event at the given position."""
        time = pos.x() / self._pixels_per_second
        
        if self.grid_system.snap_enabled:
            # Use new SnapCalculator if available
            if hasattr(self, '_snap_calculator'):
                self._sync_grid_calculator()
                time = self._snap_calculator.snap_time(
                    time,
                    self._pixels_per_second,
                    self._unit_preference
                )
            else:
                # Fallback to old system
                time = self.grid_system.snap_time(time, self._pixels_per_second)
        
        # Get layer from Y position
        layer = self._layer_manager.get_layer_from_y(pos.y())
        if not layer:
            layer_id = self._layer_manager.get_first_layer_id()
            if not layer_id:
                Log.warning("TimelineScene: No layers available for event creation")
                return
            layer = self._layer_manager.get_layer(layer_id)
        else:
            layer_id = layer.id
        
        # Default duration - use current grid minor interval if snap enabled, otherwise 0.5s
        if duration_override is not None:
            duration = duration_override
        else:
            duration = self._get_default_duration()
        
        # Emit creation signal
        result = EventCreateResult(
            time=time,
            duration=duration,
            layer_id=layer_id,
            classification=layer.name if layer else "Event"
        )
        self.event_created.emit(result)
        
        Log.debug(f"TimelineScene: Create event at {time:.3f}s in layer '{layer_id}'")

    def _create_event_at_playhead(self, pos, duration_override: Optional[float] = None):
        """Create a new event at the current playhead time for the clicked layer."""
        playhead_time = self.get_playhead_position()
        playhead_pos = QPointF(playhead_time * self._pixels_per_second, pos.y())
        self._create_event_at(playhead_pos, duration_override=duration_override)

    def _get_default_duration(self) -> float:
        """Get default clip duration based on grid settings."""
        if self.grid_system.snap_enabled:
            if hasattr(self, '_grid_calculator'):
                self._sync_grid_calculator()
                # Multipliers removed - grid auto-calculated from timebase/FPS
                _, minor_interval = self._grid_calculator.get_intervals(
                    self._pixels_per_second,
                    self._unit_preference
                )
                return minor_interval
            # Multipliers removed - grid auto-calculated from timebase/FPS
            _, minor_interval = self.grid_system.get_major_minor_intervals(
                self._pixels_per_second
            )
            return minor_interval
        return 0.5
    
    def keyPressEvent(self, event):
        """Handle keyboard input."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected_events()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel drag if active
            if self._movement_controller and self._movement_controller.is_dragging:
                self._movement_controller.cancel_drag()
                event.accept()
        elif event.key() == Qt.Key.Key_S and not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier)):
            # Slice event at playhead position (S key, no modifiers)
            slice_time = self.get_playhead_position()
            if self.slice_selected_event(slice_time):
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
