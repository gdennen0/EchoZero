"""
Timeline Commands - Undoable Timeline Event Operations

Commands for timeline event manipulation (move, resize, create, delete).
These integrate with the Editor block's timeline widget.

STANDARD TIMELINE COMMANDS
==========================

| Command                   | Redo Action              | Undo Action              |
|---------------------------|--------------------------|--------------------------|
| MoveEventCommand          | Moves event to new time  | Restores original time   |
| ResizeEventCommand        | Resizes event            | Restores original size   |
| CreateEventCommand        | Creates new event        | Deletes event            |
| DeleteEventCommand        | Deletes event            | Recreates event          |

Note: Timeline events are stored in block metadata, not as separate entities.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Callable
from .base_command import EchoZeroCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class MoveEventCommand(EchoZeroCommand):
    """
    Move a timeline event to a new time and/or layer.
    
    Redo: Moves event to new position
    Undo: Restores original position
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block containing the event
        event_id: ID of the event to move
        new_time: New start time in seconds
        new_layer_index: New layer index
        old_time: Original time (for undo)
        old_layer_index: Original layer index (for undo)
    """
    
    COMMAND_TYPE = "timeline.move_event"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        event_id: str,
        new_time: float,
        new_layer_index: int,
        old_time: Optional[float] = None,
        old_layer_index: Optional[int] = None
    ):
        super().__init__(facade, f"Move Event")
        
        self._block_id = block_id
        self._event_id = event_id
        self._new_time = new_time
        self._new_layer_index = new_layer_index
        self._old_time = old_time
        self._old_layer_index = old_layer_index
    
    def redo(self):
        """Move event to new position."""
        self._update_event_in_metadata(self._new_time, self._new_layer_index)
    
    def undo(self):
        """Restore original position."""
        if self._old_time is not None:
            self._update_event_in_metadata(self._old_time, self._old_layer_index)
    
    def _update_event_in_metadata(self, time: float, layer_index: int):
        """Update event data in block metadata."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            self._log_error(f"Block not found: {self._block_id}")
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        for event in events:
            if event.get("id") == self._event_id:
                event["start_time"] = time
                event["layer_index"] = layer_index
                break
        
        block.metadata["events"] = events
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )


class ResizeEventCommand(EchoZeroCommand):
    """
    Resize a timeline event.
    
    Redo: Sets new start time and duration
    Undo: Restores original start time and duration
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block containing the event
        event_id: ID of the event to resize
        new_time: New start time in seconds
        new_duration: New duration in seconds
        old_time: Original time (for undo)
        old_duration: Original duration (for undo)
    """
    
    COMMAND_TYPE = "timeline.resize_event"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        event_id: str,
        new_time: float,
        new_duration: float,
        old_time: Optional[float] = None,
        old_duration: Optional[float] = None
    ):
        super().__init__(facade, f"Resize Event")
        
        self._block_id = block_id
        self._event_id = event_id
        self._new_time = new_time
        self._new_duration = new_duration
        self._old_time = old_time
        self._old_duration = old_duration
    
    def redo(self):
        """Resize event."""
        self._update_event_in_metadata(self._new_time, self._new_duration)
    
    def undo(self):
        """Restore original size."""
        if self._old_time is not None:
            self._update_event_in_metadata(self._old_time, self._old_duration)
    
    def _update_event_in_metadata(self, time: float, duration: float):
        """Update event data in block metadata."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            self._log_error(f"Block not found: {self._block_id}")
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        for event in events:
            if event.get("id") == self._event_id:
                event["start_time"] = time
                event["duration"] = duration
                break
        
        block.metadata["events"] = events
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )


class CreateEventCommand(EchoZeroCommand):
    """
    Create a new timeline event.
    
    Redo: Creates the event
    Undo: Deletes the event
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        event_data: Dict with event properties (id, start_time, duration, classification, etc.)
    """
    
    COMMAND_TYPE = "timeline.create_event"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        event_data: Dict[str, Any]
    ):
        classification = event_data.get("classification", "Event")
        super().__init__(facade, f"Create {classification}")
        
        self._block_id = block_id
        self._event_data = event_data.copy()
        self._event_id = event_data.get("id")
    
    def redo(self):
        """Create the event."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            self._log_error(f"Block not found: {self._block_id}")
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        # Check if event already exists (for redo)
        exists = any(e.get("id") == self._event_id for e in events)
        if not exists:
            events.append(self._event_data)
            block.metadata["events"] = events
            self._facade.block_service.update_block(
                self._facade.current_project_id,
                self._block_id,
                block
            )
    
    def undo(self):
        """Delete the event."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        # Remove the event
        events = [e for e in events if e.get("id") != self._event_id]
        block.metadata["events"] = events
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )


class DeleteEventCommand(EchoZeroCommand):
    """
    Delete a timeline event.
    
    Redo: Deletes the event
    Undo: Recreates the event with original data
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        event_id: ID of the event to delete
    """
    
    COMMAND_TYPE = "timeline.delete_event"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        event_id: str
    ):
        super().__init__(facade, f"Delete Event")
        
        self._block_id = block_id
        self._event_id = event_id
        self._deleted_event_data: Optional[Dict[str, Any]] = None
    
    def redo(self):
        """Delete the event, storing data for undo."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        # Store event data before deletion (first time only)
        if self._deleted_event_data is None:
            for event in events:
                if event.get("id") == self._event_id:
                    self._deleted_event_data = event.copy()
                    break
        
        # Remove the event
        events = [e for e in events if e.get("id") != self._event_id]
        block.metadata["events"] = events
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )
    
    def undo(self):
        """Recreate the deleted event."""
        if not self._deleted_event_data:
            self._log_warning("No event data stored, cannot undo")
            return
        
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        # Add back the event
        events.append(self._deleted_event_data)
        block.metadata["events"] = events
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )


class BatchMoveEventsCommand(EchoZeroCommand):
    """
    Move multiple timeline events at once.
    
    Redo: Moves all events to new positions
    Undo: Restores all events to original positions
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        event_moves: List of dicts with event_id, new_time, new_layer_index, old_time, old_layer_index
    """
    
    COMMAND_TYPE = "timeline.batch_move_events"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        event_moves: List[Dict[str, Any]]
    ):
        count = len(event_moves)
        super().__init__(facade, f"Move {count} Events")
        
        self._block_id = block_id
        self._event_moves = event_moves
    
    def redo(self):
        """Move all events to new positions."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        # Build lookup for efficient updates
        events_by_id = {e.get("id"): e for e in events}
        
        for move in self._event_moves:
            event = events_by_id.get(move["event_id"])
            if event:
                event["start_time"] = move["new_time"]
                event["layer_index"] = move["new_layer_index"]
        
        block.metadata["events"] = list(events_by_id.values())
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )
    
    def undo(self):
        """Restore all events to original positions."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        events = block.metadata.get("events", [])
        
        # Build lookup for efficient updates
        events_by_id = {e.get("id"): e for e in events}
        
        for move in self._event_moves:
            event = events_by_id.get(move["event_id"])
            if event and move.get("old_time") is not None:
                event["start_time"] = move["old_time"]
                event["layer_index"] = move["old_layer_index"]
        
        block.metadata["events"] = list(events_by_id.values())
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )


# =============================================================================
# Layer Commands - UI-layer commands for timeline layer operations
# =============================================================================
# These commands work directly with LayerManager (UI-layer state) and do not
# require ApplicationFacade. Pass None for facade parameter.
# =============================================================================


class SetLayerVisibilityCommand(EchoZeroCommand):
    """
    Set visibility of a timeline layer.
    
    Redo: Sets layer to new visibility state
    Undo: Restores original visibility state
    """
    
    COMMAND_TYPE = "timeline.set_layer_visibility"
    
    def __init__(
        self,
        layer_manager: Any,  # LayerManager - avoid circular import
        layer_id: str,
        visible: bool
    ):
        if layer_manager is None:
            raise ValueError("LayerManager cannot be None")
        
        # Get layer name for description
        layer = layer_manager.get_layer(layer_id)
        layer_name = layer.name if layer else layer_id
        action = "Show" if visible else "Hide"
        
        # Pass None for facade - this is UI-layer command
        super().__init__(None, f"{action} Layer: {layer_name}")
        
        self._layer_manager = layer_manager
        self._layer_id = layer_id
        self._new_visible = visible
        self._old_visible: Optional[bool] = None
    
    def redo(self):
        """Set layer to new visibility."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        # Capture original state (first time only)
        if self._old_visible is None:
            layer = self._layer_manager.get_layer(self._layer_id)
            if layer:
                self._old_visible = layer.visible
            else:
                self._log_warning(f"Layer not found: {self._layer_id}")
                return
        
        # Apply new visibility
        self._layer_manager.update_layer(self._layer_id, visible=self._new_visible)
    
    def undo(self):
        """Restore original visibility."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        if self._old_visible is not None:
            self._layer_manager.update_layer(self._layer_id, visible=self._old_visible)


class RenameLayerCommand(EchoZeroCommand):
    """
    Rename a timeline layer.
    
    Redo: Sets layer to new name and updates all events with matching _visual_layer_name
    Undo: Restores original name and updates all events back
    
    If update_events_callback is provided, it will be called with (old_name, new_name)
    to update all events in EventDataItems that reference the old layer name.
    """
    
    COMMAND_TYPE = "timeline.rename_layer"
    
    def __init__(
        self,
        layer_manager: Any,  # LayerManager
        layer_id: str,
        new_name: str,
        update_events_callback: Optional[Callable[[str, str], None]] = None
    ):
        if layer_manager is None:
            raise ValueError("LayerManager cannot be None")
        
        layer = layer_manager.get_layer(layer_id)
        old_name = layer.name if layer else layer_id
        
        super().__init__(None, f"Rename Layer: {old_name} -> {new_name}")
        
        self._layer_manager = layer_manager
        self._layer_id = layer_id
        self._new_name = new_name
        self._old_name: Optional[str] = None
        self._update_events_callback = update_events_callback
    
    def redo(self):
        """Set layer to new name and update all events."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        # Capture original state (first time only)
        if self._old_name is None:
            layer = self._layer_manager.get_layer(self._layer_id)
            if layer:
                self._old_name = layer.name
            else:
                self._log_warning(f"Layer not found: {self._layer_id}")
                return
        
        # Update layer name in LayerManager
        self._layer_manager.update_layer(self._layer_id, name=self._new_name)
        
        # Update all events that reference the old layer name
        if self._update_events_callback and self._old_name:
            try:
                self._update_events_callback(self._old_name, self._new_name)
            except Exception as e:
                self._log_warning(f"Failed to update events on layer rename: {e}")
    
    def undo(self):
        """Restore original name and update all events back."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        if self._old_name is not None:
            # Restore layer name in LayerManager
            self._layer_manager.update_layer(self._layer_id, name=self._old_name)
            
            # Update all events back to old name
            if self._update_events_callback:
                try:
                    self._update_events_callback(self._new_name, self._old_name)
                except Exception as e:
                    self._log_warning(f"Failed to update events on layer rename undo: {e}")


class SetLayerColorCommand(EchoZeroCommand):
    """
    Change a timeline layer's color.
    
    Redo: Sets layer to new color
    Undo: Restores original color
    """
    
    COMMAND_TYPE = "timeline.set_layer_color"
    
    def __init__(
        self,
        layer_manager: Any,  # LayerManager
        layer_id: str,
        new_color: str  # Hex color string
    ):
        if layer_manager is None:
            raise ValueError("LayerManager cannot be None")
        
        layer = layer_manager.get_layer(layer_id)
        layer_name = layer.name if layer else layer_id
        
        super().__init__(None, f"Change Color: {layer_name}")
        
        self._layer_manager = layer_manager
        self._layer_id = layer_id
        self._new_color = new_color
        self._old_color: Optional[str] = None
    
    def redo(self):
        """Set layer to new color."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        # Capture original state (first time only)
        if self._old_color is None:
            layer = self._layer_manager.get_layer(self._layer_id)
            if layer:
                self._old_color = layer.color
            else:
                self._log_warning(f"Layer not found: {self._layer_id}")
                return
        
        self._layer_manager.update_layer(self._layer_id, color=self._new_color)
    
    def undo(self):
        """Restore original color."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        if self._old_color is not None:
            self._layer_manager.update_layer(self._layer_id, color=self._old_color)


class SetLayerLockCommand(EchoZeroCommand):
    """
    Lock or unlock a timeline layer.
    
    Redo: Sets layer to new lock state
    Undo: Restores original lock state
    """
    
    COMMAND_TYPE = "timeline.set_layer_lock"
    
    def __init__(
        self,
        layer_manager: Any,  # LayerManager
        layer_id: str,
        locked: bool
    ):
        if layer_manager is None:
            raise ValueError("LayerManager cannot be None")
        
        layer = layer_manager.get_layer(layer_id)
        layer_name = layer.name if layer else layer_id
        action = "Lock" if locked else "Unlock"
        
        super().__init__(None, f"{action} Layer: {layer_name}")
        
        self._layer_manager = layer_manager
        self._layer_id = layer_id
        self._new_locked = locked
        self._old_locked: Optional[bool] = None
    
    def redo(self):
        """Set layer lock state."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        # Capture original state (first time only)
        if self._old_locked is None:
            layer = self._layer_manager.get_layer(self._layer_id)
            if layer:
                self._old_locked = layer.locked
            else:
                self._log_warning(f"Layer not found: {self._layer_id}")
                return
        
        self._layer_manager.update_layer(self._layer_id, locked=self._new_locked)
    
    def undo(self):
        """Restore original lock state."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        if self._old_locked is not None:
            self._layer_manager.update_layer(self._layer_id, locked=self._old_locked)


class MoveLayerCommand(EchoZeroCommand):
    """
    Move a timeline layer to a new position in the layer order.
    
    Redo: Moves layer to new index
    Undo: Restores layer to original index
    """
    
    COMMAND_TYPE = "timeline.move_layer"
    
    def __init__(
        self,
        layer_manager: Any,  # LayerManager
        layer_id: str,
        new_index: int
    ):
        if layer_manager is None:
            raise ValueError("LayerManager cannot be None")
        
        layer = layer_manager.get_layer(layer_id)
        layer_name = layer.name if layer else layer_id
        
        super().__init__(None, f"Move Layer: {layer_name}")
        
        self._layer_manager = layer_manager
        self._layer_id = layer_id
        self._new_index = new_index
        self._old_index: Optional[int] = None
    
    def redo(self):
        """Move layer to new position."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        # Capture original state (first time only)
        if self._old_index is None:
            layer = self._layer_manager.get_layer(self._layer_id)
            if layer:
                self._old_index = layer.index
            else:
                self._log_warning(f"Layer not found: {self._layer_id}")
                return
        
        self._layer_manager.reorder_layer(self._layer_id, self._new_index)
    
    def undo(self):
        """Restore layer to original position."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        if self._old_index is not None:
            self._layer_manager.reorder_layer(self._layer_id, self._old_index)


class DeleteLayerCommand(EchoZeroCommand):
    """
    Delete a timeline layer.
    
    Redo: Deletes layer (events are NOT deleted - must be handled separately)
    Undo: Recreates layer at original position with original properties
    
    Note: This command only handles the layer itself. If events need to be
    moved or deleted, that should be done via separate commands in a macro.
    """
    
    COMMAND_TYPE = "timeline.delete_layer"
    
    def __init__(
        self,
        layer_manager: Any,  # LayerManager
        layer_id: str
    ):
        if layer_manager is None:
            raise ValueError("LayerManager cannot be None")
        
        layer = layer_manager.get_layer(layer_id)
        layer_name = layer.name if layer else layer_id
        
        super().__init__(None, f"Delete Layer: {layer_name}")
        
        self._layer_manager = layer_manager
        self._layer_id = layer_id
        # Store full layer data for undo
        self._deleted_layer_data: Optional[Dict[str, Any]] = None
    
    def redo(self):
        """Delete the layer."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        # Capture layer data before deletion (first time only)
        if self._deleted_layer_data is None:
            layer = self._layer_manager.get_layer(self._layer_id)
            if layer:
                self._deleted_layer_data = {
                    'id': layer.id,
                    'name': layer.name,
                    'index': layer.index,
                    'height': layer.height,
                    'color': layer.color,
                    'locked': layer.locked,
                    'visible': layer.visible,
                    'collapsed': layer.collapsed,
                }
            else:
                self._log_warning(f"Layer not found: {self._layer_id}")
                return
        
        self._layer_manager.delete_layer(self._layer_id)
    
    def undo(self):
        """Recreate the deleted layer."""
        if self._layer_manager is None:
            self._log_error("LayerManager is None")
            return
        
        if self._deleted_layer_data is None:
            self._log_warning("No layer data stored, cannot undo")
            return
        
        data = self._deleted_layer_data
        
        # Recreate the layer with original properties
        layer = self._layer_manager.create_layer(
            name=data['name'],
            layer_id=data['id'],
            index=data['index'],
            height=data['height'],
            color=data['color']
        )
        
        # Restore additional properties
        if layer:
            self._layer_manager.update_layer(
                layer.id,
                locked=data['locked'],
                visible=data['visible'],
                collapsed=data.get('collapsed', False)
            )



