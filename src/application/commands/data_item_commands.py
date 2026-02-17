"""
Commands for data item operations (undoable).

These commands handle operations on data items (EventDataItem, AudioData, etc.)
that need to be undoable through facade.command_bus.

UI refresh is handled by QUndoStack.indexChanged signal in MainWindow,
NOT by individual command emissions.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple
from .base_command import EchoZeroCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade

class BatchDeleteEventsFromDataItemCommand(EchoZeroCommand):
    """
    Delete multiple events from EventDataItems (undoable).
    
    This command handles batch deletion correctly by:
    1. Capturing all event objects FIRST (before any deletion)
    2. Deleting by object reference (not index) to prevent index shifting bugs
    
    Works for both single and batch operations - use this as the unified pathway.
    
    Args:
        facade: ApplicationFacade instance
        deletions: List of (data_item_id, event_index) tuples to delete
    """
    
    COMMAND_TYPE = "data_item.event.batch_delete"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        deletions: List[Tuple[str, int]]
    ):
        count = len(deletions)
        super().__init__(facade, f"Delete {count} Event{'s' if count != 1 else ''}")
        
        self._deletions = deletions  # List of (data_item_id, event_index)
        
        # Store deleted events for undo: {data_item_id: [event_dict, ...]}
        # Events stored in order they were captured (before deletion)
        self._deleted_events: Dict[str, List[Dict[str, Any]]] = {}
        
        # Store original indices for proper restoration: {data_item_id: [index, ...]}
        # Indices stored in same order as deleted_events
        self._original_indices: Dict[str, List[int]] = {}
        
        # Store event objects for deletion (captured before any deletion)
        # Type: List[Tuple[str, Any]] where second element is Event object
        self._event_objects: List[Tuple[str, Any]] = []  # (data_item_id, event)
        
        # Store data item references to ensure we use the same instance
        self._data_items: Dict[str, Any] = {}  # {data_item_id: EventDataItem}
        
        # For layer-based EventDataItem: (event_dict, layer_name, index_in_layer) for undo
        self._restore_info: Dict[str, List[Tuple[Dict[str, Any], str, int]]] = {}
    
    def redo(self):
        """Delete all events by object reference (prevents index shifting).
        
        Handles both initial execution and redo after undo:
        - First execution: Captures events by object reference
        - Redo after undo: Re-captures events at original indices (new objects after undo)
        """
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        
        Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Starting. Executed: {self._executed}, Deletions: {self._deletions}")
        
        if not self._executed:
            # First execution: capture all event objects BEFORE any deletion
            Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: First execution - capturing events")
            self._capture_all_events()
            self._executed = True
            Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Captured {len(self._event_objects)} event objects")
        else:
            # Redo after undo: Events were restored, so we need to re-capture them
            Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Redo after undo - re-capturing events at original indices")
            self._event_objects.clear()
            
            for data_item_id, original_indices in self._original_indices.items():
                data_item = self._facade.data_item_repo.get(data_item_id)
                if not data_item or not isinstance(data_item, EventDataItem):
                    continue
                events = data_item.get_events()
                for original_index in original_indices:
                    if 0 <= original_index < len(events):
                        event_obj = events[original_index]
                        self._event_objects.append((data_item_id, event_obj))
                    else:
                        Log.warning(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Index {original_index} out of range for {data_item_id} (has {len(events)} events)")
        
        # Group deletions by data_item_id to update each item once
        deletions_by_item: Dict[str, List[Any]] = {}
        for data_item_id, event_obj in self._event_objects:
            if data_item_id not in deletions_by_item:
                deletions_by_item[data_item_id] = []
            deletions_by_item[data_item_id].append(event_obj)
        
        Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Grouped into {len(deletions_by_item)} data items: {list(deletions_by_item.keys())}")
        
        # Delete all events by object reference (not index)
        # This prevents index shifting - we delete the actual objects we captured
        for data_item_id, event_objs in deletions_by_item.items():
            Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Processing {data_item_id} with {len(event_objs)} events")
            try:
                # CRITICAL: Use the SAME data_item instance that we captured events from
                # Getting a fresh copy from repo would create NEW event objects,
                # breaking the object identity comparison in remove_event()
                data_item = self._data_items.get(data_item_id)
                used_cached = data_item is not None
                if not data_item:
                    # Fallback to repo if somehow not in _data_items (shouldn't happen)
                    data_item = self._facade.data_item_repo.get(data_item_id)
                if not data_item or not isinstance(data_item, EventDataItem):
                    self._log_error(f"Data item not found: {data_item_id}")
                    continue
                
                # Delete each event by object reference
                failed_count = 0
                initial_count = len(data_item.get_events())
                Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: {data_item_id} has {initial_count} events before deletion")
                
                for i, event_obj in enumerate(event_objs):
                    event_time = getattr(event_obj, 'time', 'unknown')
                    Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Deleting event {i+1}/{len(event_objs)} from {data_item_id} (time={event_time}, obj_id={id(event_obj)})")
                    
                    if not data_item.remove_event(event_obj):
                        failed_count += 1
                        # Log details for debugging
                        events_in_item = len(data_item.get_events())
                        event_in_list = event_obj in data_item._events if hasattr(data_item, '_events') else False
                        error_msg = (
                            f"Failed to remove event from {data_item_id}. "
                            f"Event object id: {id(event_obj)}, "
                            f"Event time: {event_time}, "
                            f"Events in item: {events_in_item}, "
                            f"Event in _events list: {event_in_list}"
                        )
                        self._log_error(error_msg)
                        Log.warning(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: {error_msg}")
                    else:
                        current_count = len(data_item.get_events())
                        Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: Successfully removed event {i+1} from {data_item_id}. Remaining: {current_count}")
                
                final_count = len(data_item.get_events())
                Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: {data_item_id} now has {final_count} events (deleted {initial_count - final_count})")
                
                if failed_count == 0:
                    # Save back to database only if all deletions succeeded
                    Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: All deletions succeeded for {data_item_id}, updating repo")
                    self._facade.data_item_repo.update(data_item)
                else:
                    error_msg = f"Failed to remove {failed_count} of {len(event_objs)} events from {data_item_id}"
                    self._log_error(error_msg)
                    Log.error(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.redo: {error_msg}")
                    
            except Exception as e:
                self._log_error(f"Error deleting events from {data_item_id}: {e}")
                import traceback
                self._log_error(traceback.format_exc())
    
    def undo(self):
        """Restore all deleted events at their original positions."""
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem, Event
        
        for data_item_id, event_dicts in self._deleted_events.items():
            try:
                data_item = self._facade.data_item_repo.get(data_item_id)
                if not data_item or not isinstance(data_item, EventDataItem):
                    self._log_error(f"Data item not found for undo: {data_item_id}")
                    continue
                
                restore_info = self._restore_info.get(data_item_id, [])
                use_layer_restore = (
                    restore_info
                    and hasattr(data_item, "_layers")
                    and data_item._layers
                )
                
                if use_layer_restore:
                    # Layer-based structure: restore to correct layer at correct position
                    # Sort by (layer_name, -index_in_layer) descending so we insert high indices first
                    sorted_info = sorted(
                        restore_info,
                        key=lambda x: (x[1], -x[2]),
                        reverse=True
                    )
                    for event_dict, layer_name, index_in_layer in sorted_info:
                        event = Event(
                            time=event_dict.get("time", 0.0),
                            classification=event_dict.get("classification", ""),
                            duration=event_dict.get("duration", 0.0),
                            metadata=event_dict.get("metadata", {}),
                            id=event_dict.get("id")
                        )
                        layer = data_item.get_layer_by_name(layer_name) if layer_name else None
                        if layer and index_in_layer >= 0:
                            layer.events.insert(index_in_layer, event)
                        elif layer:
                            layer.events.append(event)
                        else:
                            from src.shared.domain.entities import EventLayer
                            new_layer = EventLayer(name=layer_name or event.classification or "Default")
                            new_layer.events.append(event)
                            data_item.add_layer(new_layer)
                    data_item.event_count = sum(len(l.events) for l in data_item._layers)
                else:
                    # Legacy flat structure
                    original_indices = self._original_indices.get(data_item_id, [])
                    if original_indices and len(original_indices) == len(event_dicts):
                        indexed_events = list(zip(original_indices, event_dicts))
                        indexed_events.sort(reverse=True, key=lambda x: x[0])
                        for original_index, event_dict in indexed_events:
                            event = Event(
                                time=event_dict.get("time", 0.0),
                                classification=event_dict.get("classification", ""),
                                duration=event_dict.get("duration", 0.0),
                                metadata=event_dict.get("metadata")
                            )
                            events = getattr(data_item, "_events", None)
                            if events is not None:
                                insert_index = min(original_index, len(events))
                                events.insert(insert_index, event)
                                data_item.event_count = len(events)
                            else:
                                data_item.add_event(
                                    time=event.time,
                                    duration=event.duration,
                                    classification=event.classification,
                                    metadata=event.metadata,
                                    layer_name=event.classification or "Default"
                                )
                    else:
                        for event_dict in reversed(event_dicts):
                            data_item.add_event(
                                time=event_dict.get("time", 0.0),
                                duration=event_dict.get("duration", 0.0),
                                classification=event_dict.get("classification", ""),
                                metadata=event_dict.get("metadata", {}),
                                layer_name=event_dict.get("classification", "Default")
                            )
                
                self._facade.data_item_repo.update(data_item)
                Log.debug(f"[DELETE DEBUG] BatchDeleteEventsFromDataItemCommand.undo: Restored {len(event_dicts)} events to {data_item_id}")
                
            except Exception as e:
                self._log_error(f"Error restoring events to {data_item_id}: {e}")
                import traceback
                self._log_error(traceback.format_exc())
    
    def _capture_all_events(self):
        """Capture all event objects and data BEFORE any deletion.
        
        This is critical - we must capture everything before deleting anything,
        otherwise indices shift and we delete the wrong events.
        
        Supports both legacy flat _events and layer-based EventDataItem structure.
        """
        try:
            from src.shared.domain.entities import EventDataItem
            
            # Group deletions by data_item_id for efficient processing
            deletions_by_item: Dict[str, List[int]] = {}
            for data_item_id, event_index in self._deletions:
                if data_item_id not in deletions_by_item:
                    deletions_by_item[data_item_id] = []
                deletions_by_item[data_item_id].append(event_index)
            
            # Store data items by ID to ensure we use the same instance
            self._data_items = {}
            
            for data_item_id, event_indices in deletions_by_item.items():
                data_item = self._facade.data_item_repo.get(data_item_id)
                if not data_item or not isinstance(data_item, EventDataItem):
                    self._log_error(f"Data item not found or wrong type: {data_item_id}")
                    continue
                
                self._data_items[data_item_id] = data_item
                
                # Use get_events() - works for both layer-based and legacy flat structure
                events = data_item.get_events()
                event_dicts = []
                original_indices = []
                restore_info = []
                
                for event_index in sorted(event_indices, reverse=True):
                    if 0 <= event_index < len(events):
                        event = events[event_index]
                        event_dict = event.to_dict()
                        event_dicts.append(event_dict)
                        original_indices.append(event_index)
                        self._event_objects.append((data_item_id, event))
                        
                        # For layer-based: find layer name and index for undo
                        layer_name = ""
                        index_in_layer = -1
                        if hasattr(data_item, '_layers') and data_item._layers:
                            flat_idx = 0
                            for layer in data_item._layers:
                                for idx, ev in enumerate(layer.events):
                                    if flat_idx == event_index:
                                        layer_name = layer.name
                                        index_in_layer = idx
                                        break
                                    flat_idx += 1
                                if layer_name:
                                    break
                        restore_info.append((event_dict, layer_name, index_in_layer))
                    else:
                        self._log_error(f"Event index {event_index} out of range for {data_item_id} (has {len(events)} events)")
                
                if event_dicts:
                    self._deleted_events[data_item_id] = event_dicts
                    self._original_indices[data_item_id] = list(reversed(original_indices))
                    self._restore_info[data_item_id] = list(reversed(restore_info))
                else:
                    self._log_error(f"No valid events captured for {data_item_id}")
                    
        except Exception as e:
            self._log_error(f"Failed to capture event data: {e}")
            import traceback
            self._log_error(traceback.format_exc())

class CreateEventDataItemCommand(EchoZeroCommand):
    """
    Create a new EventDataItem (undoable).
    
    Redo: Creates the EventDataItem
    Undo: Deletes the EventDataItem
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the block that owns this EventDataItem
        name: Name for the EventDataItem
        metadata: Optional metadata dict
    """
    
    COMMAND_TYPE = "data_item.create"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(facade, f"Create EventDataItem: {name}")
        
        self._block_id = block_id
        self._name = name
        self._metadata = metadata or {}
        self._created_data_item_id: Optional[str] = None
    
    def redo(self):
        """Create the EventDataItem."""
        if not self._facade.data_item_repo:
            self._log_error("Data item repository not available")
            return
        
        from src.shared.domain.entities import EventDataItem
        
        # Create the EventDataItem
        event_data_item = EventDataItem(
            id="",  # Will be generated by repository
            block_id=self._block_id,
            name=self._name,
            type="Event",
            metadata=self._metadata
        )
        
        created_item = self._facade.data_item_repo.create(event_data_item)
        self._created_data_item_id = created_item.id
    
    def undo(self):
        """Delete the created EventDataItem."""
        if self._created_data_item_id and self._facade.data_item_repo:
            try:
                self._facade.data_item_repo.delete(self._created_data_item_id)
            except Exception as e:
                self._log_error(f"Failed to delete EventDataItem: {e}")
    
    @property
    def created_data_item_id(self) -> Optional[str]:
        """Get the ID of the created EventDataItem (after redo)."""
        return self._created_data_item_id

class UpdateEventDataItemCommand(EchoZeroCommand):
    """
    Update an EventDataItem (undoable).
    
    Redo: Applies the new values
    Undo: Restores the original values
    
    Args:
        facade: ApplicationFacade instance
        data_item_id: ID of the EventDataItem
        new_name: Optional new name
        new_metadata: Optional new metadata dict (merged with existing)
    """
    
    COMMAND_TYPE = "data_item.update"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        data_item_id: str,
        new_name: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(facade, "Update EventDataItem")
        
        self._data_item_id = data_item_id
        self._new_name = new_name
        self._new_metadata = new_metadata
        
        # Store original values for undo
        self._old_name: Optional[str] = None
        self._old_metadata: Optional[Dict[str, Any]] = None
    
    def redo(self):
        """Apply the update."""
        if not self._facade.data_item_repo:
            self._log_error("Data item repository not available")
            return
        
        from src.shared.domain.entities import EventDataItem
        
        # Get the EventDataItem
        data_item = self._facade.data_item_repo.get(self._data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            self._log_error(f"EventDataItem not found: {self._data_item_id}")
            return
        
        # Capture original values first time
        if not self._executed:
            self._old_name = data_item.name
            self._old_metadata = data_item.metadata.copy() if data_item.metadata else {}
            self._executed = True
        
        # Apply updates
        if self._new_name is not None:
            data_item.name = self._new_name
        
        if self._new_metadata is not None:
            # Merge with existing metadata
            if data_item.metadata is None:
                data_item.metadata = {}
            data_item.metadata.update(self._new_metadata)
        
        # Save back to repository
        self._facade.data_item_repo.update(data_item)
    
    def undo(self):
        """Restore original values."""
        if self._old_name is None:
            return  # No original state captured
        
        if not self._facade.data_item_repo:
            return
        
        from src.shared.domain.entities import EventDataItem
        
        data_item = self._facade.data_item_repo.get(self._data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            return
        
        # Restore original values
        if self._old_name is not None:
            data_item.name = self._old_name
        
        if self._old_metadata is not None:
            data_item.metadata = self._old_metadata.copy()
        
        self._facade.data_item_repo.update(data_item)

class DeleteEventDataItemCommand(EchoZeroCommand):
    """
    Delete an EventDataItem (undoable).
    
    Redo: Deletes the EventDataItem
    Undo: Recreates the EventDataItem with all original properties
    
    Args:
        facade: ApplicationFacade instance
        data_item_id: ID of the EventDataItem to delete
    """
    
    COMMAND_TYPE = "data_item.delete"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        data_item_id: str
    ):
        super().__init__(facade, "Delete EventDataItem")
        
        self._data_item_id = data_item_id
        self._deleted_data: Optional[Dict[str, Any]] = None
    
    def redo(self):
        """Delete the EventDataItem."""
        if not self._facade.data_item_repo:
            self._log_error("Data item repository not available")
            return
        
        from src.shared.domain.entities import EventDataItem
        
        # Get the EventDataItem to save state for undo
        data_item = self._facade.data_item_repo.get(self._data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            self._log_error(f"EventDataItem not found: {self._data_item_id}")
            return
        
        # Save all state before deletion (first time only)
        if self._deleted_data is None:
            # Get all events for restoration
            events_list = []
            for event in data_item.get_events():
                events_list.append({
                    "time": event.time,
                    "duration": event.duration,
                    "classification": event.classification,
                    "metadata": event.metadata.copy() if event.metadata else {}
                })
            
            self._deleted_data = {
                "id": data_item.id,
                "block_id": data_item.block_id,
                "name": data_item.name,
                "type": data_item.type,
                "metadata": data_item.metadata.copy() if data_item.metadata else {},
                "events": events_list
            }
            
            # Update description with name
            self.setText(f"Delete EventDataItem: {data_item.name}")
        
        # Delete from repository
        self._facade.data_item_repo.delete(self._data_item_id)
    
    def undo(self):
        """Recreate the EventDataItem with original properties."""
        if not self._deleted_data:
            self._log_warning("No data stored, cannot undo")
            return
        
        if not self._facade.data_item_repo:
            return
        
        from src.shared.domain.entities import EventDataItem
        
        # Recreate the EventDataItem
        event_data_item = EventDataItem(
            id=self._deleted_data["id"],
            block_id=self._deleted_data["block_id"],
            name=self._deleted_data["name"],
            type=self._deleted_data["type"],
            metadata=self._deleted_data["metadata"]
        )
        
        # Restore all events
        for event_data in self._deleted_data.get("events", []):
            event_data_item.add_event(
                time=event_data["time"],
                duration=event_data.get("duration", 0.0),
                classification=event_data.get("classification", ""),
                metadata=event_data.get("metadata", {})
            )
        
        # Create in repository
        self._facade.data_item_repo.create(event_data_item)

class AddEventToDataItemCommand(EchoZeroCommand):
    """
    Add an event to an EventDataItem (undoable).
    
    Redo: Creates the event
    Undo: Deletes the created event
    
    Args:
        facade: ApplicationFacade instance
        data_item_id: ID of the EventDataItem
        time: Event time in seconds
        duration: Event duration in seconds
        classification: Event classification
        metadata: Optional event metadata
        layer_name: REQUIRED - Name of the layer to add the event to
    """
    
    COMMAND_TYPE = "data_item.event.create"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        data_item_id: str,
        time: float,
        duration: float = 0.0,
        classification: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        layer_name: str = ""
    ):
        super().__init__(facade, f"Add Event at {time:.2f}s")
        
        self._data_item_id = data_item_id
        self._time = time
        self._duration = duration
        self._classification = classification
        self._metadata = metadata
        self._layer_name = layer_name or classification or "event"  # Fallback for compatibility
        
        # Track created event for undo
        self._created_event_index: Optional[int] = None
    
    def redo(self):
        """Create the event."""
        result = self._facade.add_event_to_data_item(
            data_item_id=self._data_item_id,
            time=self._time,
            duration=self._duration,
            classification=self._classification,
            metadata=self._metadata,
            layer_name=self._layer_name
        )
        
        if result.success and result.data:
            # Store the index for undo
            self._created_event_index = result.data.get("event_index")
    
    def undo(self):
        """Delete the created event."""
        if self._created_event_index is not None:
            self._facade.delete_event_from_data_item(
                self._data_item_id,
                self._created_event_index
            )

class UpdateEventInDataItemCommand(EchoZeroCommand):
    """
    Update an event in an EventDataItem (undoable).
    
    Redo: Applies the new event values
    Undo: Restores the original event values
    
    Args:
        facade: ApplicationFacade instance
        data_item_id: ID of the EventDataItem
        event_index: Index of the event to update
        new_time: Optional new time
        new_duration: Optional new duration
        new_classification: Optional new classification
        new_metadata: Optional new metadata
    """
    
    COMMAND_TYPE = "data_item.event.update"
    MERGE_ID = 3000  # For merging consecutive updates
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        data_item_id: str,
        event_index: int,
        new_time: Optional[float] = None,
        new_duration: Optional[float] = None,
        new_classification: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(facade, "Move Event")
        
        self._data_item_id = data_item_id
        self._event_index = event_index
        
        self._new_time = new_time
        self._new_duration = new_duration
        self._new_classification = new_classification
        self._new_metadata = new_metadata
        
        # Store original values for undo
        self._old_time: Optional[float] = None
        self._old_duration: Optional[float] = None
        self._old_classification: Optional[str] = None
        self._old_metadata: Optional[Dict[str, Any]] = None
    
    def redo(self):
        """Apply the update."""
        # Capture old values first time
        if not self._executed:
            self._capture_old_values()
            self._executed = True
        
        self._apply_update(
            self._new_time,
            self._new_duration,
            self._new_classification,
            self._new_metadata
        )
    
    def undo(self):
        """Restore old values."""
        self._apply_update(
            self._old_time,
            self._old_duration,
            self._old_classification,
            self._old_metadata,
            replace_metadata=True,
        )
    
    def _capture_old_values(self):
        """Capture current event values for undo."""
        try:
            from src.shared.domain.entities import EventDataItem
            
            data_item = self._facade.data_item_repo.get(self._data_item_id)
            if not data_item or not isinstance(data_item, EventDataItem):
                return
            
            events = data_item.get_events()
            if 0 <= self._event_index < len(events):
                event = events[self._event_index]
                self._old_time = event.time
                self._old_duration = event.duration
                self._old_classification = event.classification
                self._old_metadata = event.metadata.copy() if event.metadata else {}
                
        except Exception as e:
            self._log_error(f"Failed to capture old values: {e}")
    
    def _apply_update(
        self,
        time: Optional[float],
        duration: Optional[float],
        classification: Optional[str],
        metadata: Optional[Dict[str, Any]],
        replace_metadata: bool = False,
    ):
        """Apply update to the event."""
        try:
            from src.shared.domain.entities import EventDataItem
            
            data_item = self._facade.data_item_repo.get(self._data_item_id)
            if not data_item or not isinstance(data_item, EventDataItem):
                return
            
            events = data_item.get_events()
            if 0 <= self._event_index < len(events):
                event = events[self._event_index]
                
                if time is not None:
                    event.time = time
                if duration is not None:
                    event.duration = duration
                if classification is not None:
                    event.classification = classification
                if metadata is not None:
                    if replace_metadata:
                        # Undo restores exact original metadata state.
                        event.metadata = dict(metadata)
                    else:
                        # Forward edits merge metadata to avoid dropping other keys.
                        if event.metadata is None:
                            event.metadata = {}
                        event.metadata.update(metadata)
                
                # Save back to database
                self._facade.data_item_repo.update(data_item)
                
        except Exception as e:
            self._log_error(f"Failed to apply update: {e}")
    
    def id(self) -> int:
        """Enable merging."""
        return self.MERGE_ID
    
    def mergeWith(self, other) -> bool:
        """Merge consecutive updates to same event."""
        if not isinstance(other, UpdateEventInDataItemCommand):
            return False
        if other._data_item_id != self._data_item_id:
            return False
        if other._event_index != self._event_index:
            return False
        
        # Take their new values
        if other._new_time is not None:
            self._new_time = other._new_time
        if other._new_duration is not None:
            self._new_duration = other._new_duration
        if other._new_classification is not None:
            self._new_classification = other._new_classification
        if other._new_metadata is not None:
            self._new_metadata = other._new_metadata
        
        return True

class BatchUpdateEventsCommand(EchoZeroCommand):
    """
    Update multiple events in a data item at once (undoable).
    
    This is used when multiple events are moved/resized together,
    or when debounced updates are flushed.
    
    Redo: Applies all new values
    Undo: Restores all original values
    
    Args:
        facade: ApplicationFacade instance
        data_item_id: ID of the EventDataItem
        updates: List of update dicts with event_index and new values
    """
    
    COMMAND_TYPE = "data_item.events.batch_update"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        data_item_id: str,
        updates: List[Dict[str, Any]],
        description: str = "Move Events"
    ):
        count = len(updates)
        super().__init__(facade, f"{description} ({count})" if count > 1 else description)
        
        self._data_item_id = data_item_id
        self._updates = updates  # List of {event_index, time?, duration?, classification?}
        
        # Store original values for undo: {event_index: {time, duration, classification}}
        self._old_values: Dict[int, Dict[str, Any]] = {}
    
    def redo(self):
        """Apply all updates."""
        # Capture old values first time
        if not self._executed:
            self._capture_old_values()
            self._executed = True
        
        self._apply_updates(self._updates)
    
    def undo(self):
        """Restore all old values."""
        from src.utils.message import Log
        
        Log.debug(f"BatchUpdateEventsCommand.undo: data_item={self._data_item_id}")
        Log.debug(f"BatchUpdateEventsCommand.undo: old_values={self._old_values}")
        
        # Convert old_values back to update format
        restore_updates = []
        for event_index, old_data in self._old_values.items():
            restore_updates.append({
                'event_index': event_index,
                'time': old_data.get('time'),
                'duration': old_data.get('duration'),
                'classification': old_data.get('classification'),
                'metadata': old_data.get('metadata'),
            })
            Log.debug(f"BatchUpdateEventsCommand.undo: event {event_index} restore metadata={old_data.get('metadata')}")
        
        # Use replace_metadata=True to restore exact original state
        self._apply_updates(restore_updates, replace_metadata=True)
    
    def _capture_old_values(self):
        """Capture current event values for all events being updated."""
        try:
            from src.shared.domain.entities import EventDataItem
            from src.utils.message import Log
            
            data_item = self._facade.data_item_repo.get(self._data_item_id)
            if not data_item or not isinstance(data_item, EventDataItem):
                Log.debug(f"BatchUpdateEventsCommand._capture_old_values: data_item not found or not EventDataItem")
                return
            
            events = data_item.get_events()
            Log.debug(f"BatchUpdateEventsCommand._capture_old_values: capturing {len(self._updates)} events from {data_item.name}")
            
            for update in self._updates:
                idx = update.get('event_index', -1)
                if 0 <= idx < len(events):
                    event = events[idx]
                    
                    # Start with current metadata
                    old_metadata = dict(event.metadata) if event.metadata else {}
                    
                    # If this update includes layer change info, use the OLD layer name for undo
                    # This ensures undo restores to the correct visual layer
                    if '_old_layer_name' in update:
                        old_layer_name = update['_old_layer_name']
                        if old_layer_name:
                            old_metadata['_visual_layer_name'] = old_layer_name
                        elif '_visual_layer_name' in old_metadata:
                            # If old layer name is None and we had one, remove it
                            # This handles the case where event was originally not on any specific layer
                            del old_metadata['_visual_layer_name']
                        Log.debug(f"BatchUpdateEventsCommand._capture_old_values: event {idx} using _old_layer_name='{old_layer_name}'")
                    
                    self._old_values[idx] = {
                        'time': event.time,
                        'duration': event.duration,
                        'classification': event.classification,
                        'metadata': old_metadata,
                    }
                    Log.debug(f"BatchUpdateEventsCommand._capture_old_values: event {idx} old metadata={self._old_values[idx]['metadata']}")
                    
        except Exception as e:
            self._log_error(f"Failed to capture old values: {e}")
    
    def _apply_updates(self, updates: List[Dict[str, Any]], replace_metadata: bool = False):
        """
        Apply a list of updates to events.
        
        Args:
            updates: List of update dicts
            replace_metadata: If True, replace metadata entirely (for undo).
                              If False, merge metadata (for redo/forward updates).
        """
        try:
            from src.shared.domain.entities import EventDataItem
            
            data_item = self._facade.data_item_repo.get(self._data_item_id)
            if not data_item or not isinstance(data_item, EventDataItem):
                return
            
            events = data_item.get_events()
            for update in updates:
                idx = update.get('event_index', -1)
                if 0 <= idx < len(events):
                    event = events[idx]
                    old_time = event.time
                    
                    if 'time' in update and update['time'] is not None:
                        event.time = update['time']
                    if 'duration' in update and update['duration'] is not None:
                        event.duration = update['duration']
                    if 'classification' in update and update['classification'] is not None:
                        event.classification = update['classification']
                    if 'metadata' in update and update['metadata'] is not None:
                        if isinstance(update['metadata'], dict):
                            if replace_metadata:
                                # Replace entirely - restores exact original state (for undo)
                                event.metadata = dict(update['metadata'])
                            else:
                                # Merge - adds new keys while preserving others (for redo)
                                if event.metadata is None:
                                    event.metadata = {}
                                event.metadata.update(update['metadata'])
            
            # Save back to database
            self._facade.data_item_repo.update(data_item)
            
        except Exception as e:
            self._log_error(f"Failed to apply updates: {e}")

