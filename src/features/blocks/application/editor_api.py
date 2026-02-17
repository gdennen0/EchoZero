"""
Editor API

Unified API for Editor block operations with signal integration.

LAYER OPERATIONS
================
- create_layer(name, **opts)        - Create new layer
- update_layer(name, **changes)     - Update layer properties
- delete_layer(name)                - Delete layer
- get_layer(name)                   - Get layer by name
- get_layers()                      - Get all layers
- layer_exists(name)                - Check if layer exists
- get_synced_layers()               - Get layers marked as synced
- get_layer_names()                 - Get list of layer names

Layer Convenience Methods:
- rename_layer(old, new)            - Rename layer
- set_layer_visibility(name, vis)   - Show/hide layer
- set_layer_locked(name, locked)    - Lock/unlock layer
- set_layer_color(name, color)      - Set layer color
- set_layer_height(name, height)    - Set layer height
- mark_layer_synced(name, sm_id)    - Mark layer as synced
- unmark_layer_synced(name)         - Remove sync marking

EVENT OPERATIONS
================
Core single-item methods (batch methods use these):
- add_event(time, class, dur, meta)  - Add single event (core)
- update_event(id, item_id, **chg)   - Update single event (core)
- delete_event_by_index(item, idx)   - Delete by index (core)
- delete_event(id, item_id)          - Delete by ID (uses core)
- move_event(id, item_id, time)      - Move event (uses update_event)

Batch methods (iterate over singular methods):
- add_events(events, source)         - Add multiple events
- delete_events(ids, item_id)        - Delete multiple by ID
- delete_events_by_index(deletions)  - Delete multiple by index
- move_events(events, offset)        - Move multiple by offset
- clear_layer_events(layer)          - Clear all in layer

Query methods:
- get_events(layer_name, source)     - Get events (optional filter)
- get_event(event_id)                - Get single event by ID
- get_events_in_layer(layer)         - Get events in specific layer
- get_event_count(layer)             - Count events
- get_data_item_ids()                - Get EventDataItem IDs

SIGNAL EMISSION
===============
All operations emit signals via SyncSubscriptionService for automatic sync.

Local signals (EditorAPI):
- layer_created(layer_name)
- layer_updated(layer_name, props)
- layer_deleted(layer_name)
- events_added(layer_name, source, events)
- events_updated(layer_name, count)
- events_deleted(layer_name, count)

Usage:
    from src.features.blocks.application.editor_api import EditorAPI
    
    api = EditorAPI(facade, editor_block_id, sync_service)
    
    # Create layer (emits layer_created signal)
    layer = api.create_layer("kicks", color="#ff0000")
    
    # Add events (emits events_added signal)
    api.add_events([{"time": 1.5, "classification": "kicks"}])
    
    # Subscribe to changes
    sync_service.layer_added.connect(on_layer_added)
"""
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.features.show_manager.application.sync_subscription_service import SyncSubscriptionService

@dataclass
class LayerInfo:
    """Information about an Editor layer."""
    name: str
    height: int = 40
    color: Optional[str] = None
    visible: bool = True
    locked: bool = False
    is_synced: bool = False
    show_manager_block_id: Optional[str] = None
    ma3_track_coord: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    event_count: int = 0
    derived_from_ma3: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'name': self.name,
            'height': self.height,
            'color': self.color,
            'visible': self.visible,
            'locked': self.locked,
            'is_synced': self.is_synced,
            'show_manager_block_id': self.show_manager_block_id,
            'ma3_track_coord': self.ma3_track_coord,
            'group_id': self.group_id,
            'group_name': self.group_name,
            'event_count': self.event_count,
        }
        if self.derived_from_ma3:
            result['derived_from_ma3'] = True
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerInfo':
        """Create from dictionary."""
        return cls(
            name=data.get('name', ''),
            height=data.get('height', 40),
            color=data.get('color'),
            visible=data.get('visible', True),
            locked=data.get('locked', False),
            is_synced=data.get('is_synced', False),
            show_manager_block_id=data.get('show_manager_block_id'),
            ma3_track_coord=data.get('ma3_track_coord'),
            group_id=data.get('group_id'),
            group_name=data.get('group_name'),
            event_count=data.get('event_count', 0),
            derived_from_ma3=data.get('derived_from_ma3', False),
        )

@dataclass 
class EventInfo:
    """Information about an Editor event."""
    id: str
    time: float
    duration: float = 0.0
    classification: str = "event"
    layer_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'time': self.time,
            'duration': self.duration,
            'classification': self.classification,
            'layer_id': self.layer_id,
            'metadata': self.metadata,
        }

class EditorAPIError(Exception):
    """Exception raised by EditorAPI operations."""
    pass

class EditorAPI(QObject):
    """
    Unified API for Editor block operations.
    
    Provides a clean interface for layer and event operations
    with automatic signal emission for sync subscribers.
    
    Signals:
        layer_created: Emitted when a layer is created (layer_name)
        layer_updated: Emitted when a layer is updated (layer_name)
        layer_deleted: Emitted when a layer is deleted (layer_name)
        events_added: Emitted when events are added (layer_name, count)
        events_updated: Emitted when events are updated (layer_name, count)
        events_deleted: Emitted when events are deleted (layer_name, count)
    """
    
    # Local signals for direct connections
    layer_created = pyqtSignal(str)  # layer_name
    layer_updated = pyqtSignal(str)  # layer_name
    layer_deleted = pyqtSignal(str)  # layer_name
    events_added = pyqtSignal(str, int)  # layer_name, count
    events_updated = pyqtSignal(str, int)  # layer_name, count
    events_deleted = pyqtSignal(str, int)  # layer_name, count
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        sync_service: Optional["SyncSubscriptionService"] = None,
        timeline_widget: Optional[Any] = None,
        parent: Optional[QObject] = None
    ):
        """
        Initialize EditorAPI.
        
        Args:
            facade: ApplicationFacade instance
            block_id: Editor block ID
            sync_service: Optional SyncSubscriptionService for emitting sync events
            timeline_widget: Optional TimelineWidget for direct visual updates
            parent: Optional parent QObject
        """
        super().__init__(parent)
        
        self._facade = facade
        self._block_id = block_id
        self._sync_service = sync_service
        self._timeline_widget = timeline_widget
        
        Log.debug(f"EditorAPI: Initialized for block {block_id}")
    
    @property
    def block_id(self) -> str:
        """Get the Editor block ID."""
        return self._block_id
    
    # =========================================================================
    # Layer Operations
    # =========================================================================
    
    def create_layer(
        self,
        name: str,
        height: Optional[int] = None,
        color: Optional[str] = None,
        visible: bool = True,
        locked: bool = False,
        is_synced: bool = False,
        show_manager_block_id: Optional[str] = None,
        ma3_track_coord: Optional[str] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None
    ) -> LayerInfo:
        """
        Create a new layer in the Editor.
        
        Args:
            name: Layer name
            height: Layer height in pixels
            color: Layer color (hex string)
            visible: Whether layer is visible
            locked: Whether layer is locked
            is_synced: Whether layer is synced to ShowManager
            show_manager_block_id: ShowManager block ID if synced
            ma3_track_coord: MA3 track coordinate if synced
            group_id: Visual group ID
            group_name: Visual group name
            
        Returns:
            LayerInfo with created layer details
            
        Raises:
            EditorAPIError: If layer creation fails
        """
        from src.application.commands.editor_commands import EditorCreateLayerCommand
        
        properties = {
            'visible': visible,
            'locked': locked,
            'is_synced': is_synced,
        }
        if height is not None:
            properties['height'] = height
        if color is not None:
            properties['color'] = color
        if show_manager_block_id:
            properties['show_manager_block_id'] = show_manager_block_id
        if ma3_track_coord:
            properties['ma3_track_coord'] = ma3_track_coord
        if group_id:
            properties['group_id'] = group_id
        if group_name:
            properties['group_name'] = group_name
        
        cmd = EditorCreateLayerCommand(
            facade=self._facade,
            block_id=self._block_id,
            layer_name=name,
            properties=properties
        )
        self._facade.command_bus.execute(cmd)
        
        created_name = cmd.created_layer_name or name
        
        # Build LayerInfo from the properties we just set
        # Note: We don't use get_layer() because it requires backing EventDataItems
        # which may not exist yet for newly synced MA3 layers
        layer_info = LayerInfo(
            name=created_name,
            height=properties.get('height', 40),
            color=properties.get('color'),
            visible=properties.get('visible', True),
            locked=properties.get('locked', False),
            is_synced=properties.get('is_synced', False),
            show_manager_block_id=properties.get('show_manager_block_id'),
            ma3_track_coord=properties.get('ma3_track_coord'),
            group_id=properties.get('group_id'),
            group_name=properties.get('group_name'),
        )
        
        # Emit signals
        self.layer_created.emit(created_name)
        self._emit_layer_change('added', created_name, layer_info.to_dict())
        
        Log.info(f"EditorAPI: Created layer '{created_name}'")
        return layer_info
    
    def update_layer(
        self,
        name: str,
        new_name: Optional[str] = None,
        height: Optional[int] = None,
        color: Optional[str] = None,
        visible: Optional[bool] = None,
        locked: Optional[bool] = None,
        **kwargs
    ) -> LayerInfo:
        """
        Update layer properties.
        
        Args:
            name: Current layer name
            new_name: New layer name (optional)
            height: New height (optional)
            color: New color (optional)
            visible: New visibility (optional)
            locked: New locked state (optional)
            **kwargs: Additional properties to update
            
        Returns:
            Updated LayerInfo
            
        Raises:
            EditorAPIError: If layer not found or update fails
        """
        from src.application.commands.editor_commands import EditorUpdateLayerCommand
        
        properties = {}
        if new_name is not None:
            properties['name'] = new_name
        if height is not None:
            properties['height'] = height
        if color is not None:
            properties['color'] = color
        if visible is not None:
            properties['visible'] = visible
        if locked is not None:
            properties['locked'] = locked
        properties.update(kwargs)
        
        if not properties:
            # Nothing to update
            layer_info = self.get_layer(name)
            if not layer_info:
                raise EditorAPIError(f"Layer '{name}' not found")
            return layer_info
        
        cmd = EditorUpdateLayerCommand(
            facade=self._facade,
            block_id=self._block_id,
            layer_name=name,
            properties=properties
        )
        self._facade.command_bus.execute(cmd)
        
        # Get updated layer info
        final_name = new_name if new_name else name
        layer_info = self.get_layer(final_name)
        if not layer_info:
            raise EditorAPIError(f"Layer '{final_name}' not found after update")
        
        # Emit signals
        self.layer_updated.emit(final_name)
        self._emit_layer_change('modified', final_name, layer_info.to_dict())
        
        Log.info(f"EditorAPI: Updated layer '{name}'" + (f" -> '{new_name}'" if new_name else ""))
        return layer_info
    
    def delete_layer(self, name: str) -> bool:
        """
        Delete a layer from the Editor.
        
        Args:
            name: Layer name to delete
            
        Returns:
            True if layer was deleted
            
        Raises:
            EditorAPIError: If layer not found or deletion fails
        """
        from src.application.commands.editor_commands import EditorDeleteLayerCommand
        
        # Check if layer exists
        layer_info = self.get_layer(name)
        if not layer_info:
            raise EditorAPIError(f"Layer '{name}' not found")
        
        cmd = EditorDeleteLayerCommand(
            facade=self._facade,
            block_id=self._block_id,
            layer_name=name
        )
        self._facade.command_bus.execute(cmd)
        
        # Emit signals
        self.layer_deleted.emit(name)
        self._emit_layer_change('deleted', name)
        
        Log.info(f"EditorAPI: Deleted layer '{name}'")
        return True
    
    def get_layer(self, name: str) -> Optional[LayerInfo]:
        """
        Get layer information by name.
        
        Args:
            name: Layer name
            
        Returns:
            LayerInfo or None if not found
        """
        layers = self.get_layers()
        for layer in layers:
            if layer.name == name:
                return layer
        return None
    
    def get_layers(self) -> List[LayerInfo]:
        """
        Get all layers in the Editor.
        
        Returns:
            List of LayerInfo objects
        """
        from src.application.commands.editor_commands import EditorGetLayersCommand
        
        cmd = EditorGetLayersCommand(
            facade=self._facade,
            block_id=self._block_id
        )
        self._facade.command_bus.execute(cmd)
        
        layers = []
        for layer_data in cmd.layers:
            layers.append(LayerInfo.from_dict(layer_data))
        
        return layers
    
    def layer_exists(self, name: str) -> bool:
        """
        Check if a layer exists.
        
        Args:
            name: Layer name
            
        Returns:
            True if layer exists
        """
        return self.get_layer(name) is not None
    
    def get_synced_layers(self) -> List[LayerInfo]:
        """
        Get all synced layers.
        
        Returns:
            List of LayerInfo for layers with is_synced=True
        """
        return [layer for layer in self.get_layers() if layer.is_synced]
    
    def rename_layer(self, old_name: str, new_name: str) -> LayerInfo:
        """
        Rename a layer.
        
        Args:
            old_name: Current layer name
            new_name: New layer name
            
        Returns:
            Updated LayerInfo
            
        Raises:
            EditorAPIError: If layer not found or rename fails
        """
        return self.update_layer(old_name, new_name=new_name)
    
    def set_layer_visibility(self, name: str, visible: bool) -> LayerInfo:
        """
        Set layer visibility.
        
        Args:
            name: Layer name
            visible: Whether layer should be visible
            
        Returns:
            Updated LayerInfo
        """
        return self.update_layer(name, visible=visible)
    
    def set_layer_locked(self, name: str, locked: bool) -> LayerInfo:
        """
        Set layer locked state.
        
        Args:
            name: Layer name
            locked: Whether layer should be locked
            
        Returns:
            Updated LayerInfo
        """
        return self.update_layer(name, locked=locked)
    
    def set_layer_color(self, name: str, color: str) -> LayerInfo:
        """
        Set layer color.
        
        Args:
            name: Layer name
            color: Color as hex string (e.g., "#ff0000")
            
        Returns:
            Updated LayerInfo
        """
        return self.update_layer(name, color=color)
    
    def set_layer_height(self, name: str, height: int) -> LayerInfo:
        """
        Set layer height.
        
        Args:
            name: Layer name
            height: Height in pixels
            
        Returns:
            Updated LayerInfo
        """
        return self.update_layer(name, height=height)
    
    def mark_layer_synced(
        self,
        name: str,
        show_manager_block_id: str,
        ma3_track_coord: Optional[str] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None
    ) -> LayerInfo:
        """
        Mark a layer as synced to ShowManager.
        
        Args:
            name: Layer name
            show_manager_block_id: ShowManager block ID
            ma3_track_coord: Optional MA3 track coordinate
            
        Returns:
            Updated LayerInfo
        """
        return self.update_layer(
            name,
            is_synced=True,
            show_manager_block_id=show_manager_block_id,
            ma3_track_coord=ma3_track_coord,
            group_id=group_id,
            group_name=group_name
        )
    
    def unmark_layer_synced(self, name: str) -> LayerInfo:
        """
        Remove sync marking from a layer.
        
        Args:
            name: Layer name
            
        Returns:
            Updated LayerInfo
        """
        return self.update_layer(
            name,
            is_synced=False,
            show_manager_block_id=None,
            ma3_track_coord=None
        )
    
    def get_layer_names(self) -> List[str]:
        """
        Get all layer names.
        
        Returns:
            List of layer names
        """
        return [layer.name for layer in self.get_layers()]
    
    # =========================================================================
    # Event Operations
    # =========================================================================
    
    def add_event(
        self,
        time: float,
        classification: str = "event",
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> bool:
        """
        Add a single event to the Editor.
        
        Args:
            time: Event time in seconds
            classification: Layer name (default "event")
            duration: Event duration (default 0.0)
            metadata: Optional metadata dict
            source: Source identifier (e.g., "ma3", "onset")
            
        Returns:
            True if event was added
            
        Raises:
            EditorAPIError: If adding event fails
        """
        from src.application.commands.editor_commands import EditorAddEventsCommand
        
        event = {
            'time': time,
            'classification': classification,
            'duration': duration,
            'metadata': metadata or {}
        }
        
        cmd = EditorAddEventsCommand(
            facade=self._facade,
            block_id=self._block_id,
            events=[event],
            source=source
        )
        self._facade.command_bus.execute(cmd)
        
        # Emit signals
        self.events_added.emit(classification, 1)
        self._emit_events_change('added', classification, [event])
        
        Log.debug(f"EditorAPI: Added event at {time}s to layer '{classification}'")
        return True
    
    def add_events(
        self,
        events: List[Dict[str, Any]],
        source: Optional[str] = None,
        update_source: Optional[str] = None
    ) -> int:
        """
        Add multiple events to the Editor.
        
        Each event dict should contain:
        - time: float (required)
        - duration: float (optional, default 0.0)
        - classification: str (optional, layer name)
        - metadata: dict (optional)
        
        Note: Uses batch command for atomic undo, but conceptually
        adds each event individually.
        
        Args:
            events: List of event dicts
            source: Source identifier for data item selection (e.g., "ma3", "onset")
            update_source: Optional override for BlockUpdated source
            
        Returns:
            Number of events added
        """
        if not events:
            return 0
        
        import uuid
        
        # Ensure all events have IDs (needed for visual update)
        for event in events:
            if 'id' not in event or not event['id']:
                event['id'] = str(uuid.uuid4())
        
        # For batch operations, use batch command for atomic undo
        from src.application.commands.editor_commands import EditorAddEventsCommand
        
        cmd = EditorAddEventsCommand(
            facade=self._facade,
            block_id=self._block_id,
            events=events,
            source=source
        )
        self._facade.command_bus.execute(cmd)
        
        # Group events by layer for signals
        events_by_layer: Dict[str, List[Dict]] = {}
        for event in events:
            layer_name = event.get('classification', 'default')
            if layer_name not in events_by_layer:
                events_by_layer[layer_name] = []
            events_by_layer[layer_name].append(event)
        
        # Emit signals per layer
        for layer_name, layer_events in events_by_layer.items():
            self.events_added.emit(layer_name, len(layer_events))
            self._emit_events_change('added', layer_name, layer_events)
        
        # Publish BlockUpdated to trigger sync - include layer names that changed
        try:
            from src.application.events.events import BlockUpdated
            event_bus = getattr(self._facade, "event_bus", None)
            if event_bus:
                # Include all layer names that were affected
                affected_layers = list(events_by_layer.keys())
                update_payload = {
                    "id": self._block_id,
                    "events_updated": True,
                    "source": update_source or source or "editor",
                    "layer_names": affected_layers  # List of layer names that changed
                }
                event_bus.publish(BlockUpdated(
                    project_id=self._facade.current_project_id,
                    data=update_payload
                ))
                Log.debug(f"EditorAPI: Published BlockUpdated for add_events ({len(events)} events) in layers: {affected_layers}")
        except Exception as e:
            Log.warning(f"EditorAPI: Failed to publish BlockUpdated: {e}")
        
        # Update visual directly if timeline widget is available
        if self._timeline_widget:
            from ui.qt_gui.widgets.timeline.types import TimelineEvent
            
            for event in events:
                try:
                    timeline_event = TimelineEvent(
                        id=event['id'],
                        time=event.get('time', 0.0),
                        duration=event.get('duration', 0.0),
                        classification=event.get('classification', 'Event'),
                        user_data=event.get('metadata', {})
                    )
                    self._timeline_widget.add_event(timeline_event)
                except Exception as e:
                    Log.debug(f"EditorAPI: Could not add event to timeline: {e}")
            
            Log.debug(f"EditorAPI: Visual update - added {len(events)} events to timeline")
        
        Log.info(f"EditorAPI: Added {len(events)} event(s) to {len(events_by_layer)} layer(s)")
        return len(events)
    
    def get_events(
        self,
        layer_name: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[EventInfo]:
        """
        Get events from the Editor.
        
        Args:
            layer_name: Optional filter by EventLayer name (NOT classification)
            source: Optional filter by source
            
        Returns:
            List of EventInfo objects
        """
        from src.application.commands.editor_commands import EditorGetEventsCommand
        
        # OPTIMIZED: Pass layer_name to command so it gets events directly from that layer
        cmd = EditorGetEventsCommand(
            facade=self._facade,
            block_id=self._block_id,
            layer_name=layer_name,  # Command will get events directly from this layer
            source=source
        )
        self._facade.command_bus.execute(cmd)
        
        # Convert event dicts to EventInfo objects
        # No filtering needed - command already filtered by layer_name if provided
        events = []
        for event_data in cmd.events:
            event_id = event_data.get('id')
            if not event_id:
                raise EditorAPIError("Event missing required id", self._block_id)
            
            # Verify layer_name matches (safety check)
            event_layer_name = event_data.get('layer_name')
            if layer_name and event_layer_name != layer_name:
                continue
            
            events.append(EventInfo(
                id=event_id,
                time=event_data.get('time', 0.0),
                duration=event_data.get('duration', 0.0),
                classification=event_data.get('classification', 'event'),
                layer_id=event_data.get('layer_id'),
                metadata=event_data.get('metadata', {})
            ))
        
        return events
    
    def get_event(self, event_id: str) -> Optional[EventInfo]:
        """
        Get a single event by ID.
        
        Args:
            event_id: Event ID to find
            
        Returns:
            EventInfo or None if not found
        """
        events = self.get_events()
        for event in events:
            if event.id == event_id:
                return event
        return None
    
    def get_event_count(self, layer_name: Optional[str] = None) -> int:
        """
        Get event count, optionally filtered by layer.
        
        Args:
            layer_name: Optional layer name filter
            
        Returns:
            Number of events
        """
        return len(self.get_events(layer_name=layer_name))
    
    def get_events_in_layer(self, layer_name: str) -> List[EventInfo]:
        """
        Get all events in a specific EventLayer.
        
        Args:
            layer_name: EventLayer name (NOT classification)
            
        Returns:
            List of EventInfo objects in the layer
        """
        return self.get_events(layer_name=layer_name)
    
    def update_event(
        self,
        event_id: str,
        data_item_id: str,
        time: Optional[float] = None,
        duration: Optional[float] = None,
        classification: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing event.
        
        Args:
            event_id: Event ID to update
            data_item_id: EventDataItem ID containing the event
            time: New time (optional)
            duration: New duration (optional)
            classification: New classification/layer (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if event was updated
            
        Raises:
            EditorAPIError: If event not found or update fails
        """
        from src.application.commands.data_item_commands import UpdateEventInDataItemCommand
        from src.shared.domain.entities import EventDataItem
        
        # Check if there's anything to update
        if time is None and duration is None and classification is None and metadata is None:
            return True  # Nothing to update
        
        # Find the event's index in the data item
        data_item = self._facade.data_item_repo.get(data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            raise EditorAPIError(f"Data item not found: {data_item_id}")
        
        events = data_item.get_events()
        event_index = None
        for idx, evt in enumerate(events):
            if getattr(evt, 'id', None) == event_id:
                event_index = idx
                break
        
        if event_index is None:
            raise EditorAPIError(f"Event not found: {event_id} in data item {data_item_id}")
        
        cmd = UpdateEventInDataItemCommand(
            facade=self._facade,
            data_item_id=data_item_id,
            event_index=event_index,
            new_time=time,
            new_duration=duration,
            new_classification=classification,
            new_metadata=metadata
        )
        self._facade.command_bus.execute(cmd)
        
        # Emit signals
        layer_name = classification or 'default'
        self.events_updated.emit(layer_name, 1)
        updates = {}
        if time is not None:
            updates['time'] = time
        if duration is not None:
            updates['duration'] = duration
        if classification is not None:
            updates['classification'] = classification
        if metadata is not None:
            updates['metadata'] = metadata
        self._emit_events_change('modified', layer_name, [{'id': event_id, **updates}])
        
        # Publish BlockUpdated to trigger sync - include layer name that changed
        try:
            from src.application.events.events import BlockUpdated
            event_bus = getattr(self._facade, "event_bus", None)
            if event_bus:
                update_payload = {
                    "id": self._block_id,
                    "events_updated": True,
                    "source": "editor",
                    "layer_names": [layer_name]  # Single layer that changed
                }
                event_bus.publish(BlockUpdated(
                    project_id=self._facade.current_project_id,
                    data=update_payload
                ))
                Log.debug(f"EditorAPI: Published BlockUpdated for update_event in layer: {layer_name}")
        except Exception as e:
            Log.warning(f"EditorAPI: Failed to publish BlockUpdated: {e}")
        
        Log.info(f"EditorAPI: Updated event {event_id}")
        return True
    
    def delete_event_by_index(
        self,
        data_item_id: str,
        event_index: int,
        layer_name: Optional[str] = None
    ) -> bool:
        """
        Delete a single event by index (core deletion method).
        
        This is the fundamental deletion operation that all other
        delete methods build upon.
        
        Args:
            data_item_id: EventDataItem ID containing the event
            event_index: Index of event in the data item
            layer_name: Optional layer name for signal emission
            
        Returns:
            True if event was deleted
        """
        from src.application.commands.data_item_commands import BatchDeleteEventsFromDataItemCommand
        
        cmd = BatchDeleteEventsFromDataItemCommand(
            facade=self._facade,
            deletions=[(data_item_id, event_index)]
        )
        self._facade.command_bus.execute(cmd)
        
        # Emit signals
        layer = layer_name or 'default'
        self.events_deleted.emit(layer, 1)
        self._emit_events_change('deleted', layer, [{'index': event_index}])
        
        # Publish BlockUpdated to trigger sync - include layer name that changed
        try:
            from src.application.events.events import BlockUpdated
            event_bus = getattr(self._facade, "event_bus", None)
            if event_bus:
                update_payload = {
                    "id": self._block_id,
                    "events_updated": True,
                    "source": "editor",
                    "layer_names": [layer] if layer else []  # Layer that changed (if known)
                }
                event_bus.publish(BlockUpdated(
                    project_id=self._facade.current_project_id,
                    data=update_payload
                ))
                Log.debug(f"EditorAPI: Published BlockUpdated for delete_event_by_index in layer: {layer}")
        except Exception as e:
            Log.warning(f"EditorAPI: Failed to publish BlockUpdated: {e}")
        
        Log.debug(f"EditorAPI: Deleted event at index {event_index}")
        return True
    
    def delete_event(
        self,
        event_id: str,
        data_item_id: str,
        layer_name: Optional[str] = None
    ) -> bool:
        """
        Delete a single event by ID.
        
        Finds the event index and calls delete_event_by_index.
        
        Args:
            event_id: Event ID to delete
            data_item_id: EventDataItem ID containing the event
            layer_name: Optional layer name for signal emission
            
        Returns:
            True if event was deleted
        """
        from src.shared.domain.entities import EventDataItem
        
        if not self._facade.data_item_repo:
            raise EditorAPIError("Data item repository not available", self._block_id)
        
        data_item = self._facade.data_item_repo.get(data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            raise EditorAPIError(f"EventDataItem not found: {data_item_id}", self._block_id)
        
        # Find event index
        events = data_item.get_events()
        for idx, event in enumerate(events):
            eid = getattr(event, 'id', None) or (event.get('id') if isinstance(event, dict) else None)
            if eid == event_id:
                return self.delete_event_by_index(data_item_id, idx, layer_name)
        
        Log.warning(f"EditorAPI: Event not found: {event_id}")
        return False
    
    def delete_events(
        self,
        event_ids: List[str],
        data_item_id: str,
        layer_name: Optional[str] = None
    ) -> int:
        """
        Delete multiple events by ID.
        
        Iterates through event_ids and calls delete_event for each.
        
        Note: For atomic undo of batch deletions, consider using
        delete_events_by_index with a pre-built deletions list.
        
        Args:
            event_ids: List of event IDs to delete
            data_item_id: EventDataItem ID containing the events
            layer_name: Optional layer name for signal emission
            
        Returns:
            Number of events deleted
        """
        if not event_ids:
            return 0
        
        deleted = 0
        for event_id in event_ids:
            if self.delete_event(event_id, data_item_id, layer_name):
                deleted += 1
        
        Log.info(f"EditorAPI: Deleted {deleted}/{len(event_ids)} event(s)")
        return deleted
    
    def delete_events_by_index(
        self,
        deletions: List[tuple],
        layer_name: Optional[str] = None
    ) -> int:
        """
        Delete multiple events by (data_item_id, event_index) tuples.
        
        Uses batch command for atomic undo of all deletions.
        Iterates through deletions calling delete_event_by_index.
        
        Args:
            deletions: List of (data_item_id, event_index) tuples
            layer_name: Optional layer name for signal emission
            
        Returns:
            Number of events deleted
        """
        if not deletions:
            return 0
        
        deleted = 0
        for data_item_id, event_index in deletions:
            if self.delete_event_by_index(data_item_id, event_index, layer_name):
                deleted += 1
        
        Log.info(f"EditorAPI: Deleted {deleted} event(s) by index")
        return deleted
    
    def move_event(
        self,
        event_id: str,
        data_item_id: str,
        new_time: float,
        layer_name: Optional[str] = None
    ) -> bool:
        """
        Move an event to a new time.
        
        Args:
            event_id: Event ID to move
            data_item_id: EventDataItem ID containing the event
            new_time: New time in seconds
            layer_name: Optional layer name for signal emission
            
        Returns:
            True if event was moved
        """
        # Update repository (for persistence)
        result = self.update_event(
            event_id=event_id,
            data_item_id=data_item_id,
            time=new_time
        )
        
        # Update visual directly (for immediate feedback)
        if result and self._timeline_widget:
            self._timeline_widget.update_event(event_id, start_time=new_time)
            Log.debug(f"EditorAPI: Visual update for event {event_id} to {new_time}s")
        
        return result
    
    def move_events(
        self,
        events: List[Dict[str, Any]],
        data_item_id: str,
        time_offset: float
    ) -> int:
        """
        Move multiple events by a time offset.
        
        Iterates through events and calls move_event for each.
        
        Each event dict should have:
        - id: str (event ID)
        - time: float (current time)
        
        Args:
            events: List of event dicts with id and current time
            data_item_id: EventDataItem ID
            time_offset: Time offset to add (positive = forward, negative = backward)
            
        Returns:
            Number of events moved
        """
        if not events:
            return 0
        
        moved = 0
        for event in events:
            event_id = event.get('id')
            current_time = event.get('time', 0.0)
            if event_id:
                new_time = current_time + time_offset
                if self.move_event(event_id, data_item_id, new_time):
                    moved += 1
        
        Log.info(f"EditorAPI: Moved {moved}/{len(events)} event(s) by {time_offset}s")
        return moved
    
    def clear_layer_events(self, layer_name: str) -> int:
        """
        Clear all events from a layer.
        
        Args:
            layer_name: Layer name (classification)
            
        Returns:
            Number of events cleared
        """
        # Get all events in layer
        events = self.get_events_in_layer(layer_name)
        
        if not events:
            return 0
        
        # Group by data_item_id and delete
        from src.shared.domain.entities import EventDataItem
        
        if not self._facade.data_item_repo:
            raise EditorAPIError("Data item repository not available")
        
        items = self._facade.data_item_repo.list_by_block(self._block_id)
        event_items = [item for item in items if isinstance(item, EventDataItem)]
        
        total_deleted = 0
        for item in event_items:
            # Get events in this layer from this item
            layer_events = [e for e in item.get_events() if e.classification == layer_name]
            if layer_events:
                event_ids = [e.id for e in layer_events if hasattr(e, 'id')]
                if event_ids:
                    self.delete_events(event_ids, item.id, layer_name)
                    total_deleted += len(event_ids)
        
        Log.info(f"EditorAPI: Cleared {total_deleted} event(s) from layer '{layer_name}'")
        return total_deleted

    def apply_layer_snapshot(
        self,
        layer_name: str,
        events: List[Dict[str, Any]],
        source: Optional[str] = None,
        update_source: Optional[str] = None,
        data_item_id: Optional[str] = None
    ) -> int:
        """
        Replace all events in a layer with a full snapshot (atomic).
        
        Args:
            layer_name: Target layer name
            events: List of event dicts
            source: Source identifier for data item selection
            update_source: Optional override for BlockUpdated source
            data_item_id: Optional explicit EventDataItem ID
        
        Returns:
            Number of events applied
        """
        if not self._facade.data_item_repo:
            raise EditorAPIError("Data item repository not available")
        
        # Resolve data item
        if not data_item_id:
            data_item_ids = self.get_data_item_ids()
            if not data_item_ids:
                Log.warning("EditorAPI: No data items found for apply_layer_snapshot")
                return 0
            if source:
                from src.shared.domain.entities import EventDataItem
                for item in self._facade.data_item_repo.list_by_block(self._block_id):
                    if isinstance(item, EventDataItem) and item.metadata.get("source") == source:
                        data_item_id = item.id
                        break
            if not data_item_id:
                data_item_id = data_item_ids[0]
        
        # Ensure layer exists in UI state
        layer_result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        existing_layer_names = set()
        if layer_result.success and layer_result.data:
            existing_layers = layer_result.data.get('layers', [])
            existing_layer_names = {layer.get('name') for layer in existing_layers if layer.get('name')}
        if layer_name not in existing_layer_names:
            from src.application.commands.editor_commands import EditorCreateLayerCommand
            synced = False
            show_manager_block_id = None
            ma3_track_coord = None
            group_id = None
            group_name = None
            for evt in events or []:
                metadata = evt.get("metadata") or {}
                if metadata.get("_synced_from_ma3"):
                    synced = True
                show_manager_block_id = show_manager_block_id or metadata.get("_show_manager_block_id") or metadata.get("show_manager_block_id")
                ma3_track_coord = ma3_track_coord or metadata.get("_ma3_track_coord") or metadata.get("ma3_track_coord") or metadata.get("ma3_coord")
            if data_item_id:
                from src.shared.domain.entities import EventDataItem
                item = self._facade.data_item_repo.get(data_item_id)
                if item and isinstance(item, EventDataItem):
                    meta = item.metadata or {}
                    group_id = meta.get("group_id")
                    group_name = meta.get("group_name") or item.name
            create_cmd = EditorCreateLayerCommand(
                facade=self._facade,
                block_id=self._block_id,
                layer_name=layer_name,
                properties={
                    "is_synced": synced,
                    "show_manager_block_id": show_manager_block_id,
                    "ma3_track_coord": ma3_track_coord,
                    "group_id": group_id,
                    "group_name": group_name
                }
            )
            self._facade.command_bus.execute(create_cmd)
        
        from src.application.commands.editor_commands import ApplyLayerSnapshotCommand
        cmd = ApplyLayerSnapshotCommand(
            facade=self._facade,
            data_item_id=data_item_id,
            layer_name=layer_name,
            events=events or []
        )
        self._facade.command_bus.execute(cmd)
        
        # Publish BlockUpdated to trigger sync and UI refresh
        try:
            from src.application.events.events import BlockUpdated
            event_bus = getattr(self._facade, "event_bus", None)
            if event_bus:
                update_payload = {
                    "id": self._block_id,
                    "events_updated": True,
                    "source": update_source or source or "editor",
                    "layer_names": [layer_name]
                }
                event_bus.publish(BlockUpdated(
                    project_id=self._facade.current_project_id,
                    data=update_payload
                ))
        except Exception as e:
            Log.warning(f"EditorAPI: Failed to publish BlockUpdated for apply_layer_snapshot: {e}")
        
        return len(events or [])
    
    def sync_layer(
        self,
        layer_name: str,
        to_add: List[Dict[str, Any]],
        to_delete: List[str],
        to_update: List[Dict[str, Any]],
        source: str = "sync"
    ) -> Dict[str, int]:
        """
        Apply sync changes to a layer.
        
        This is the unified method for applying sync operations from external
        sources (like MA3) to a layer. It handles all three types of changes
        in a single operation:
        
        1. Delete events by ID
        2. Update events by ID
        3. Add new events
        
        Args:
            layer_name: Layer name (classification)
            to_add: List of event dicts to add (each should have 'id', 'time', etc.)
            to_delete: List of event IDs to delete
            to_update: List of event dicts to update (each should have 'id' and updated fields)
            source: Source identifier for the sync operation
            
        Returns:
            Dict with counts: {'added': int, 'deleted': int, 'updated': int, 'errors': int}
        """
        result = {'added': 0, 'deleted': 0, 'updated': 0, 'errors': 0}
        
        # Get the data item ID for this layer
        data_item_ids = self.get_data_item_ids()
        if not data_item_ids:
            Log.warning(f"EditorAPI: No data items found for sync_layer")
            return result
        
        # Use the first data item (or find the one for ma3_sync source)
        from src.shared.domain.entities import EventDataItem
        data_item_id = None
        
        if self._facade.data_item_repo:
            items = self._facade.data_item_repo.list_by_block(self._block_id)
            for item in items:
                if isinstance(item, EventDataItem):
                    if item.metadata and item.metadata.get('source') == source:
                        data_item_id = item.id
                        break
            if not data_item_id and data_item_ids:
                data_item_id = data_item_ids[0]
        
        if not data_item_id:
            Log.warning(f"EditorAPI: No suitable data item for sync_layer")
            return result
        
        # 1. Delete events
        for event_id in to_delete:
            try:
                if self.delete_event(event_id, data_item_id, layer_name):
                    result['deleted'] += 1
                else:
                    result['errors'] += 1
            except Exception as e:
                Log.warning(f"EditorAPI: Failed to delete event {event_id}: {e}")
                result['errors'] += 1
        
        # 2. Update events
        for event_data in to_update:
            try:
                event_id = event_data.get('id')
                if event_id:
                    success = self.update_event(
                        event_id=event_id,
                        data_item_id=data_item_id,
                        time=event_data.get('time'),
                        duration=event_data.get('duration'),
                        classification=event_data.get('classification'),
                        metadata=event_data.get('metadata')
                    )
                    if success:
                        result['updated'] += 1
                    else:
                        result['errors'] += 1
                else:
                    Log.warning(f"EditorAPI: Update event missing ID")
                    result['errors'] += 1
            except Exception as e:
                Log.warning(f"EditorAPI: Failed to update event: {e}")
                result['errors'] += 1
        
        # 3. Add events
        if to_add:
            # Ensure all events have the correct classification
            for event in to_add:
                if 'classification' not in event:
                    event['classification'] = layer_name
            
            added = self.add_events(to_add, source=source)
            result['added'] = added
        
        Log.info(
            f"EditorAPI: sync_layer '{layer_name}' - "
            f"added {result['added']}, deleted {result['deleted']}, "
            f"updated {result['updated']}, errors {result['errors']}"
        )
        
        return result
    
    def sync_layer_replace(
        self,
        layer_name: str,
        events: List[Dict[str, Any]],
        source: str = "sync"
    ) -> Dict[str, int]:
        """
        Replace all events in a layer with provided events.
        
        This is a convenience method for full sync operations where the
        external source (like MA3) is the source of truth.
        
        ATOMIC OPERATION: For ma3_sync source, directly replaces events in the
        ma3_sync EventDataItem to avoid race conditions from individual deletes.
        
        Args:
            layer_name: Layer name (classification)
            events: Complete list of events that should exist in the layer
            source: Source identifier for the sync operation
            
        Returns:
            Dict with counts: {'added': int, 'deleted': int, 'cleared': int}
        """
        result = {'added': 0, 'deleted': 0, 'cleared': 0}
        
        # For ma3_sync source, use atomic replace to avoid race conditions
        if source == "ma3_sync":
            result = self._atomic_layer_replace(layer_name, events, source)
        else:
            # Standard path for other sources
            result['cleared'] = self.clear_layer_events(layer_name)
            
            if events:
                for event in events:
                    if 'classification' not in event:
                        event['classification'] = layer_name
                
                result['added'] = self.add_events(events, source=source)
        
        Log.info(
            f"EditorAPI: sync_layer_replace '{layer_name}' - "
            f"cleared {result['cleared']}, added {result['added']}"
        )
        
        return result
    
    def _atomic_layer_replace(
        self,
        layer_name: str,
        events: List[Dict[str, Any]],
        source: str
    ) -> Dict[str, int]:
        """
        Atomically replace all events in a layer for a specific source.
        
        This directly manipulates the EventDataItem to avoid race conditions
        that can occur with individual delete/add commands during MA3 sync.
        
        Args:
            layer_name: Layer name (classification)
            events: Complete list of events that should exist in the layer
            source: Source identifier (e.g., "ma3_sync")
            
        Returns:
            Dict with counts: {'added': int, 'deleted': int, 'cleared': int}
        """
        from src.shared.domain.entities import EventDataItem, EventLayer
        from src.shared.domain.entities.event_data_item import Event
        import uuid
        
        result = {'added': 0, 'deleted': 0, 'cleared': 0}
        
        if not self._facade.data_item_repo:
            raise EditorAPIError("Data item repository not available", self._block_id)
        
        # Find the ma3_sync EventDataItem for this block
        items = self._facade.data_item_repo.list_by_block(self._block_id)
        sync_item = None
        
        for item in items:
            if isinstance(item, EventDataItem) and 'ma3_sync' in item.name:
                sync_item = item
                break
        
        if not sync_item:
            # No existing ma3_sync item - just add events normally
            if events:
                for event in events:
                    if 'classification' not in event:
                        event['classification'] = layer_name
                result['added'] = self.add_events(events, source=source)
            return result

        show_manager_block_id = None
        ma3_track_coord = None
        # Validate MA3 sync metadata for ma3_sync source (no backfill)
        if source == "ma3_sync":
            missing_meta = False
            for evt_dict in events:
                metadata = evt_dict.get("metadata") or {}
                if metadata.get("_synced_from_ma3") is not True:
                    missing_meta = True
                if not show_manager_block_id:
                    show_manager_block_id = (
                        metadata.get("_show_manager_block_id")
                        or metadata.get("show_manager_block_id")
                    )
                if not ma3_track_coord:
                    ma3_track_coord = (
                        metadata.get("_ma3_track_coord")
                        or metadata.get("ma3_track_coord")
                        or metadata.get("ma3_coord")
                    )
                if show_manager_block_id and ma3_track_coord and not missing_meta:
                    break

            if missing_meta or not show_manager_block_id or not ma3_track_coord:
                raise EditorAPIError(
                    "MA3 sync events missing required metadata (show_manager_block_id, ma3_track_coord, _synced_from_ma3)",
                    self._block_id
                )

        if not sync_item.metadata.get("group_name"):
            for evt_dict in events:
                metadata = evt_dict.get("metadata") or {}
                coord = metadata.get("_ma3_track_coord") or metadata.get("ma3_track_coord") or metadata.get("ma3_coord")
                if isinstance(coord, str):
                    coord_lower = coord.lower()
                    if coord_lower.startswith("tc"):
                        num = ""
                        for ch in coord_lower[2:]:
                            if ch.isdigit():
                                num += ch
                            else:
                                break
                        if num:
                            sync_item.metadata["group_name"] = f"TC {num}"
                            if not sync_item.metadata.get("group_id"):
                                sync_item.metadata["group_id"] = f"tc_{num}"
                            break
                    elif coord_lower and coord_lower[0].isdigit():
                        head = coord_lower.split(".", 1)[0]
                        if head.isdigit():
                            sync_item.metadata["group_name"] = f"TC {head}"
                            if not sync_item.metadata.get("group_id"):
                                sync_item.metadata["group_id"] = f"tc_{head}"
                            break
            if sync_item.metadata.get("group_name"):
                self._facade.data_item_repo.update(sync_item)
        
        # Find or create the target layer
        target_layer = sync_item.get_layer_by_name(layer_name)
        
        # SINGLE SOURCE OF TRUTH: Capture existing event metadata BEFORE clearing
        # This preserves user-set fields like render_as_marker during MA3 sync
        existing_events_by_ma3_idx = {}
        existing_events_by_id = {}
        if target_layer and hasattr(target_layer, 'events'):
            result['cleared'] = len(target_layer.events)
            # Index existing events by _ma3_idx (for MA3 sync matching) and by ID
            for existing_event in target_layer.events:
                existing_metadata = getattr(existing_event, 'metadata', {}) or {}
                ma3_idx = existing_metadata.get('_ma3_idx')
                if ma3_idx is not None:
                    existing_events_by_ma3_idx[ma3_idx] = existing_event
                existing_events_by_id[existing_event.id] = existing_event
            # Clear the layer's events
            target_layer.clear_events()
        else:
            # Create new layer
            target_layer = EventLayer(name=layer_name, events=[])
            sync_item._layers.append(target_layer)
            result['cleared'] = 0
        
        # Create new Event objects and add to the layer
        # SINGLE SOURCE OF TRUTH: Preserve existing event metadata (including render_as_marker)
        # when replacing events during sync operations
        for evt_dict in events:
            event_id = evt_dict.get('id') or str(uuid.uuid4())
            new_metadata = evt_dict.get('metadata', {})
            
            # Try to find existing event by _ma3_idx first (MA3 sync matching)
            ma3_idx = new_metadata.get('_ma3_idx')
            existing_event = None
            if ma3_idx is not None and ma3_idx in existing_events_by_ma3_idx:
                existing_event = existing_events_by_ma3_idx[ma3_idx]
            elif event_id in existing_events_by_id:
                existing_event = existing_events_by_id[event_id]
            
            
            # Preserve existing metadata (including render_as_marker) from database
            # Merge new metadata into existing to preserve user-set fields
            if existing_event:
                existing_metadata = getattr(existing_event, 'metadata', {}) or {}
                # Merge: existing metadata first (preserves render_as_marker), then new metadata (sync fields)
                merged_metadata = dict(existing_metadata)
                merged_metadata.update(new_metadata)
                new_metadata = merged_metadata
                # Preserve the existing event ID to maintain database consistency
                event_id = existing_event.id
                
            
            new_event = Event(
                time=evt_dict.get('time', 0.0),
                duration=evt_dict.get('duration', 0.0),
                classification=evt_dict.get('classification', layer_name),
                metadata=new_metadata,  # Merged metadata preserving render_as_marker
                id=event_id
            )
            target_layer.add_event(new_event)
        
        # Update event count
        sync_item.event_count = sum(len(l.events) for l in sync_item._layers)
        
        # Persist the changes
        self._facade.data_item_repo.update(sync_item)

        if source == "ma3_sync" and show_manager_block_id and ma3_track_coord:
            try:
                self.mark_layer_synced(
                    layer_name,
                    show_manager_block_id,
                    ma3_track_coord,
                    group_id=sync_item.id,
                    group_name=sync_item.name,
                )
            except Exception:
                Log.warning(
                    f"EditorAPI: Failed to mark layer synced for '{layer_name}' "
                    f"(show_manager_block_id={show_manager_block_id}, ma3_track_coord={ma3_track_coord})"
                )
        
        result['added'] = len(events)
        
        Log.debug(
            f"EditorAPI: _atomic_layer_replace '{layer_name}' - "
            f"cleared {result['cleared']}, added {result['added']}, "
            f"total events in data item: {sync_item.event_count}"
        )
        
        # Emit signals
        self.events_deleted.emit(layer_name, result['cleared'])
        self.events_added.emit(layer_name, result['added'])
        
        # Publish BlockUpdated to trigger EditorPanel refresh
        try:
            from src.application.events.events import BlockUpdated
            event_bus = getattr(self._facade, "event_bus", None)
            if not event_bus:
                raise RuntimeError("Event bus unavailable on facade")
            update_payload = {
                "id": self._block_id,
                "events_updated": True,
                "source": source
            }
            event_bus.publish(BlockUpdated(
                project_id=self._facade.current_project_id,
                data=update_payload
            ))
            Log.debug(f"EditorAPI: Published BlockUpdated for atomic layer replace")
        except ImportError:
            pass
        except Exception as e:
            Log.warning(f"EditorAPI: Failed to publish BlockUpdated: {e}")
        
        return result
    
    def get_data_item_ids(self) -> List[str]:
        """
        Get all EventDataItem IDs for this Editor block.
        
        Returns:
            List of data item IDs
        """
        from src.shared.domain.entities import EventDataItem
        
        if not self._facade.data_item_repo:
            return []
        
        items = self._facade.data_item_repo.list_by_block(self._block_id)
        return [item.id for item in items if isinstance(item, EventDataItem)]
    
    def find_data_item_for_event(self, event_id: str) -> Optional[str]:
        """
        Find the data item ID that contains a specific event.
        
        Args:
            event_id: The event ID to search for
            
        Returns:
            Data item ID containing the event, or None if not found
        """
        from src.shared.domain.entities import EventDataItem
        
        if not self._facade.data_item_repo:
            return None
        
        items = self._facade.data_item_repo.list_by_block(self._block_id)
        for item in items:
            if not isinstance(item, EventDataItem):
                continue
            for evt in item.get_events():
                if getattr(evt, 'id', None) == event_id:
                    return item.id
        
        return None
    
    # =========================================================================
    # Signal Emission (for SyncSubscriptionService)
    # =========================================================================
    
    def _emit_layer_change(
        self,
        change_type: str,
        layer_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit layer change to SyncSubscriptionService.
        
        Args:
            change_type: 'added', 'modified', or 'deleted'
            layer_name: Layer name
            data: Optional additional data
        """
        if not self._sync_service:
            return
        
        from src.features.show_manager.application.sync_subscription_service import (
            LayerChangeEvent, ChangeType, SourceType
        )
        
        ct_map = {
            'added': ChangeType.ADDED,
            'modified': ChangeType.MODIFIED,
            'deleted': ChangeType.DELETED,
        }
        
        event = LayerChangeEvent(
            source=SourceType.EDITOR,
            change_type=ct_map.get(change_type, ChangeType.MODIFIED),
            layer_id=layer_name,
            block_id=self._block_id,
            data=data or {}
        )
        
        self._sync_service.emit_layer_change(event)
    
    def _emit_events_change(
        self,
        change_type: str,
        layer_name: str,
        events: List[Dict[str, Any]]
    ) -> None:
        """
        Emit events change to SyncSubscriptionService.
        
        Args:
            change_type: 'added', 'modified', or 'deleted'
            layer_name: Layer name
            events: List of event dicts
        """
        if not self._sync_service:
            return
        
        from src.features.show_manager.application.sync_subscription_service import (
            EventChangeEvent, ChangeType, SourceType
        )
        
        ct_map = {
            'added': ChangeType.ADDED,
            'modified': ChangeType.MODIFIED,
            'deleted': ChangeType.DELETED,
        }
        
        event_ids = [e.get('id', '') for e in events]
        
        event = EventChangeEvent(
            source=SourceType.EDITOR,
            change_type=ct_map.get(change_type, ChangeType.MODIFIED),
            layer_id=layer_name,
            block_id=self._block_id,
            event_ids=event_ids,
            events=events
        )
        
        self._sync_service.emit_events_change(event)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def set_sync_service(self, sync_service: "SyncSubscriptionService") -> None:
        """
        Set or update the SyncSubscriptionService.
        
        Args:
            sync_service: SyncSubscriptionService instance
        """
        self._sync_service = sync_service
    
    def get_block_info(self) -> Dict[str, Any]:
        """
        Get Editor block information.
        
        Returns:
            Dict with block info (id, name, type, etc.)
        """
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return {'id': self._block_id, 'type': 'Editor'}
        
        block = result.data
        return {
            'id': block.id,
            'name': block.name,
            'type': block.type,
            'x': block.x,
            'y': block.y,
        }

# Factory function for convenience
def create_editor_api(
    facade: "ApplicationFacade",
    block_id: str,
    sync_service: Optional["SyncSubscriptionService"] = None,
    timeline_widget: Optional[Any] = None
) -> EditorAPI:
    """
    Create an EditorAPI instance.
    
    Args:
        facade: ApplicationFacade instance
        block_id: Editor block ID
        sync_service: Optional SyncSubscriptionService
        timeline_widget: Optional TimelineWidget for direct visual updates
        
    Returns:
        EditorAPI instance
    """
    return EditorAPI(facade, block_id, sync_service, timeline_widget)
