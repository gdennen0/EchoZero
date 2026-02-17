"""
Event Data Item

Concrete implementation of EventDataItem with event management capabilities.
Supports both legacy flat event structure and new layer-based structure.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from src.shared.domain.entities.data_item import EventDataItem as BaseEventDataItem
from src.utils.message import Log

# Import EventLayer here to avoid circular dependency
try:
    from src.shared.domain.entities import EventLayer
except ImportError:
    # Handle circular import during initial load
    EventLayer = None

class Event:
    """
    Represents a single event.
    
    An event has a stable ID, time, optional duration, classification, and optional metadata.
    The ID is used for matching events across sync operations (MA3 <-> EchoZero).
    """
    def __init__(
        self,
        time: float,
        classification: str = "",
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ):
        """
        Initialize event.
        
        Args:
            time: Event time in seconds
            classification: Event classification (e.g., "kick", "snare")
            duration: Event duration in seconds
            metadata: Optional metadata dictionary
            id: Optional stable identifier (auto-generated if not provided)
        """
        self.id = id if id else str(uuid.uuid4())
        self.time = time
        self.classification = classification
        self.duration = duration
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "time": self.time,
            "classification": self.classification,
            "duration": self.duration,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        """Create from dictionary"""
        # SINGLE SOURCE OF TRUTH: Load metadata from database
        # This is the authoritative source - all metadata including render_as_marker comes from here
        metadata = data.get("metadata", {})
        
        
        return cls(
            id=data.get("id"),  # Will auto-generate if None
            time=data["time"],
            classification=data.get("classification", ""),
            duration=data.get("duration", 0.0),
            metadata=metadata  # SINGLE SOURCE OF TRUTH: Database metadata
        )

class EventDataItem(BaseEventDataItem):
    """
    Event data item with event collection management.
    
    Supports both legacy flat structure and new layer-based structure:
    - Legacy: events stored as flat list (backward compatible)
    - New: events organized into layers (matches timeline visualization)
    
    Handles:
    - Adding/removing events and layers
    - Querying events (flat or by layer)
    - Event serialization/deserialization
    """
    
    def __init__(
        self,
        id: str = "",
        block_id: str = "",
        name: str = "",
        type: str = "Event",
        created_at: Optional[datetime] = None,
        file_path: Optional[str] = None,
        event_count: int = 0,
        metadata: Optional[dict] = None,
        events: Optional[List[Event]] = None,
        layers: Optional[List['EventLayer']] = None
    ):
        """
        Initialize event data item.
        
        Args:
            events: Optional list of Event objects (legacy flat structure)
            layers: Optional list of EventLayer objects (new structure)
            Other args: See base class
            
        Note: If both events and layers are provided, layers take precedence.
              If only events are provided, they're stored in legacy flat format.
        """
        if not id:
            id = str(uuid.uuid4())
        if not created_at:
            created_at = datetime.utcnow()
        if metadata is None:
            metadata = {}
        
        super().__init__(
            id=id,
            block_id=block_id,
            name=name or "EventData",
            type=type,
            created_at=created_at,
            file_path=file_path,
            event_count=event_count,
            metadata=metadata
        )
        
        # EventLayers are REQUIRED - this is the ONLY supported structure
        # Structure: EventDataItem -> EventLayers -> Events
        if layers is not None:
            # Layer-based structure (REQUIRED)
            self._layers = layers
            self._events = []  # Not used - layers are single source of truth
            self.event_count = sum(len(layer.events) for layer in self._layers)
        elif events is not None:
            # DEPRECATED: Legacy flat events - FAIL LOUD
            # Instead of silently converting, warn loudly about the incorrect usage
            # The source block MUST output EventLayers, not flat events
            from src.shared.domain.entities import EventLayer
            from collections import defaultdict
            
            # Log ERROR-level message to make it obvious this is wrong
            Log.error(
                f"EventDataItem.__init__(): DEPRECATED - Received {len(events)} flat events without EventLayers! "
                f"This is NOT the correct structure. "
                f"Structure MUST be: EventDataItem -> EventLayers -> Events. "
                f"The source block should output EventLayers explicitly. "
                f"Converting to layers for compatibility, but this WILL be removed in a future version."
            )
            
            # Group events by classification into layers (temporary compatibility)
            events_by_class = defaultdict(list)
            for event in events:
                layer_name = event.classification or "Unclassified"
                events_by_class[layer_name].append(event)
            
            # Create layers from grouped events
            self._layers = []
            for layer_name, layer_events in events_by_class.items():
                layer = EventLayer(name=layer_name, events=layer_events)
                self._layers.append(layer)
            
            self._events = []  # Clear legacy events
            self.event_count = sum(len(l.events) for l in self._layers)
        else:
            # No events or layers - empty item with empty layers list
            self._layers = []
            self._events = []
            self.event_count = 0
    
    def add_event(
        self,
        time: float,
        classification: str = "",
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        layer_name: str = ""
    ) -> Event:
        """
        Add an event to a specific layer.
        
        IMPORTANT: layer_name is REQUIRED. Events must be explicitly assigned to layers.
        The classification field is stored on the Event but is NOT used to determine layer.
        
        Args:
            time: Event time in seconds
            classification: Event classification (stored on event, not used for layer assignment)
            duration: Event duration in seconds
            metadata: Optional metadata
            layer_name: REQUIRED - Name of the layer to add the event to.
                       If the layer doesn't exist, it will be created.
            
        Returns:
            Created Event object
            
        Raises:
            ValueError: If layer_name is not provided
        """
        # FAIL LOUD: layer_name is REQUIRED
        if not layer_name or not layer_name.strip():
            raise ValueError(
                f"EventDataItem.add_event(): layer_name is REQUIRED. "
                f"Events must be explicitly assigned to layers. "
                f"Event at time={time}s with classification='{classification}' cannot be added "
                f"without specifying a target layer. "
                f"DO NOT use classification to auto-assign layers - be explicit."
            )
        
        layer_name = layer_name.strip()
        event = Event(time, classification, duration, metadata)
        
        # EventLayers are REQUIRED - always use layers
        # Import EventLayer if needed (handle circular import)
        if EventLayer is None:
            from src.shared.domain.entities import EventLayer as EventLayerClass
        else:
            EventLayerClass = EventLayer
        
        # Add to specific layer (create if doesn't exist)
        layer = self.get_layer_by_name(layer_name)
        if not layer:
            layer = EventLayerClass(name=layer_name)
            self._layers.append(layer)
        layer.add_event(event)
        
        self.event_count = sum(len(l.events) for l in self._layers)
        return event
    
    def remove_event(self, event: Event) -> bool:
        """
        Remove an event from the collection.
        
        Args:
            event: Event to remove
            
        Returns:
            True if removed successfully
        """
        # EventLayers are REQUIRED - always use layers
        for layer in self._layers:
            if layer.remove_event(event):
                self.event_count = sum(len(l.events) for l in self._layers)
                return True
        return False
    
    def get_events(self) -> List[Event]:
        """
        Get all events (backward compatible - flattens layers if using layer structure).
        
        Returns:
            List of Event objects (flat)
        """
        # EventLayers are REQUIRED - flatten layers
        all_events = []
        for layer in self._layers:
            all_events.extend(layer.events)
        return all_events
    
    def get_layers(self) -> List['EventLayer']:
        """
        Get all layers.
        
        Structure: EventDataItem -> EventLayers -> Events
        EventLayers are the single source of truth.
        
        Returns:
            List of EventLayer objects
            
        Raises:
            RuntimeError: If legacy flat events are found without layers
        """
        if self._layers:
            return self._layers.copy()
        elif self._events:
            # FAIL LOUD: Legacy flat structure should not exist
            raise RuntimeError(
                f"EventDataItem.get_layers(): INVALID STATE - Found {len(self._events)} flat events "
                f"without EventLayers! Structure MUST be: EventDataItem -> EventLayers -> Events. "
                f"This EventDataItem was created incorrectly. Re-execute the source block."
            )
        else:
            return []
    
    def get_layer_by_name(self, layer_name: str) -> Optional['EventLayer']:
        """
        Get layer by name.
        
        Args:
            layer_name: Layer name to find
            
        Returns:
            EventLayer if found, None otherwise
        """
        layers = self.get_layers()
        for layer in layers:
            if layer.name == layer_name:
                return layer
        return None
    
    def add_layer(self, layer: 'EventLayer') -> None:
        """
        Add a layer to this event data item.
        
        Args:
            layer: EventLayer to add
        """
        # Import EventLayer if needed (handle circular import)
        if EventLayer is None:
            from src.shared.domain.entities import EventLayer as EventLayerClass
        else:
            EventLayerClass = EventLayer
        
        if not isinstance(layer, EventLayerClass):
            raise TypeError(f"Expected EventLayer, got {type(layer)}")
        
        # Convert to layer-based structure if still using legacy flat
        if self._layers is None or len(self._layers) == 0:
            if self._events:
                # Migrate existing events to a default layer
                default_layer = EventLayerClass(name="Default", events=self._events.copy())
                self._layers = [default_layer]
                self._events = []  # Clear legacy structure
            else:
                self._layers = []
        
        # Check if layer with same name already exists
        existing = self.get_layer_by_name(layer.name)
        if existing:
            # Merge events into existing layer
            for event in layer.events:
                existing.add_event(event)
        else:
            # Add new layer
            self._layers.append(layer)
        
        # Update event count
        self.event_count = sum(len(l.events) for l in self._layers)
    
    def remove_layer(self, layer_name: str) -> bool:
        """
        Remove a layer by name, including all events in that layer.
        
        Args:
            layer_name: Name of the layer to remove
            
        Returns:
            True if layer was found and removed, False otherwise
        """
        if not self._layers:
            return False
        
        # Find and remove the layer
        layer_to_remove = None
        for layer in self._layers:
            if layer.name == layer_name:
                layer_to_remove = layer
                break
        
        if layer_to_remove:
            self._layers.remove(layer_to_remove)
            # Update event count
            self.event_count = sum(len(l.events) for l in self._layers)
            return True
        
        return False
    
    def get_events_by_classification(self, classification: str) -> List[Event]:
        """
        Get events filtered by classification.
        
        Args:
            classification: Classification to filter by
            
        Returns:
            List of matching Event objects
        """
        return [e for e in self.get_events() if e.classification == classification]
    
    def get_events_in_range(self, start_time: float, end_time: float) -> List[Event]:
        """
        Get events within a time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of Event objects in range
        """
        return [e for e in self.get_events() if start_time <= e.time <= end_time]
    
    def clear_events(self):
        """Clear all events"""
        if self._layers:
            for layer in self._layers:
                layer.clear_events()
            self._layers.clear()
        else:
            self._events.clear()
        self.event_count = 0
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.
        
        Structure: EventDataItem -> EventLayers -> Events
        EventLayers are the single source of truth.
        
        Raises:
            RuntimeError: If legacy flat events are found without layers
        """
        data = super().to_dict()
        
        # FAIL LOUD: If no layers but has events, this is an invalid state
        if not self._layers and self._events:
            raise RuntimeError(
                f"EventDataItem.to_dict(): INVALID STATE - Found {len(self._events)} flat events "
                f"without EventLayers! Cannot serialize. "
                f"Structure MUST be: EventDataItem -> EventLayers -> Events. "
                f"This EventDataItem was created incorrectly. Re-execute the source block."
            )
        
        # Serialize layers (the single source of truth)
        data["layers"] = [layer.to_dict() for layer in self._layers]
        
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EventDataItem':
        """Create from dictionary - EventLayers are REQUIRED (no backward compatibility)"""
        layers_data = data.get("layers", [])
        events_data = data.get("events", [])  # Only for migration
        
        if layers_data:
            # Layer-based structure (REQUIRED)
            # Import EventLayer if needed (handle circular import)
            try:
                if EventLayer is None:
                    from src.shared.domain.entities import EventLayer as EventLayerClass
                else:
                    EventLayerClass = EventLayer
            except (NameError, ImportError):
                from src.shared.domain.entities import EventLayer as EventLayerClass
            layers = [EventLayerClass.from_dict(l) for l in layers_data]
            event_count = sum(len(l.events) for l in layers)
            
            return cls(
                id=data["id"],
                block_id=data["block_id"],
                name=data["name"],
                type=data.get("type", "Event"),
                created_at=datetime.fromisoformat(data["created_at"]),
                file_path=data.get("file_path"),
                event_count=event_count,
                metadata=data.get("metadata", {}),
                layers=layers
            )
        elif events_data:
            # DEPRECATED: Legacy flat structure - FAIL LOUD
            # This happens when loading old data from DB
            from src.shared.domain.entities import EventLayer
            from collections import defaultdict
            
            # Log ERROR-level message to make it obvious this is wrong
            Log.error(
                f"EventDataItem.from_dict(): DEPRECATED - Loading legacy EventDataItem '{data.get('name', 'unknown')}' "
                f"with {len(events_data)} flat events instead of EventLayers! "
                f"Structure MUST be: EventDataItem -> EventLayers -> Events. "
                f"Re-execute the source block to generate proper EventLayers. "
                f"Converting to layers for compatibility, but this WILL be removed."
            )
            
            # Group events by classification into layers (temporary compatibility)
            events_by_class = defaultdict(list)
            for event_data in events_data:
                event = Event.from_dict(event_data)
                layer_name = event.classification or "Unclassified"
                events_by_class[layer_name].append(event)
            
            # Create layers from grouped events
            layers = []
            for layer_name, layer_events in events_by_class.items():
                layer = EventLayer(name=layer_name, events=layer_events)
                layers.append(layer)
            
            event_count = sum(len(l.events) for l in layers)
            
            return cls(
                id=data["id"],
                block_id=data["block_id"],
                name=data["name"],
                type=data.get("type", "Event"),
                created_at=datetime.fromisoformat(data["created_at"]),
                file_path=data.get("file_path"),
                event_count=event_count,
                metadata=data.get("metadata", {}),
                layers=layers
            )
        else:
            # No events or layers - create empty item with default layer
            from src.shared.domain.entities import EventLayer
            return cls(
                id=data["id"],
                block_id=data["block_id"],
                name=data["name"],
                type=data.get("type", "Event"),
                created_at=datetime.fromisoformat(data["created_at"]),
                file_path=data.get("file_path"),
                event_count=0,
                metadata=data.get("metadata", {}),
                layers=[]  # Empty layers list
            )

