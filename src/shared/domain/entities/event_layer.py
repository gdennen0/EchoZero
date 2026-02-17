"""
Event Layer entity

Represents a layer of events within an EventDataItem.
Layers provide explicit grouping that matches timeline visualization.
"""
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from src.shared.domain.entities.event_data_item import Event
else:
    # Import at runtime to avoid circular dependency
    Event = None


class EventLayer:
    """
    Event layer - a named container for events.
    
    Layers provide explicit grouping that matches how events are visualized
    in the timeline. Each layer has a name that directly corresponds to a
    timeline layer name.
    
    Attributes:
        id: Unique identifier for the layer
        name: Layer name (must match timeline layer name)
        events: List of Event objects in this layer
        metadata: Optional layer metadata
    """
    
    def __init__(
        self,
        name: str,
        id: Optional[str] = None,
        events: Optional[List[Event]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize event layer.
        
        Args:
            name: Layer name (must match timeline layer name)
            id: Optional unique identifier (generated if not provided)
            events: Optional list of Event objects
            metadata: Optional metadata dictionary
        """
        if not name or not name.strip():
            raise ValueError("Layer name cannot be empty")
        
        self.id = id or str(uuid.uuid4())
        self.name = name.strip()
        self.events = events or []
        self.metadata = metadata or {}
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to this layer.
        
        Args:
            event: Event to add
        """
        if event not in self.events:
            self.events.append(event)
    
    def remove_event(self, event: Event) -> bool:
        """
        Remove an event from this layer.
        
        Args:
            event: Event to remove
            
        Returns:
            True if removed successfully
        """
        if event in self.events:
            self.events.remove(event)
            return True
        return False
    
    def get_events(self) -> List[Event]:
        """
        Get all events in this layer.
        
        Returns:
            List of Event objects
        """
        return self.events.copy()
    
    def get_events_by_classification(self, classification: str) -> List[Event]:
        """
        Get events filtered by classification.
        
        Args:
            classification: Classification to filter by
            
        Returns:
            List of matching Event objects
        """
        return [e for e in self.events if e.classification == classification]
    
    def get_events_in_range(self, start_time: float, end_time: float) -> List[Event]:
        """
        Get events within a time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of Event objects in range
        """
        return [e for e in self.events if start_time <= e.time <= end_time]
    
    def clear_events(self) -> None:
        """Clear all events from this layer"""
        self.events.clear()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EventLayer':
        """Create from dictionary"""
        # Import at runtime to avoid circular dependency
        try:
            if Event is None:
                from src.shared.domain.entities.event_data_item import Event as EventClass
            else:
                EventClass = Event
        except (NameError, ImportError):
            from src.shared.domain.entities.event_data_item import Event as EventClass
        events_data = data.get("events", [])
        events = [EventClass.from_dict(e) for e in events_data]
        
        return cls(
            id=data.get("id"),
            name=data["name"],
            events=events,
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        return f"EventLayer(id='{self.id}', name='{self.name}', events={len(self.events)})"
