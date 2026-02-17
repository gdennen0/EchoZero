"""
Event Filter Manager

Service for filtering events within EventDataItems based on user-defined criteria.
Part of the "processes" improvement area for the Editor block.

Provides functionality for:
- Filtering events by classification, time range, duration, and metadata
- Managing filter state per block
- Applying filters to events in both visualization and processing
- Extensible for future batch/event processing features

Design Principles:
- Extensible: Supports custom predicates for future filter types
- Block-agnostic: Can work with any storage mechanism
- Simple: Minimal abstraction, add complexity only when needed

EXTENSIBILITY GUIDE
==================

Adding Custom Filter Predicates:
---------------------------------
To add custom filtering logic, use the custom_predicates parameter:

```python
def my_custom_filter(event: Event) -> bool:
    # Your custom logic here
    return event.metadata.get("confidence", 0) > 0.8

filter = EventFilter(
    enabled_classifications={"kick", "snare"},
    custom_predicates=[my_custom_filter]
)
```

Note: Custom predicates are not serialized to dict (they're functions).
If you need persistence, store predicate configuration in metadata_filters
and reconstruct the predicate on load.

Creating Custom Storage:
------------------------
To use filters with contexts other than blocks (e.g., projects, presets):

```python
class ProjectFilterStorage(EventFilterStorage):
    def get_filter_data(self, context_id: str) -> Optional[Dict[str, Any]]:
        # Load from project metadata
        return project.metadata.get("event_filter")
    
    def save_filter_data(self, context_id: str, filter_data: Dict[str, Any]) -> None:
        # Save to project metadata
        project.metadata["event_filter"] = filter_data
    
    def clear_filter_data(self, context_id: str) -> None:
        # Clear from project metadata
        if "event_filter" in project.metadata:
            del project.metadata["event_filter"]

# Use it
manager = EventFilterManager(storage=ProjectFilterStorage())
```

Future Event Processing:
-------------------------
For event transformations beyond filtering, implement EventProcessor:

```python
class EventTimeShiftProcessor(EventProcessor):
    def __init__(self, offset_seconds: float):
        self.offset = offset_seconds
    
    def process_events(self, events: List[Event]) -> List[Event]:
        return [Event(e.time + self.offset, e.classification, e.duration, e.metadata) 
                for e in events]
    
    def process_event_data_item(self, event_item: EventDataItem) -> EventDataItem:
        shifted_events = self.process_events(event_item.get_events())
        return EventDataItem(..., events=shifted_events)
```

This allows chaining operations:
```python
# Filter first, then transform
filtered = filter_manager.filter_event_data_item(item, filter)
shifted = time_shift_processor.process_event_data_item(filtered)
```

See EchoZero feature development guide for more patterns.
"""
from typing import List, Optional, Dict, Any, Set, Callable
from abc import ABC, abstractmethod
from src.features.blocks.domain import Block
from src.shared.domain.entities import EventDataItem, Event
from src.utils.message import Log


class EventFilter:
    """
    Represents a filter configuration for events.
    
    Filters can be applied by:
    - Classification (include/exclude specific classifications)
    - Time range (min/max time)
    - Duration range (min/max duration)
    - Metadata keys/values
    - Custom predicates (extensible for future features)
    
    Extensibility:
    - Add custom_predicates list of callable functions for custom filtering logic
    - Each predicate should accept (event: Event) -> bool
    """
    
    def __init__(
        self,
        enabled_classifications: Optional[Set[str]] = None,
        excluded_classifications: Optional[Set[str]] = None,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        custom_predicates: Optional[List[Callable[[Event], bool]]] = None,
        enabled: bool = True
    ):
        """
        Initialize event filter.
        
        Args:
            enabled_classifications: Set of classifications to include (None = all)
            excluded_classifications: Set of classifications to exclude
            min_time: Minimum event time in seconds (None = no limit)
            max_time: Maximum event time in seconds (None = no limit)
            min_duration: Minimum event duration in seconds (None = no limit)
            max_duration: Maximum event duration in seconds (None = no limit)
            metadata_filters: Dict of metadata key-value pairs to match (all must match)
            custom_predicates: Optional list of custom filter functions (event) -> bool
            enabled: Whether filter is active
        """
        self.enabled_classifications = enabled_classifications or set()
        self.excluded_classifications = excluded_classifications or set()
        self.min_time = min_time
        self.max_time = max_time
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.metadata_filters = metadata_filters or {}
        self.custom_predicates = custom_predicates or []
        self.enabled = enabled
    
    def matches(self, event: Event) -> bool:
        """
        Check if an event matches the filter criteria.
        
        Args:
            event: Event to check
            
        Returns:
            True if event matches all filter criteria
        """
        if not self.enabled:
            return True
        
        # Classification filter
        # If enabled_classifications is set and non-empty, only include those classifications
        # If empty or None, include all classifications (no filter)
        if self.enabled_classifications and len(self.enabled_classifications) > 0:
            if event.classification not in self.enabled_classifications:
                return False
        
        if event.classification in self.excluded_classifications:
            return False
        
        # Time range filter
        if self.min_time is not None and event.time < self.min_time:
            return False
        if self.max_time is not None and event.time > self.max_time:
            return False
        
        # Duration filter
        if self.min_duration is not None and event.duration < self.min_duration:
            return False
        if self.max_duration is not None and event.duration > self.max_duration:
            return False
        
        # Metadata filter (all specified keys must match)
        # Supports comparison operators: ==, !=, >, <, >=, <=, contains, not_contains, in, not_in
        if self.metadata_filters:
            for key, filter_spec in self.metadata_filters.items():
                if key not in event.metadata:
                    return False
                
                event_value = event.metadata[key]
                
                # Handle different filter spec formats:
                # Old format (backward compatible): {"key": value} -> exact match
                # New format: {"key": {"operator": "==", "value": value}}
                if isinstance(filter_spec, dict) and "operator" in filter_spec:
                    operator = filter_spec.get("operator", "==")
                    filter_value = filter_spec.get("value")
                    
                    if not self._compare_metadata_values(event_value, operator, filter_value):
                        return False
                else:
                    # Legacy format: exact match
                    if event_value != filter_spec:
                        return False
        
        # Custom predicates (extensible for future features)
        # All custom predicates must return True for event to match
        for predicate in self.custom_predicates:
            try:
                if not predicate(event):
                    return False
            except Exception as e:
                Log.warning(f"EventFilter: Custom predicate failed: {e}")
                return False
        
        return True
    
    def _compare_metadata_values(self, event_value: Any, operator: str, filter_value: Any) -> bool:
        """
        Compare event metadata value with filter value using specified operator.
        
        Args:
            event_value: Value from event metadata
            operator: Comparison operator (==, !=, >, <, >=, <=, contains, not_contains, in, not_in)
            filter_value: Value to compare against
            
        Returns:
            True if comparison matches
        """
        try:
            if operator == "==":
                return event_value == filter_value
            elif operator == "!=":
                return event_value != filter_value
            elif operator == ">":
                return self._try_compare(event_value, filter_value, lambda a, b: a > b)
            elif operator == "<":
                return self._try_compare(event_value, filter_value, lambda a, b: a < b)
            elif operator == ">=":
                return self._try_compare(event_value, filter_value, lambda a, b: a >= b)
            elif operator == "<=":
                return self._try_compare(event_value, filter_value, lambda a, b: a <= b)
            elif operator == "contains":
                if isinstance(event_value, str) and isinstance(filter_value, str):
                    return filter_value.lower() in event_value.lower()
                elif isinstance(event_value, (list, tuple, set)):
                    return filter_value in event_value
                return False
            elif operator == "not_contains":
                if isinstance(event_value, str) and isinstance(filter_value, str):
                    return filter_value.lower() not in event_value.lower()
                elif isinstance(event_value, (list, tuple, set)):
                    return filter_value not in event_value
                return True
            elif operator == "in":
                if isinstance(filter_value, (list, tuple, set)):
                    return event_value in filter_value
                return False
            elif operator == "not_in":
                if isinstance(filter_value, (list, tuple, set)):
                    return event_value not in filter_value
                return True
            else:
                Log.warning(f"EventFilter: Unknown operator '{operator}', defaulting to ==")
                return event_value == filter_value
        except Exception as e:
            Log.warning(f"EventFilter: Comparison failed ({operator}): {e}")
            return False
    
    def _try_compare(self, a: Any, b: Any, compare_fn) -> bool:
        """Try to compare two values, handling type conversion"""
        try:
            # Try numeric comparison
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return compare_fn(a, b)
            # Try string comparison
            if isinstance(a, str) and isinstance(b, str):
                return compare_fn(a, b)
            # Try converting both to float
            return compare_fn(float(a), float(b))
        except (ValueError, TypeError):
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert filter to dictionary for persistence.
        
        Note: Custom predicates are not serialized (they're callable functions).
        If you need to persist custom predicates, store them as metadata with
        a key that can be used to reconstruct the predicate.
        """
        return {
            "enabled_classifications": list(self.enabled_classifications) if self.enabled_classifications else None,
            "excluded_classifications": list(self.excluded_classifications) if self.excluded_classifications else None,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "metadata_filters": self.metadata_filters,
            "enabled": self.enabled,
            # Note: custom_predicates are not serialized (callable functions)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventFilter':
        """Create filter from dictionary"""
        return cls(
            enabled_classifications=set(data.get("enabled_classifications", [])) if data.get("enabled_classifications") else None,
            excluded_classifications=set(data.get("excluded_classifications", [])) if data.get("excluded_classifications") else None,
            min_time=data.get("min_time"),
            max_time=data.get("max_time"),
            min_duration=data.get("min_duration"),
            max_duration=data.get("max_duration"),
            metadata_filters=data.get("metadata_filters", {}),
            enabled=data.get("enabled", True)
        )


class EventFilterStorage(ABC):
    """
    Abstract interface for storing/loading event filters.
    
    Allows EventFilterManager to work with any storage mechanism,
    not just block metadata. This makes it extensible for future use cases.
    """
    
    @abstractmethod
    def get_filter_data(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get filter data for a context (e.g., block_id)"""
        pass
    
    @abstractmethod
    def save_filter_data(self, context_id: str, filter_data: Dict[str, Any]) -> None:
        """Save filter data for a context"""
        pass
    
    @abstractmethod
    def clear_filter_data(self, context_id: str) -> None:
        """Clear filter data for a context"""
        pass


class BlockMetadataFilterStorage(EventFilterStorage):
    """
    Default storage implementation using block metadata.
    
    This is the standard implementation for Editor blocks.
    """
    
    def __init__(self, get_block_fn: Callable[[str], Optional[Block]]):
        """
        Initialize with a function to get blocks by ID.
        
        Args:
            get_block_fn: Function that takes block_id and returns Block or None
        """
        self._get_block = get_block_fn
    
    def get_filter_data(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get filter from block metadata"""
        block = self._get_block(context_id)
        if not block:
            return None
        return block.metadata.get("event_filter")
    
    def save_filter_data(self, context_id: str, filter_data: Dict[str, Any]) -> None:
        """Save filter to block metadata"""
        block = self._get_block(context_id)
        if block:
            block.metadata["event_filter"] = filter_data
    
    def clear_filter_data(self, context_id: str) -> None:
        """Clear filter from block metadata"""
        block = self._get_block(context_id)
        if block and "event_filter" in block.metadata:
            del block.metadata["event_filter"]


class EventFilterManager:
    """
    Service for managing event filters.
    
    Provides functionality for:
    - Loading and saving filter configurations (block-agnostic)
    - Applying filters to EventDataItems
    - Filtering individual events
    
    Extensibility:
    - Works with any storage mechanism via EventFilterStorage interface
    - Default implementation uses block metadata (BlockMetadataFilterStorage)
    - Can be extended for other contexts (projects, presets, etc.)
    """
    
    def __init__(self, storage: Optional[EventFilterStorage] = None):
        """
        Initialize event filter manager.
        
        Args:
            storage: Optional storage implementation. If None, creates default
                     BlockMetadataFilterStorage (requires get_block_fn to be set later)
        """
        self._storage = storage
        self._get_block_fn: Optional[Callable[[str], Optional[Block]]] = None
    
    def set_block_getter(self, get_block_fn: Callable[[str], Optional[Block]]) -> None:
        """
        Set the function to get blocks by ID.
        
        Required for default BlockMetadataFilterStorage.
        Should be called during initialization if using default storage.
        
        Args:
            get_block_fn: Function that takes block_id and returns Block or None
        """
        self._get_block_fn = get_block_fn
        if self._storage is None:
            self._storage = BlockMetadataFilterStorage(get_block_fn)
    
    def get_filter(self, context_id: str, storage: Optional[EventFilterStorage] = None) -> Optional[EventFilter]:
        """
        Get event filter for a context (e.g., block_id).
        
        Args:
            context_id: Context identifier (e.g., block_id)
            storage: Optional storage to use (defaults to instance storage)
            
        Returns:
            EventFilter instance, or None if no filter configured
        """
        storage = storage or self._storage
        if not storage:
            Log.warning("EventFilterManager: No storage configured")
            return None
        
        filter_data = storage.get_filter_data(context_id)
        if not filter_data:
            return None
        
        try:
            return EventFilter.from_dict(filter_data)
        except Exception as e:
            Log.warning(f"EventFilterManager: Failed to load filter for {context_id}: {e}")
            return None
    
    def get_filter_for_block(self, block: Block) -> Optional[EventFilter]:
        """
        Convenience method to get filter for a block.
        
        Supports both:
        1. Settings abstraction pattern (EditorEventFilterSettings)
        2. Legacy format (event_filter in metadata)
        
        Args:
            block: Block to get filter for
            
        Returns:
            EventFilter instance, or None if no filter configured
        """
        # Try settings abstraction pattern first (new format)
        try:
            # Check if settings exist in metadata (settings manager stores at top level)
            if hasattr(block, 'metadata') and block.metadata:
                # Check for settings fields (from EditorEventFilterSettings)
                if 'event_filter_enabled' in block.metadata or any(
                    key.startswith(('enabled_classifications', 'excluded_classifications', 
                                   'min_time', 'max_time', 'min_duration', 'max_duration',
                                   'metadata_filters'))
                    for key in block.metadata.keys()
                ):
                    # Load via settings schema
                    from src.application.settings.editor_event_filter_settings import EditorEventFilterSettings
                    settings = EditorEventFilterSettings.from_dict(block.metadata)
                    filter_dict = settings.to_event_filter_dict()
                    if filter_dict and filter_dict.get("enabled", True):
                        return EventFilter.from_dict(filter_dict)
        except Exception as e:
            Log.debug(f"EventFilterManager: Failed to load from settings schema: {e}")
        
        # Fall back to legacy format (event_filter key)
        return self.get_filter(block.id)
    
    def save_filter(self, context_id: str, filter: EventFilter, storage: Optional[EventFilterStorage] = None) -> None:
        """
        Save event filter for a context.
        
        Args:
            context_id: Context identifier (e.g., block_id)
            filter: EventFilter instance to save
            storage: Optional storage to use (defaults to instance storage)
        """
        storage = storage or self._storage
        if not storage:
            Log.warning("EventFilterManager: No storage configured")
            return
        
        storage.save_filter_data(context_id, filter.to_dict())
        Log.debug(f"EventFilterManager: Saved event filter for {context_id}")
    
    def save_filter_for_block(self, block: Block, filter: EventFilter) -> None:
        """
        Convenience method to save filter for a block.
        
        Note: The caller must persist the block via the facade or command system.
        
        Args:
            block: Block to save filter for
            filter: EventFilter instance to save
        """
        self.save_filter(block.id, filter)
    
    def clear_filter(self, context_id: str, storage: Optional[EventFilterStorage] = None) -> None:
        """
        Clear event filter for a context.
        
        Args:
            context_id: Context identifier (e.g., block_id)
            storage: Optional storage to use (defaults to instance storage)
        """
        storage = storage or self._storage
        if not storage:
            return
        
        storage.clear_filter_data(context_id)
        Log.debug(f"EventFilterManager: Cleared event filter for {context_id}")
    
    def clear_filter_for_block(self, block: Block) -> None:
        """
        Convenience method to clear filter for a block.
        
        Args:
            block: Block to clear filter for
        """
        self.clear_filter(block.id)
    
    def filter_events(
        self,
        event_item: EventDataItem,
        filter: Optional[EventFilter] = None,
        context_id: Optional[str] = None,
        block: Optional[Block] = None
    ) -> List[Event]:
        """
        Filter events from an EventDataItem.
        
        Args:
            event_item: EventDataItem to filter
            filter: Optional EventFilter to apply (if None, loads from context/block)
            context_id: Optional context ID to load filter from if filter is None
            block: Optional Block to load filter from if filter is None (convenience)
            
        Returns:
            List of filtered Event objects
        """
        if filter is None:
            if block:
                filter = self.get_filter_for_block(block)
            elif context_id:
                filter = self.get_filter(context_id)
        
        if not filter:
            # No filter - return all events
            return event_item.get_events()
        
        return [event for event in event_item.get_events() if filter.matches(event)]
    
    def filter_event_data_item(
        self,
        event_item: EventDataItem,
        filter: Optional[EventFilter] = None,
        context_id: Optional[str] = None,
        block: Optional[Block] = None
    ) -> EventDataItem:
        """
        Create a filtered copy of an EventDataItem.
        
        Creates a new EventDataItem with only the filtered events.
        The new item has the same block_id and metadata as the original.
        
        Args:
            event_item: EventDataItem to filter
            filter: Optional EventFilter to apply (if None, loads from context/block)
            context_id: Optional context ID to load filter from if filter is None
            block: Optional Block to load filter from if filter is None (convenience)
            
        Returns:
            New EventDataItem with filtered events
        """
        filtered_events = self.filter_events(event_item, filter, context_id, block)
        
        # Create new EventDataItem with filtered events
        filtered_item = EventDataItem(
            id=event_item.id,  # Keep same ID for consistency
            block_id=event_item.block_id,
            name=event_item.name,
            type=event_item.type,
            created_at=event_item.created_at,
            file_path=event_item.file_path,
            metadata=event_item.metadata.copy(),
            events=filtered_events
        )
        
        return filtered_item


# Event Processor Interface (for future batch/event processing features)
class EventProcessor(ABC):
    """
    Abstract interface for event processing operations.
    
    This provides a foundation for future event processing features beyond filtering:
    - Event transformations (time shift, duration adjustment, etc.)
    - Event enrichment (add metadata, classifications, etc.)
    - Event aggregation (group, merge, split, etc.)
    - Batch operations (process multiple EventDataItems)
    
    Design: Keep it simple. Add concrete implementations only when needed.
    """
    
    @abstractmethod
    def process_events(self, events: List[Event]) -> List[Event]:
        """
        Process a list of events.
        
        Args:
            events: Input events to process
            
        Returns:
            Processed events (may be filtered, transformed, or enriched)
        """
        pass
    
    @abstractmethod
    def process_event_data_item(self, event_item: EventDataItem) -> EventDataItem:
        """
        Process an EventDataItem.
        
        Args:
            event_item: Input EventDataItem
            
        Returns:
            Processed EventDataItem (new instance with processed events)
        """
        pass

