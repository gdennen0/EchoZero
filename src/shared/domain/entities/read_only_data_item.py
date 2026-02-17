"""
Read-Only DataItem Wrapper

Provides read-only access to DataItems to prevent mutation of input data.
Used to enforce data ownership boundaries.

This is part of Phase 1 of the block data ownership refactor.
"""
from typing import Optional, Any, List
from datetime import datetime

from src.shared.domain.entities import DataItem, AudioDataItem, EventDataItem


class ReadOnlyDataItem:
    """
    Read-only wrapper around a DataItem that prevents mutation.
    
    This wrapper provides read access to all DataItem attributes but
    prevents modification. Attempts to modify raise AttributeError.
    
    Used to wrap input DataItems passed between blocks, ensuring
    that receiving blocks cannot mutate source data.
    """
    
    def __init__(self, data_item: DataItem):
        """
        Wrap a DataItem in a read-only wrapper.
        
        Args:
            data_item: The DataItem to wrap (must not be modified)
        """
        if not isinstance(data_item, DataItem):
            raise TypeError(f"ReadOnlyDataItem requires DataItem, got {type(data_item)}")
        
        self._data_item = data_item
    
    # Read-only property accessors
    @property
    def id(self) -> str:
        """Get the data item ID"""
        return self._data_item.id
    
    @property
    def block_id(self) -> str:
        """Get the block ID (read-only)"""
        return self._data_item.block_id
    
    @property
    def name(self) -> str:
        """Get the data item name"""
        return self._data_item.name
    
    @property
    def type(self) -> str:
        """Get the data item type"""
        return self._data_item.type
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp"""
        return self._data_item.created_at
    
    @property
    def file_path(self) -> Optional[str]:
        """Get the file path (read-only)"""
        return self._data_item.file_path
    
    @property
    def metadata(self) -> dict:
        """Get metadata (returns copy to prevent mutation)"""
        return dict(self._data_item.metadata)
    
    # Prevent mutation
    @block_id.setter
    def block_id(self, value: str):
        """Prevent mutation of block_id"""
        raise AttributeError("Cannot modify block_id on read-only DataItem")
    
    @file_path.setter
    def file_path(self, value: Optional[str]):
        """Prevent mutation of file_path"""
        raise AttributeError("Cannot modify file_path on read-only DataItem")
    
    # Prevent direct attribute assignment
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent setting any attributes except during initialization"""
        if name == '_data_item' or not hasattr(self, '_data_item'):
            # Allow setting _data_item during initialization
            super().__setattr__(name, value)
        else:
            # Prevent all other attribute assignments
            raise AttributeError(f"Cannot modify '{name}' on read-only DataItem")
    
    # Delegate other operations to wrapped item
    def to_dict(self) -> dict:
        """Convert to dictionary (delegates to wrapped item)"""
        return self._data_item.to_dict()
    
    # Type-specific properties
    @property
    def sample_rate(self) -> Optional[int]:
        """Get sample rate (for AudioDataItem)"""
        if isinstance(self._data_item, AudioDataItem):
            return self._data_item.sample_rate
        raise AttributeError("sample_rate only available on AudioDataItem")
    
    @property
    def length_ms(self) -> Optional[float]:
        """Get length in milliseconds (for AudioDataItem)"""
        if isinstance(self._data_item, AudioDataItem):
            return self._data_item.length_ms
        raise AttributeError("length_ms only available on AudioDataItem")
    
    @property
    def event_count(self) -> int:
        """Get event count (for EventDataItem)"""
        if isinstance(self._data_item, EventDataItem):
            return self._data_item.event_count
        raise AttributeError("event_count only available on EventDataItem")
    
    # For EventDataItem - get events (read-only)
    def get_events(self):
        """Get events (for EventDataItem) - returns copy to prevent mutation"""
        from src.shared.domain.entities import EventDataItem as FullEventDataItem
        if isinstance(self._data_item, FullEventDataItem):
            # Return copy of events list to prevent mutation
            events = self._data_item.get_events()
            # Return list of event copies (shallow copy is sufficient as events are simple)
            return list(events) if events else []
        raise AttributeError("get_events() only available on EventDataItem")
    
    # Utility methods
    def __repr__(self) -> str:
        """String representation"""
        return f"ReadOnlyDataItem({repr(self._data_item)})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison (compares wrapped item)"""
        if isinstance(other, ReadOnlyDataItem):
            return self._data_item == other._data_item
        return self._data_item == other
    
    def __hash__(self) -> int:
        """Hash (uses wrapped item's ID)"""
        return hash(self._data_item.id)
    
    def unwrap(self) -> DataItem:
        """
        Get the underlying DataItem (for internal use only).
        
        WARNING: This breaks the read-only guarantee. Only use when
        you need mutable access and are certain it's safe (e.g., when
        creating an owned copy).
        """
        return self._data_item
    
    @classmethod
    def is_read_only(cls, item: Any) -> bool:
        """Check if an item is a read-only wrapper"""
        return isinstance(item, cls)
    
    @classmethod
    def unwrap_if_needed(cls, item: Any) -> DataItem:
        """Unwrap if read-only, otherwise return as-is"""
        if isinstance(item, cls):
            return item.unwrap()
        if isinstance(item, DataItem):
            return item
        raise TypeError(f"Expected DataItem or ReadOnlyDataItem, got {type(item)}")





