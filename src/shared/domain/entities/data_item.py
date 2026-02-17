"""
Data item entity

Represents processed data stored in blocks.
Data items can be audio files, events, or other data types.
"""
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class DataItem(ABC):
    """
    Base data item entity - represents processed data.
    
    Data items are stored in blocks and represent:
    - Processed audio data
    - Events (onsets, classifications, etc.)
    - Other analysis results
    
    Subclasses define specific data types (AudioData, EventData, etc.)
    """
    id: str
    block_id: str
    name: str
    type: str  # "Audio", "Event", etc.
    created_at: datetime
    file_path: Optional[str] = None  # Path to binary file if stored on disk
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize data item with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        
        # Validate required fields
        if not self.name or not self.name.strip():
            raise ValueError("Data item name cannot be empty")
        if not self.type or not self.type.strip():
            raise ValueError("Data item type cannot be empty")
        if not self.block_id:
            raise ValueError("Block ID cannot be empty")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "block_id": self.block_id,
            "name": self.name,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "file_path": self.file_path,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DataItem':
        """Create from dictionary - must be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement from_dict")


@dataclass
class AudioDataItem(DataItem):
    """
    Audio data item - represents an audio file.
    
    Note: For full audio loading/saving functionality, use 
    src.domain.entities.audio_data_item.AudioDataItem
    """
    sample_rate: Optional[int] = None
    length_ms: Optional[float] = None
    
    def __post_init__(self):
        """Initialize audio data item"""
        if not hasattr(self, 'type') or not self.type:
            self.type = "Audio"
        super().__post_init__()
    
    def to_dict(self) -> dict:
        """Convert to dictionary including audio-specific fields"""
        data = super().to_dict()
        data["sample_rate"] = self.sample_rate
        data["length_ms"] = self.length_ms
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AudioDataItem':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            block_id=data["block_id"],
            name=data["name"],
            type=data.get("type", "Audio"),
            created_at=datetime.fromisoformat(data["created_at"]),
            file_path=data.get("file_path"),
            sample_rate=data.get("sample_rate"),
            length_ms=data.get("length_ms"),
            metadata=data.get("metadata", {})
        )


@dataclass
class EventDataItem(DataItem):
    """
    Event data item - represents a collection of events.
    
    Note: For full event management functionality, use
    src.domain.entities.event_data_item.EventDataItem
    """
    event_count: int = 0
    
    def __post_init__(self):
        """Initialize event data item"""
        if not hasattr(self, 'type') or not self.type:
            self.type = "Event"
        super().__post_init__()
    
    def to_dict(self) -> dict:
        """Convert to dictionary including event-specific fields"""
        data = super().to_dict()
        data["event_count"] = self.event_count
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EventDataItem':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            block_id=data["block_id"],
            name=data["name"],
            type=data.get("type", "Event"),
            created_at=datetime.fromisoformat(data["created_at"]),
            file_path=data.get("file_path"),
            event_count=data.get("event_count", 0),
            metadata=data.get("metadata", {})
        )

