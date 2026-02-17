"""
Data State Snapshot entity

Represents a snapshot of execution results (data items, block state, settings overrides).
Used by setlist system to save and restore execution states per song.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List
import uuid


@dataclass
class DataStateSnapshot:
    """
    Snapshot of execution results.
    
    Contains:
    - Serialized data items (from all blocks)
    - Block local state (input/output references)
    - Block settings overrides (per-song setting changes)
    
    Used to save execution state after processing a song,
    then restore it when switching between songs in a setlist.
    """
    id: str
    song_id: str  # Which song this snapshot belongs to
    created_at: datetime
    data_items: List[Dict[str, Any]] = field(default_factory=list)  # Serialized data items
    block_local_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # block_id -> port -> data_item_id
    block_settings_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # block_id -> {setting_key: value}
    
    def __post_init__(self):
        """Initialize snapshot with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        
        # Validate required fields
        if not self.song_id:
            raise ValueError("Song ID cannot be empty")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "song_id": self.song_id,
            "created_at": self.created_at.isoformat(),
            "data_items": self.data_items,
            "block_local_state": self.block_local_state,
            "block_settings_overrides": self.block_settings_overrides
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DataStateSnapshot':
        """Create from dictionary"""
        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            song_id=data["song_id"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
            data_items=data.get("data_items", []),
            block_local_state=data.get("block_local_state", {}),
            block_settings_overrides=data.get("block_settings_overrides", {})
        )

