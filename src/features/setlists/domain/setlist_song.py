"""
Setlist Song entity

Represents a single song in a setlist.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


@dataclass
class SetlistSong:
    """
    Song in a setlist.
    
    A song has:
    - Identity (id, order_index)
    - Audio file path
    - Processing status
    - Action overrides (per-song action configuration)
    - Error message (if processing failed)
    - Metadata
    """
    id: str
    setlist_id: str
    audio_path: str
    order_index: int
    status: str = "pending"  # 'pending', 'processing', 'completed', 'failed'
    processed_at: Optional[datetime] = None
    action_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Override default actions: {block_id: {action_name: action_args, ...}, ...}
    error_message: Optional[str] = None  # Error if processing failed
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize song with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Validate required fields
        if not self.setlist_id:
            raise ValueError("Setlist ID cannot be empty")
        if not self.audio_path or not self.audio_path.strip():
            raise ValueError("Audio path cannot be empty")
        if self.order_index < 0:
            raise ValueError("Order index must be >= 0")
        
        # Validate status
        valid_statuses = {"pending", "processing", "completed", "failed"}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")
    
    def mark_processing(self):
        """Mark song as processing"""
        self.status = "processing"
    
    def mark_completed(self):
        """Mark song as completed"""
        self.status = "completed"
        self.processed_at = datetime.utcnow()
    
    def mark_failed(self, error_message: Optional[str] = None):
        """Mark song as failed"""
        self.status = "failed"
        self.processed_at = datetime.utcnow()
        if error_message:
            self.error_message = error_message
    
    def reset(self):
        """Reset song to pending state"""
        self.status = "pending"
        self.processed_at = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "setlist_id": self.setlist_id,
            "audio_path": self.audio_path,
            "order_index": self.order_index,
            "status": self.status,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "action_overrides": self.action_overrides,
            "error_message": self.error_message,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SetlistSong':
        """Create from dictionary"""
        processed_at = None
        if data.get("processed_at"):
            if isinstance(data["processed_at"], str):
                processed_at = datetime.fromisoformat(data["processed_at"])
            else:
                processed_at = data["processed_at"]
        
        # Handle migration from old format (block_settings_overrides -> action_overrides)
        action_overrides = data.get("action_overrides", {})
        if not action_overrides and "block_settings_overrides" in data:
            # Migration: convert block_settings_overrides to action_overrides format
            # This is a simplified migration - may need refinement
            action_overrides = data.get("block_settings_overrides", {})
        
        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            setlist_id=data["setlist_id"],
            audio_path=data["audio_path"],
            order_index=data["order_index"],
            status=data.get("status", "pending"),
            processed_at=processed_at,
            action_overrides=action_overrides,
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {})
        )

