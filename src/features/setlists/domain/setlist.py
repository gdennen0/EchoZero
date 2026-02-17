"""
Setlist entity

Represents a setlist - a collection of songs processed through the current project.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

@dataclass
class Setlist:
    """
    Setlist entity - collection of songs processed through current project.
    
    A setlist is identified by its audio folder path (unique).
    Contains:
    - Audio folder path (folder containing audio files, unique identifier)
    - Project reference (current project when setlist was created)
    - Default actions (actions to apply to all songs)
    - Reference to songs (via repository)
    """
    id: str
    audio_folder_path: str  # Folder with audio files (unique identifier)
    project_id: str  # Current project when setlist was created
    default_actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Default actions for all songs: {block_id: {action_name: action_args, ...}, ...}
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize setlist with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.modified_at:
            self.modified_at = datetime.utcnow()
        
        # Validate required fields
        # audio_folder_path can be empty (for empty/initialized setlists)
        if self.audio_folder_path is None:
            self.audio_folder_path = ""
        if not self.project_id or not self.project_id.strip():
            raise ValueError("Project ID cannot be empty")
    
    def update_modified(self):
        """Update the modified timestamp"""
        self.modified_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "audio_folder_path": self.audio_folder_path,
            "project_id": self.project_id,
            "default_actions": self.default_actions,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Setlist':
        """Create from dictionary"""
        setlist_id = data.get("id") or data.get("setlist_id")
        audio_folder_path = data.get("audio_folder_path", "")
        
        return cls(
            id=setlist_id,
            audio_folder_path=audio_folder_path,
            project_id=data.get("project_id", ""),
            default_actions=data.get("default_actions", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            modified_at=datetime.fromisoformat(data["modified_at"]) if isinstance(data.get("modified_at"), str) else data.get("modified_at", datetime.utcnow()),
            metadata=data.get("metadata", {})
        )

