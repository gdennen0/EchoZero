"""
Project entity

Top-level container for blocks and connections in EchoZero.
Represents a complete audio processing project.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class Project:
    """
    Project entity - represents an EchoZero project.
    
    A project contains:
    - Metadata (name, version, timestamps)
    - Save location
    - Reference to blocks (via repository)
    - Reference to connections (via repository)
    """
    id: str
    name: str
    version: str
    save_directory: Optional[str]  # None for untitled/unsaved projects
    created_at: datetime
    modified_at: datetime
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize project with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.modified_at:
            self.modified_at = datetime.utcnow()
        
        # Validate required fields
        if not self.name or not self.name.strip():
            raise ValueError("Project name cannot be empty")
        # save_directory can be None for untitled projects
        if not self.version:
            raise ValueError("Version cannot be empty")
    
    def is_untitled(self) -> bool:
        """Check if project is untitled (not yet saved)"""
        return self.save_directory is None or not self.save_directory.strip()
    
    def update_modified(self):
        """Update the modified timestamp"""
        self.modified_at = datetime.utcnow()
    
    def rename(self, new_name: str):
        """Rename project"""
        if not new_name or not new_name.strip():
            raise ValueError("Project name cannot be empty")
        self.name = new_name.strip()
        self.update_modified()
    
    def set_save_directory(self, directory: str):
        """Update save directory"""
        if not directory or not directory.strip():
            raise ValueError("Save directory cannot be empty")
        self.save_directory = directory.strip()
        self.update_modified()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "save_directory": self.save_directory,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Project':
        """Create from dictionary"""
        project_id = data.get("id") or data.get("project_id")
        return cls(
            id=project_id,
            name=data["name"],
            version=data["version"],
            save_directory=data["save_directory"],
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            metadata=data.get("metadata", {})
        )

