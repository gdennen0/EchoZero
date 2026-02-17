"""
Action Set entity

Represents a named set of actions that can be saved, loaded, and reused.
Actions are executed in order, waiting for each to complete before proceeding.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class ActionItem:
    """
    Single action item in an action set.
    
    Represents one action to execute - either on a block or at project level.
    Stored in the database for auto-save and project persistence.
    
    Attributes:
        id: Unique identifier for the action item
        action_set_id: Foreign key to parent ActionSet
        project_id: Foreign key to project
        action_type: Type of action - "block" or "project"
        block_id: Block this action applies to (required if action_type == "block")
        block_name: Block name for display purposes
        action_name: Name of the action to execute
        action_description: Description for display purposes
        action_args: Action parameters/arguments
        order_index: Order within the action set (0-based)
        created_at: Creation timestamp
        modified_at: Last modification timestamp
        metadata: Additional metadata
    """
    action_type: str  # "block" or "project"
    action_name: str
    action_description: str  # For display purposes
    block_id: Optional[str] = None  # Required if action_type == "block"
    block_name: str = ""  # For display purposes
    action_args: Dict[str, Any] = field(default_factory=dict)  # Action parameters
    id: str = ""  # Database identity
    action_set_id: str = ""  # Foreign key to ActionSet
    project_id: str = ""  # Foreign key to project
    order_index: int = 0  # Order within action set
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.modified_at:
            self.modified_at = datetime.utcnow()
        
        # Validate action_type and block_id consistency
        if self.action_type not in ["block", "project"]:
            raise ValueError(f"Invalid action_type: {self.action_type}. Must be 'block' or 'project'")
        
        # For block actions, block_id or block_name must be provided
        # (block_name allowed for backward compatibility with old JSON files)
        if self.action_type == "block" and not self.block_id and not self.block_name:
            raise ValueError("block_id or block_name is required when action_type is 'block'")
        
        if self.action_type == "project" and self.block_id:
            raise ValueError("block_id must be None when action_type is 'project'")
    
    def update_modified(self):
        """Update the modified timestamp"""
        self.modified_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "action_set_id": self.action_set_id,
            "project_id": self.project_id,
            "action_type": self.action_type,
            "block_id": self.block_id,
            "block_name": self.block_name,
            "action_name": self.action_name,
            "action_description": self.action_description,
            "action_args": self.action_args,
            "order_index": self.order_index,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "modified_at": self.modified_at.isoformat() if isinstance(self.modified_at, datetime) else self.modified_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ActionItem':
        """Create from dictionary"""
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow())
        modified_at = datetime.fromisoformat(data["modified_at"]) if isinstance(data.get("modified_at"), str) else data.get("modified_at", datetime.utcnow())
        
        # Backward compatibility: convert empty string block_id to None
        # Old JSON files may have empty strings instead of null
        block_id = data.get("block_id")
        if block_id == "":
            block_id = None
        
        # Backward compatibility: infer action_type from block_id/block_name if missing
        # Old JSON files may not have action_type field
        action_type = data.get("action_type")
        if not action_type:
            # If block_name exists (even without block_id), assume it's a block action
            # Old format uses block_name to identify blocks
            if data.get("block_name"):
                action_type = "block"
            elif block_id:
                action_type = "block"
            else:
                # No block identifier - could be project action, but default to block for compatibility
                action_type = "block"
        
        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            action_set_id=data.get("action_set_id", ""),
            project_id=data.get("project_id", ""),
            action_type=action_type,
            block_id=block_id,
            block_name=data.get("block_name", ""),
            action_name=data["action_name"],
            action_description=data.get("action_description", ""),
            action_args=data.get("action_args", {}),
            order_index=data.get("order_index", 0),
            created_at=created_at,
            modified_at=modified_at,
            metadata=data.get("metadata", {})
        )


@dataclass
class ActionSet:
    """
    Action Set - a named collection of actions executed sequentially.
    
    Actions are executed in order, waiting for each to complete before proceeding.
    Can be saved, loaded, and reused across different workflows.
    """
    id: str
    name: str
    description: str = ""
    actions: List[ActionItem] = field(default_factory=list)  # Ordered list of actions
    project_id: Optional[str] = None  # Optional project association
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize action set with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.modified_at:
            self.modified_at = datetime.utcnow()
        
        # Validate required fields
        if not self.name or not self.name.strip():
            raise ValueError("Action set name cannot be empty")
    
    def update_modified(self):
        """Update the modified timestamp"""
        self.modified_at = datetime.utcnow()
    
    def add_action(self, action: ActionItem):
        """Add an action to the set"""
        self.actions.append(action)
        self.update_modified()
    
    def remove_action(self, index: int):
        """Remove an action by index"""
        if 0 <= index < len(self.actions):
            self.actions.pop(index)
            self.update_modified()
    
    def move_action(self, from_index: int, to_index: int):
        """Move an action to a different position"""
        if 0 <= from_index < len(self.actions) and 0 <= to_index < len(self.actions):
            action = self.actions.pop(from_index)
            self.actions.insert(to_index, action)
            self.update_modified()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "actions": [action.to_dict() for action in self.actions],
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ActionSet':
        """Create from dictionary"""
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow())
        modified_at = datetime.fromisoformat(data["modified_at"]) if isinstance(data.get("modified_at"), str) else data.get("modified_at", datetime.utcnow())
        
        actions = []
        for action_data in data.get("actions", []):
            actions.append(ActionItem.from_dict(action_data))
        
        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            name=data["name"],
            description=data.get("description", ""),
            actions=actions,
            project_id=data.get("project_id"),
            created_at=created_at,
            modified_at=modified_at,
            metadata=data.get("metadata", {})
        )
    
    def to_setlist_actions_format(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert to setlist default_actions format.
        
        Returns:
            Dict mapping block_id -> {action_name: action_args, ...}
            Only includes block actions (project actions are excluded)
        """
        result = {}
        for action in self.actions:
            if action.action_type == "block" and action.block_id:
                if action.block_id not in result:
                    result[action.block_id] = {}
                result[action.block_id][action.action_name] = action.action_args
        return result

