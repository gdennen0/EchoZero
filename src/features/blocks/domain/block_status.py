"""
Block Status Value Objects

Represents configurable status levels for blocks with condition-based evaluation.
Blocks define their own status levels with conditions, and the system evaluates
them in priority order to determine the current status.
"""
from dataclasses import dataclass, field
from typing import List, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.features.blocks.domain import Block
    from src.application.api.application_facade import ApplicationFacade


@dataclass
class BlockStatusLevel:
    """
    A status level for a block with conditions.
    
    Status levels are evaluated in ascending priority order (0, 1, 2...).
    The first level where ANY condition is False becomes the active status.
    If all levels pass, the highest priority level is returned.
    
    Args:
        priority: Evaluation order (lower = checked first, ascending)
        name: Unique identifier for this status level
        display_name: Human-readable label for UI
        color: Hex color code for UI display
        conditions: List of callables that return bool - all must be True for this level to pass
    """
    priority: int
    name: str
    display_name: str
    color: str
    conditions: List[Callable[["Block", "ApplicationFacade"], bool]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate status level configuration"""
        if not self.name:
            raise ValueError("Status level name cannot be empty")
        if not self.display_name:
            raise ValueError("Status level display_name cannot be empty")
        if not self.color:
            raise ValueError("Status level color cannot be empty")
        if not self.color.startswith("#"):
            raise ValueError(f"Status level color must be hex format (got: {self.color})")


@dataclass
class BlockStatus:
    """
    Current status of a block.
    
    Represents the active status level determined by evaluating
    all status levels in priority order.
    
    Args:
        level: The active status level
        message: Optional status message for additional context
    """
    level: BlockStatusLevel
    message: Optional[str] = None
    
    @property
    def color(self) -> str:
        """Get color from the status level"""
        return self.level.color
    
    @property
    def display_name(self) -> str:
        """Get display name from the status level"""
        return self.level.display_name
    
    @property
    def name(self) -> str:
        """Get name from the status level"""
        return self.level.name
