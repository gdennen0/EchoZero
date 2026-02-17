"""
Execution Strategy Value Object

Defines how setlist processing should execute blocks.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExecutionStrategy:
    """
    Execution strategy for setlist processing.
    
    Determines how blocks are executed during setlist processing:
    - Full re-execution: Execute all blocks from scratch (default, safest)
    - Actions only: Only apply configured actions, don't re-execute blocks (faster)
    - Hybrid: Re-execute some blocks, action-only for others (advanced)
    """
    mode: str  # "full" | "actions_only" | "hybrid"
    re_execute_blocks: List[str] = field(default_factory=list)  # Block IDs to fully re-execute (empty = all)
    action_only_blocks: List[str] = field(default_factory=list)  # Block IDs to only apply actions (no execution)
    skip_blocks: List[str] = field(default_factory=list)  # Block IDs to skip entirely
    
    def __post_init__(self):
        """Validate execution strategy"""
        valid_modes = {"full", "actions_only", "hybrid"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid execution mode: {self.mode}. Must be one of {valid_modes}")
    
    @classmethod
    def default(cls) -> 'ExecutionStrategy':
        """Create default execution strategy (full re-execution)"""
        return cls(
            mode="full",
            re_execute_blocks=[],
            action_only_blocks=[],
            skip_blocks=[]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "mode": self.mode,
            "re_execute_blocks": self.re_execute_blocks,
            "action_only_blocks": self.action_only_blocks,
            "skip_blocks": self.skip_blocks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionStrategy':
        """Create from dictionary"""
        return cls(
            mode=data.get("mode", "full"),
            re_execute_blocks=data.get("re_execute_blocks", []),
            action_only_blocks=data.get("action_only_blocks", []),
            skip_blocks=data.get("skip_blocks", [])
        )

