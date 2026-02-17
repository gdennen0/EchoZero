"""
Connection entity

Represents a connection between blocks.
Connections define the data flow in the project graph.

SIMPLIFIED: Connections now reference blocks directly by block_id + port_name.
No separate Port entities - ports are just metadata on Block.
"""
from dataclasses import dataclass
from typing import Optional
import uuid


@dataclass
class Connection:
    """
    Connection entity - links ports between blocks.
    
    A connection represents data flow:
    - For regular connections: Source block + output port -> Target block + input port
    - For bidirectional connections: First block + bidirectional port <-> Second block + bidirectional port
      (semantically, source_output_name = first block's port, target_input_name = second block's port)
    
    Simplified design:
    - No separate Port entities
    - Connections reference blocks directly by ID + port name
    - Port definitions are stored as metadata on Block
    
    Invariants:
    - For regular connections:
      - Source block must have output port with given name
      - Target block must have input port with given name
    - For bidirectional connections:
      - Both blocks must have bidirectional port with given names
      - Port types must be compatible
    - Port types must be compatible
    - Multiple connections to the same input port are allowed (e.g., Event ports)
    - Bidirectional ports can only have one connection (1:1 relationship)
    """
    id: str
    source_block_id: str  # Source block ID
    source_output_name: str  # Output port name on source block
    target_block_id: str  # Target block ID
    target_input_name: str  # Input port name on target block
    
    def __post_init__(self):
        """Initialize connection"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Validate required fields
        if not self.source_block_id:
            raise ValueError("Source block ID cannot be empty")
        if not self.source_output_name:
            raise ValueError("Source output name cannot be empty")
        if not self.target_block_id:
            raise ValueError("Target block ID cannot be empty")
        if not self.target_input_name:
            raise ValueError("Target input name cannot be empty")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "source_block_id": self.source_block_id,
            "source_output_name": self.source_output_name,
            "target_block_id": self.target_block_id,
            "target_input_name": self.target_input_name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Connection':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            source_block_id=data["source_block_id"],
            source_output_name=data["source_output_name"],
            target_block_id=data["target_block_id"],
            target_input_name=data["target_input_name"]
        )
    
    def __str__(self) -> str:
        return f"{self.source_block_id}.{self.source_output_name} -> {self.target_block_id}.{self.target_input_name}"
    
    def __repr__(self) -> str:
        return f"Connection(id={self.id}, {self})"

