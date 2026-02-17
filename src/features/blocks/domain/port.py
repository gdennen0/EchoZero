"""
Port entity

Unified port representation - single abstract port class with configuration.

There are no separate "input ports" or "output ports" - just instances
of Port with different configuration (direction, port_type, metadata).
"""
from dataclasses import dataclass, field
from typing import Optional

from src.shared.domain.value_objects.port_type import PortType
from src.features.blocks.domain.port_direction import PortDirection


@dataclass(frozen=True)
class Port:
    """
    Unified port representation - single abstract port class with configuration.
    
    There are no separate "input ports" or "output ports" - just instances
    of Port with different configuration (direction, port_type, metadata).
    
    A port has:
    - name: Unique identifier within block (unique per direction)
    - port_type: Data type configuration (Audio, Event, Manipulator, etc.)
    - direction: Direction configuration (INPUT, OUTPUT, or BIDIRECTIONAL)
    - metadata: Optional port-specific configuration (UI hints, port config, etc.)
    """
    name: str
    port_type: PortType  # Simple configurable field
    direction: PortDirection  # Configuration option
    metadata: dict = field(default_factory=dict)  # Additional configuration
    
    def __post_init__(self):
        """Validate port configuration"""
        if not self.name or not self.name.strip():
            raise ValueError("Port name cannot be empty")
    
    def is_input(self) -> bool:
        """Check if this is an input port"""
        return self.direction == PortDirection.INPUT
    
    def is_output(self) -> bool:
        """Check if this is an output port"""
        return self.direction == PortDirection.OUTPUT
    
    def is_bidirectional(self) -> bool:
        """Check if this is a bidirectional port"""
        return self.direction == PortDirection.BIDIRECTIONAL
    
    def can_connect_to(self, other: 'Port') -> bool:
        """
        Domain rule: Check if this port can connect to another.
        
        Connection rules:
        - Input ports connect to output ports
        - Output ports connect to input ports
        - Bidirectional ports connect to bidirectional ports
        - Port types must be compatible
        """
        # Input ports connect to output ports
        if self.is_input() and other.is_output():
            return self.port_type.is_compatible_with(other.port_type)
        # Output ports connect to input ports
        if self.is_output() and other.is_input():
            return self.port_type.is_compatible_with(other.port_type)
        # Bidirectional ports connect to bidirectional ports
        if self.is_bidirectional() and other.is_bidirectional():
            return self.port_type.is_compatible_with(other.port_type)
        return False
    
    def __str__(self) -> str:
        return f"Port(name='{self.name}', type='{self.port_type.name}', direction='{self.direction.value}')"
    
    def __repr__(self) -> str:
        return f"Port(name='{self.name}', port_type={self.port_type!r}, direction={self.direction!r}, metadata={self.metadata})"
