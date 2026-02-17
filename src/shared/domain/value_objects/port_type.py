"""
Port type value object

Represents a data type for ports (inputs/outputs).
Immutable and contains domain rules for compatibility.
"""
from dataclasses import dataclass
from typing import Set


@dataclass(frozen=True)
class PortType:
    """
    Immutable port type identifier.
    
    Port types define what kind of data flows through a port.
    Examples: "Audio", "Event", "OSC", etc.
    """
    name: str
    
    def __post_init__(self):
        """Validate port type name"""
        if not self.name or not self.name.strip():
            raise ValueError("Port type name cannot be empty")
    
    def is_compatible_with(self, other: 'PortType') -> bool:
        """
        Domain rule: Check if this port type is compatible with another.
        
        Currently, ports are compatible if they have the same name.
        This can be extended with type hierarchies if needed.
        """
        return self.name == other.name
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"PortType(name='{self.name}')"


# Common port types
AUDIO_TYPE = PortType("Audio")
EVENT_TYPE = PortType("Event")
OSC_TYPE = PortType("OSC")
MANIPULATOR_TYPE = PortType("Manipulator")


def get_port_type(name: str) -> PortType:
    """Factory function to get or create port type"""
    known_types = {
        "Audio": AUDIO_TYPE,
        "Event": EVENT_TYPE,
        "OSC": OSC_TYPE,
        "Manipulator": MANIPULATOR_TYPE,
    }
    return known_types.get(name, PortType(name))


def is_manipulator_type(port_type: PortType) -> bool:
    """Check if a port type is a manipulator (bidirectional command) port."""
    return port_type.name == "Manipulator"


def is_bidirectional_type(port_type: PortType) -> bool:
    """
    Check if a port type is bidirectional.
    
    Bidirectional ports don't fit the input/output paradigm - they both
    send and receive data. Examples: Manipulator ports for command exchange.
    """
    bidirectional_types = {"Manipulator"}
    return port_type.name in bidirectional_types

