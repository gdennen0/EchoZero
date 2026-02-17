"""
Port direction value object

Standardizes port direction/type for unified port management.
"""
from enum import Enum


class PortDirection(Enum):
    """
    Port direction/type enumeration.
    
    Standardizes the three types of ports:
    - INPUT: Data flows into the block
    - OUTPUT: Data flows out of the block
    - BIDIRECTIONAL: Data flows both ways (e.g., manipulator ports)
    """
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    
    @classmethod
    def from_string(cls, value: str) -> 'PortDirection':
        """Create PortDirection from string"""
        value_lower = value.lower()
        if value_lower == "input":
            return cls.INPUT
        elif value_lower == "output":
            return cls.OUTPUT
        elif value_lower == "bidirectional":
            return cls.BIDIRECTIONAL
        else:
            raise ValueError(f"Invalid port direction: {value}")
