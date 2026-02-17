"""
Unit Preferences

Defines preferred unit alignment for grid calculations.
Used to determine which boundaries grid lines should align to.
"""

from enum import Enum, auto


class UnitPreference(Enum):
    """
    Preferred unit alignment for grid calculations.
    
    Determines which boundaries grid intervals should align to.
    """
    FRAMES = auto()      # Align to frame boundaries (requires frame_rate)
    SECONDS = auto()     # Align to second boundaries and sub-second units
    MILLISECONDS = auto()  # Align to millisecond boundaries
    BEATS = auto()       # Align to beat boundaries (requires BPM) - future
    AUTO = auto()        # Choose automatically based on zoom level

