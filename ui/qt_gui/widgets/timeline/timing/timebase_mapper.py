"""
Timebase Mapper

Maps old TimebaseMode enum to new UnitPreference enum.
Provides backward compatibility during migration.
"""

from enum import Enum, auto
from .unit_preferences import UnitPreference


class TimebaseMode(Enum):
    """
    Legacy timebase mode enum.
    
    Kept for backward compatibility during migration.
    Maps to UnitPreference for new timing system.
    """
    SECONDS = auto()      # Simple seconds (0.000)
    TIMECODE = auto()     # hh:mm:ss.ff (frames)
    FRAMES = auto()       # Raw frame count at given FPS
    MILLISECONDS = auto() # Milliseconds display


def timebase_to_unit_preference(timebase_mode: TimebaseMode) -> UnitPreference:
    """
    Convert legacy TimebaseMode to new UnitPreference.
    
    Args:
        timebase_mode: Legacy timebase mode
    
    Returns:
        Corresponding UnitPreference
    """
    if timebase_mode == TimebaseMode.TIMECODE:
        return UnitPreference.AUTO  # Timecode uses frames + seconds
    elif timebase_mode == TimebaseMode.FRAMES:
        return UnitPreference.FRAMES
    elif timebase_mode == TimebaseMode.SECONDS:
        return UnitPreference.SECONDS
    elif timebase_mode == TimebaseMode.MILLISECONDS:
        return UnitPreference.MILLISECONDS
    else:
        return UnitPreference.SECONDS  # Default fallback

