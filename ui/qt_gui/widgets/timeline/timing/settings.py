"""
Grid Settings

Configuration settings for grid snapping and time display.
Extracted from grid_system.py to consolidate timing-related settings.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from .unit_preferences import UnitPreference


# Valid explicit snap interval modes
VALID_SNAP_MODES = {"auto", "1f", "2f", "5f", "10f", "1s"}

# Common frame rates for UI selection
COMMON_FRAME_RATES: List[Tuple[str, float]] = [
    ("23.976 fps (Film NTSC)", 23.976),
    ("24 fps (Film)", 24.0),
    ("25 fps (PAL)", 25.0),
    ("29.97 fps (NTSC)", 29.97),
    ("30 fps", 30.0),
    ("48 fps", 48.0),
    ("60 fps", 60.0),
    ("120 fps", 120.0),
]


@dataclass
class GridSettings:
    """Grid configuration settings."""
    snap_enabled: bool = True
    # Snap interval mode: "auto" uses current grid, or explicit like "1f", "5f", "10f", "1s"
    snap_interval_mode: str = "auto"
    unit_preference: UnitPreference = UnitPreference.AUTO
    frame_rate: float = 30.0  # FPS for frame-based modes
    show_grid_lines: bool = True
    
    def get_snap_interval_seconds(self) -> Optional[float]:
        """
        Get the explicit snap interval in seconds, or None for auto mode.
        
        Returns:
            Snap interval in seconds based on current mode and frame rate,
            or None if mode is "auto" (use grid interval).
        """
        mode = self.snap_interval_mode
        if mode == "auto":
            return None
        
        fps = self.frame_rate
        frame_duration = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        if mode == "1f":
            return frame_duration
        elif mode == "2f":
            return frame_duration * 2
        elif mode == "5f":
            return frame_duration * 5
        elif mode == "10f":
            return frame_duration * 10
        elif mode == "1s":
            return 1.0
        else:
            return None
    
    def set_snap_interval_mode(self, mode: str) -> bool:
        """
        Set snap interval mode.
        
        Args:
            mode: "auto" for grid-based, or "1f", "2f", "5f", "10f", "1s"
        
        Returns:
            True if mode was valid and set, False otherwise
        """
        if mode in VALID_SNAP_MODES:
            self.snap_interval_mode = mode
            return True
        return False
    
    @staticmethod
    def get_frame_rate_options() -> List[Tuple[str, float]]:
        """
        Get list of common frame rate options.
        
        Returns:
            List of (display_name, fps) tuples
        """
        return COMMON_FRAME_RATES.copy()




