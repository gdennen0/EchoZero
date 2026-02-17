"""
Grid System

Facade for the timing system - provides backward-compatible API for grid snapping
and time display formatting.

This module delegates to the timing/ subpackage which contains the actual implementations:
- timing/grid_calculator.py: Interval calculations
- timing/snap_calculator.py: Snapping logic
- timing/time_converter.py: Time formatting
- timing/settings.py: Settings dataclass

Key Features:
- Unit-aligned grid lines (frames, seconds, minutes, hours)
- Multiple timebase modes (TIMECODE, FRAMES, SECONDS, MILLISECONDS)
- Snap-to-grid for event positioning
- Time formatting and parsing
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass

# Import from timing module - these are the actual implementations
from .timing import (
    GridCalculator,
    SnapCalculator,
    TimeConverter,
    TimeFormat,
    UnitPreference,
    TimebaseMode,
    timebase_to_unit_preference,
    GridSettings as TimingGridSettings,
    COMMON_FRAME_RATES,
)


# Re-export TimebaseMode for backward compatibility
__all__ = ['TimebaseMode', 'GridSettings', 'GridSystem']


@dataclass
class GridSettings:
    """
    Grid configuration settings.
    
    Wraps timing.GridSettings for backward compatibility with TimebaseMode.
    """
    snap_enabled: bool = True
    snap_interval_mode: str = "auto"
    timebase_mode: TimebaseMode = TimebaseMode.TIMECODE
    frame_rate: float = 30.0
    show_grid_lines: bool = True
    
    def get_snap_interval_seconds(self) -> Optional[float]:
        """Get explicit snap interval in seconds, or None for auto mode."""
        mode = self.snap_interval_mode
        if mode == "auto":
            return None
        
        fps = self.frame_rate
        frame_duration = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        intervals = {
            "1f": frame_duration,
            "2f": frame_duration * 2,
            "5f": frame_duration * 5,
            "10f": frame_duration * 10,
            "1s": 1.0,
        }
        return intervals.get(mode)


class GridSystem:
    """
    Facade for grid snapping and time display formatting.
    
    Delegates to timing/ module components for actual implementation.
    Provides backward-compatible API for existing code.
    """
    
    def __init__(self):
        self._settings = GridSettings()
        self._snap_threshold_pixels = 8
        
        # Timing module components
        self._grid_calculator = GridCalculator(frame_rate=self._settings.frame_rate)
        self._snap_calculator = SnapCalculator(self._grid_calculator, self._settings)
        self._time_converter = TimeConverter()
    
    @property
    def settings(self) -> GridSettings:
        """Get current grid settings."""
        return self._settings
    
    @property
    def snap_enabled(self) -> bool:
        """Check if snap is enabled."""
        return self._settings.snap_enabled
    
    @snap_enabled.setter
    def snap_enabled(self, value: bool):
        """Enable or disable snapping."""
        self._settings.snap_enabled = value
        self._snap_calculator.snap_enabled = value
    
    @property
    def snap_interval_mode(self) -> str:
        """Get snap interval mode (auto, 1f, 2f, 5f, 10f, 1s)."""
        return self._settings.snap_interval_mode
    
    def set_snap_interval_mode(self, mode: str):
        """Set snap interval mode."""
        valid_modes = {"auto", "1f", "2f", "5f", "10f", "1s"}
        if mode in valid_modes:
            self._settings.snap_interval_mode = mode
    
    def get_snap_interval_seconds(self) -> Optional[float]:
        """Get explicit snap interval in seconds, or None for auto mode."""
        return self._settings.get_snap_interval_seconds()
    
    @property
    def frame_rate(self) -> float:
        """Get current frame rate."""
        return self._settings.frame_rate
    
    @frame_rate.setter
    def frame_rate(self, value: float):
        """Set frame rate (FPS)."""
        if value <= 0:
            raise ValueError(f"Frame rate must be positive, got {value}")
        if value > 1000:
            raise ValueError(f"Frame rate too high (max 1000 fps), got {value}")
        self._settings.frame_rate = max(1.0, min(1000.0, value))
        self._grid_calculator.frame_rate = self._settings.frame_rate
    
    @property
    def timebase_mode(self) -> TimebaseMode:
        """Get current timebase display mode."""
        return self._settings.timebase_mode
    
    @timebase_mode.setter
    def timebase_mode(self, mode: TimebaseMode):
        """Set timebase display mode."""
        self._settings.timebase_mode = mode
    
    def snap_time(self, time: float, pixels_per_second: Optional[float] = None) -> float:
        """
        Snap a time value to the displayed grid.
        
        Args:
            time: Time in seconds to snap
            pixels_per_second: Current zoom level (required for grid calculation)
            
        Returns:
            Snapped time value aligned to displayed grid lines
        """
        if not self._settings.snap_enabled:
            return time
        
        if pixels_per_second is None or pixels_per_second <= 0:
            return time
        
        # Convert TimebaseMode to UnitPreference
        unit_pref = timebase_to_unit_preference(self._settings.timebase_mode)
        
        # Use SnapCalculator (handles explicit intervals internally)
        return self._snap_calculator.snap_time(time, pixels_per_second, unit_pref)
    
    def snap_to_frame(self, time: float) -> float:
        """Snap time to nearest frame boundary."""
        if not self._settings.snap_enabled:
            return time
        
        frame_duration = 1.0 / self._settings.frame_rate
        frame_number = round(time / frame_duration)
        return frame_number * frame_duration
    
    def snap_to_unit_boundary(self, time: float, interval: float) -> float:
        """Snap time to nearest unit boundary."""
        if interval <= 0:
            return time
        return round(time / interval) * interval
    
    def get_snap_points(
        self,
        start_time: float,
        end_time: float,
        pixels_per_second: float
    ) -> List[float]:
        """Get grid line positions for drawing."""
        _, minor_interval = self.get_major_minor_intervals(pixels_per_second)
        
        points = []
        t = (int(start_time / minor_interval)) * minor_interval
        while t <= end_time:
            if t >= 0:
                points.append(t)
            t += minor_interval
        
        return points
    
    def get_major_minor_intervals(
        self,
        pixels_per_second: float,
        major_multiplier: float = 1.0,
        minor_multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Get major and minor grid intervals based on zoom level.
        
        Args:
            pixels_per_second: Current zoom level
            major_multiplier: Deprecated - kept for compatibility
            minor_multiplier: Deprecated - kept for compatibility
            
        Returns:
            Tuple of (major_interval, minor_interval) in seconds
        """
        # Convert TimebaseMode to UnitPreference
        unit_pref = timebase_to_unit_preference(self._settings.timebase_mode)
        
        # Delegate to GridCalculator
        return self._grid_calculator.get_intervals(pixels_per_second, unit_pref)
    
    def format_time(self, seconds: float) -> str:
        """Format time according to current timebase mode."""
        mode = self._settings.timebase_mode
        frame_rate = self._settings.frame_rate
        
        # Map TimebaseMode to TimeFormat
        format_map = {
            TimebaseMode.SECONDS: TimeFormat.SECONDS,
            TimebaseMode.MILLISECONDS: TimeFormat.MILLISECONDS,
            TimebaseMode.FRAMES: TimeFormat.FRAMES,
            TimebaseMode.TIMECODE: TimeFormat.TIMECODE,
        }
        time_format = format_map.get(mode, TimeFormat.SECONDS)
        
        return self._time_converter.format_time(
            seconds, time_format, frame_rate, compact=True
        )
    
    def parse_timecode(self, timecode: str) -> Optional[float]:
        """Parse a timecode string to seconds."""
        return self._time_converter.parse_timecode(timecode, self._settings.frame_rate)
    
    def get_frame_rate_options(self) -> List[Tuple[str, float]]:
        """Get list of common frame rate options."""
        return list(COMMON_FRAME_RATES)
