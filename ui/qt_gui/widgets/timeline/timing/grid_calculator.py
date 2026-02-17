"""
Grid Calculator

Calculates grid intervals based on zoom level and unit preferences.
Always returns intervals in seconds (single source of truth).

Design:
- Works purely in seconds internally
- Unit preferences affect which boundaries to align to
- Zoom-adaptive (more lines when zoomed in)
- Applies multipliers while maintaining unit alignment
"""

from typing import Tuple
from .unit_preferences import UnitPreference
from .time_converter import TimeConverter


class GridCalculator:
    """
    Calculates grid intervals based on zoom and unit preferences.
    
    Always returns intervals in seconds. Unit preferences determine
    which boundaries intervals align to (frames, seconds, etc.).
    """
    
    # Target pixel spacing for grid lines
    TARGET_MAJOR_PIXELS = 80
    TARGET_MINOR_PIXELS = 40
    
    def __init__(self, frame_rate: float = 30.0):
        """
        Initialize grid calculator.
        
        Args:
            frame_rate: Frames per second (for frame-based calculations)
        """
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        self.frame_rate = frame_rate
        # Note: Multipliers removed - grid automatically calculated from timebase/FPS
    
    def get_intervals(
        self,
        pixels_per_second: float,
        unit_preference: UnitPreference,
        major_multiplier: float = None,
        minor_multiplier: float = None
    ) -> Tuple[float, float]:
        """
        Calculate major and minor grid intervals.
        
        Note: Grid intervals are automatically calculated from timebase/FPS.
        Multiplier parameters kept for backward compatibility but always use 1.0.
        
        Args:
            pixels_per_second: Current zoom level
            unit_preference: Preferred unit alignment (frames, seconds, etc.)
            major_multiplier: Deprecated - kept for compatibility (always 1.0)
            minor_multiplier: Deprecated - kept for compatibility (always 1.0)
        
        Returns:
            (major_interval, minor_interval) in seconds, always aligned with unit boundaries
        """
        # Ignore multipliers - grid is always based on timebase/FPS
        major_mult = 1.0
        minor_mult = 1.0
        
        # Calculate base intervals based on unit preference
        if unit_preference == UnitPreference.FRAMES:
            major, minor = self._get_frame_intervals(pixels_per_second)
        elif unit_preference == UnitPreference.SECONDS:
            major, minor = self._get_seconds_intervals(pixels_per_second)
        elif unit_preference == UnitPreference.MILLISECONDS:
            major, minor = self._get_milliseconds_intervals(pixels_per_second)
        elif unit_preference == UnitPreference.AUTO:
            # Auto mode: use timecode-style (frames + seconds + minutes)
            major, minor = self._get_timecode_intervals(pixels_per_second)
        else:
            # Fallback to seconds
            major, minor = self._get_seconds_intervals(pixels_per_second)
        
        # Apply multipliers while maintaining unit alignment
        major = self._apply_multiplier(major, major_mult, unit_preference)
        minor = self._apply_multiplier(minor, minor_mult, unit_preference)
        
        # Ensure minor <= major
        if minor > major:
            minor = major
        
        return major, minor
    
    def _get_timecode_intervals(self, pixels_per_second: float) -> Tuple[float, float]:
        """
        Get unit-aligned intervals for timecode mode (frames + seconds + minutes + hours).
        Intervals align with frames, seconds, minutes, hours.
        """
        frame_duration = 1.0 / self.frame_rate
        
        # Calculate desired intervals in seconds
        desired_major = self.TARGET_MAJOR_PIXELS / pixels_per_second
        desired_minor = self.TARGET_MINOR_PIXELS / pixels_per_second
        
        # Timecode unit hierarchy (in seconds):
        # Frame, Second, 10 seconds, Minute, 10 minutes, Hour
        
        # Find appropriate major interval (aligned to units)
        if desired_major >= 3600:  # >= 1 hour
            major = 3600.0  # 1 hour
        elif desired_major >= 600:  # >= 10 minutes
            major = 600.0  # 10 minutes
        elif desired_major >= 60:  # >= 1 minute
            major = 60.0  # 1 minute
        elif desired_major >= 10:  # >= 10 seconds
            major = 10.0  # 10 seconds
        elif desired_major >= 1:  # >= 1 second
            major = 1.0  # 1 second
        elif desired_major >= frame_duration * 10:  # >= 10 frames
            # Round to nearest multiple of 10 frames
            major = round(desired_major / (frame_duration * 10)) * (frame_duration * 10)
            major = max(major, frame_duration * 10)
        elif desired_major >= frame_duration:  # >= 1 frame
            # Round to nearest multiple of frames
            major = round(desired_major / frame_duration) * frame_duration
            major = max(major, frame_duration)
        else:
            major = frame_duration
        
        # Find appropriate minor interval (must be a divisor of major and unit-aligned)
        if major >= 60:  # Major is minute or larger
            # Minor can be: 10 seconds, 1 second, or frame-aligned
            if desired_minor >= 10:
                minor = 10.0
            elif desired_minor >= 1:
                minor = 1.0
            else:
                # Frame-aligned: find largest frame multiple that fits
                frame_multiple = max(1, int(desired_minor / frame_duration))
                minor = frame_multiple * frame_duration
        elif major >= 1:  # Major is second or 10 seconds
            # Minor can be: 1 second, or frame-aligned
            if desired_minor >= 1:
                minor = 1.0
            else:
                # Frame-aligned
                frame_multiple = max(1, int(desired_minor / frame_duration))
                minor = frame_multiple * frame_duration
        else:  # Major is frame-aligned
            # Minor must be frame-aligned and divide major evenly
            if major >= frame_duration * 10:
                # Minor can be: 10 frames, 5 frames, 2 frames, or 1 frame
                if desired_minor >= frame_duration * 10:
                    minor = frame_duration * 10
                elif desired_minor >= frame_duration * 5:
                    minor = frame_duration * 5
                elif desired_minor >= frame_duration * 2:
                    minor = frame_duration * 2
                else:
                    minor = frame_duration
            elif major >= frame_duration * 5:
                # Minor can be: 5 frames, 2 frames, or 1 frame
                if desired_minor >= frame_duration * 5:
                    minor = frame_duration * 5
                elif desired_minor >= frame_duration * 2:
                    minor = frame_duration * 2
                else:
                    minor = frame_duration
            elif major >= frame_duration * 2:
                # Minor can be: 2 frames or 1 frame
                if desired_minor >= frame_duration * 2:
                    minor = frame_duration * 2
                else:
                    minor = frame_duration
            else:
                minor = frame_duration
        
        # Ensure minor divides major evenly (with floating point tolerance)
        if abs(major % minor) > 0.0001 and abs(major - (major // minor) * minor) > 0.0001:
            # Find largest divisor of major that's <= minor and unit-aligned
            if major >= 1.0:
                # Try seconds
                if minor >= 1.0 and major >= minor:
                    # minor is already <= major, try to find a divisor
                    if major >= 10.0 and minor < 10.0:
                        minor = 10.0
                    elif major >= 1.0 and minor < 1.0:
                        minor = 1.0
                else:
                    # Use frame-aligned
                    minor = frame_duration
            else:
                # Frame-aligned: use single frame
                minor = frame_duration
        
        return major, minor
    
    def _get_frame_intervals(self, pixels_per_second: float) -> Tuple[float, float]:
        """
        Get unit-aligned intervals for FRAMES mode.
        Intervals align strictly to frame boundaries.
        """
        frame_duration = 1.0 / self.frame_rate
        
        desired_major = self.TARGET_MAJOR_PIXELS / pixels_per_second
        desired_minor = self.TARGET_MINOR_PIXELS / pixels_per_second
        
        # Frame unit hierarchy: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 frames
        frame_units = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        
        # Find appropriate major interval (in frames)
        desired_major_frames = desired_major / frame_duration
        major_frames = 1
        for unit in frame_units:
            if desired_major_frames >= unit:
                major_frames = unit
            else:
                break
        
        # Find appropriate minor interval (must divide major evenly)
        desired_minor_frames = desired_minor / frame_duration
        minor_frames = 1
        
        # Find largest divisor of major_frames that's <= desired_minor_frames
        for unit in frame_units:
            if major_frames % unit == 0 and desired_minor_frames >= unit:
                minor_frames = unit
        
        return major_frames * frame_duration, minor_frames * frame_duration
    
    def _get_seconds_intervals(self, pixels_per_second: float) -> Tuple[float, float]:
        """
        Get unit-aligned intervals for SECONDS mode.
        Intervals align with second boundaries and sub-second units.
        """
        desired_major = self.TARGET_MAJOR_PIXELS / pixels_per_second
        desired_minor = self.TARGET_MINOR_PIXELS / pixels_per_second
        
        # Second unit hierarchy: 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 30, 60
        second_units = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        
        # Find appropriate major interval
        major = 0.001
        for unit in second_units:
            if desired_major >= unit:
                major = unit
            else:
                break
        
        # Find appropriate minor interval (must divide major evenly)
        minor = 0.001
        for unit in second_units:
            if major >= unit and (major % unit < 0.0001 or unit == major):  # unit divides major
                if desired_minor >= unit:
                    minor = unit
        
        return major, minor
    
    def _get_milliseconds_intervals(self, pixels_per_second: float) -> Tuple[float, float]:
        """
        Get unit-aligned intervals for MILLISECONDS mode.
        Intervals align with millisecond boundaries.
        """
        ms_duration = 0.001  # 1 millisecond in seconds
        
        desired_major = self.TARGET_MAJOR_PIXELS / pixels_per_second
        desired_minor = self.TARGET_MINOR_PIXELS / pixels_per_second
        
        # Millisecond unit hierarchy: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 ms
        ms_units = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        
        # Find appropriate major interval (in milliseconds)
        desired_major_ms = desired_major / ms_duration
        major_ms = 1
        for unit in ms_units:
            if desired_major_ms >= unit:
                major_ms = unit
            else:
                break
        
        # Find appropriate minor interval (must divide major evenly)
        desired_minor_ms = desired_minor / ms_duration
        minor_ms = 1
        
        for unit in ms_units:
            if major_ms % unit == 0 and desired_minor_ms >= unit:
                minor_ms = unit
        
        return major_ms * ms_duration, minor_ms * ms_duration
    
    def _apply_multiplier(
        self,
        interval: float,
        multiplier: float,
        unit_preference: UnitPreference
    ) -> float:
        """
        Apply multiplier to a unit-aligned interval while maintaining unit alignment.
        
        For multipliers < 1.0, finds the next smaller unit-aligned interval.
        For multipliers > 1.0, multiplies then rounds to nearest unit-aligned value.
        
        Args:
            interval: Current unit-aligned interval in seconds
            multiplier: Multiplier to apply (0.1 to 10.0)
            unit_preference: Current unit preference (for unit-aware snapping)
        
        Returns:
            New unit-aligned interval in seconds
        """
        if abs(multiplier - 1.0) < 0.0001:
            return interval
        
        # Validate multiplier
        multiplier = max(0.1, min(10.0, multiplier))
        
        # Calculate desired interval
        desired = interval * multiplier
        
        # Recalculate using the same unit-aware logic but with desired value
        # This ensures we get a unit-aligned result
        if unit_preference == UnitPreference.FRAMES:
            return self._snap_to_frame_unit(desired)
        elif unit_preference == UnitPreference.SECONDS:
            return self._snap_to_second_unit(desired)
        elif unit_preference == UnitPreference.MILLISECONDS:
            return self._snap_to_millisecond_unit(desired)
        elif unit_preference == UnitPreference.AUTO:
            return self._snap_to_timecode_unit(desired)
        else:
            return desired
    
    def _snap_to_timecode_unit(self, interval: float) -> float:
        """Snap interval to nearest timecode unit (frame, second, minute, hour)."""
        frame_duration = 1.0 / self.frame_rate
        
        # Timecode units in order of preference
        units = [
            3600.0,  # 1 hour
            600.0,   # 10 minutes
            60.0,    # 1 minute
            10.0,    # 10 seconds
            1.0,     # 1 second
            frame_duration * 10,  # 10 frames
            frame_duration * 5,   # 5 frames
            frame_duration * 2,   # 2 frames
            frame_duration,       # 1 frame
        ]
        
        # Find closest unit
        best = units[-1]  # Default to 1 frame
        for unit in units:
            if abs(interval - unit) < abs(interval - best):
                best = unit
            # Also check if interval is close to a multiple
            if interval >= unit:
                multiple = round(interval / unit)
                candidate = multiple * unit
                if abs(interval - candidate) < abs(interval - best):
                    best = candidate
        
        return max(frame_duration, best)  # Never smaller than 1 frame
    
    def _snap_to_frame_unit(self, interval: float) -> float:
        """Snap interval to nearest frame unit."""
        frame_duration = 1.0 / self.frame_rate
        frames = round(interval / frame_duration)
        frames = max(1, frames)  # At least 1 frame
        return frames * frame_duration
    
    def _snap_to_second_unit(self, interval: float) -> float:
        """Snap interval to nearest second unit."""
        units = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        
        best = units[0]
        for unit in units:
            if abs(unit - interval) < abs(best - interval):
                best = unit
            # Check multiples
            if interval >= unit:
                multiple = round(interval / unit)
                candidate = multiple * unit
                if abs(interval - candidate) < abs(interval - best):
                    best = candidate
        
        return max(0.001, best)  # Never smaller than 1ms
    
    def _snap_to_millisecond_unit(self, interval: float) -> float:
        """Snap interval to nearest millisecond unit."""
        ms_duration = 0.001
        ms = round(interval / ms_duration)
        ms = max(1, ms)  # At least 1ms
        return ms * ms_duration

