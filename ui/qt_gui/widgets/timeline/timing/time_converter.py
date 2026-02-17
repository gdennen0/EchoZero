"""
Time Converter

Converts between seconds (internal format) and display units (external format).
All conversions use seconds as the base unit.

Design:
- Pure functions (no side effects)
- Frame rate, BPM are parameters (not stored state)
- Easy to test independently
"""

from typing import Optional
from enum import Enum, auto


class TimeFormat(Enum):
    """Time display format types."""
    SECONDS = auto()      # Simple seconds (0.000)
    TIMECODE = auto()     # hh:mm:ss.ff (frames)
    FRAMES = auto()       # Raw frame count
    MILLISECONDS = auto() # Milliseconds


class TimeConverter:
    """
    Converts between seconds (internal) and display units (external).
    
    All methods are static - pure functions with no state.
    Seconds are always the base unit for internal calculations.
    """
    
    @staticmethod
    def seconds_to_frames(seconds: float, frame_rate: float) -> int:
        """
        Convert seconds to frame number.
        
        Args:
            seconds: Time in seconds
            frame_rate: Frames per second
        
        Returns:
            Frame number (0-based)
        """
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        return int(seconds * frame_rate)
    
    @staticmethod
    def frames_to_seconds(frames: int, frame_rate: float) -> float:
        """
        Convert frame number to seconds.
        
        Args:
            frames: Frame number (0-based)
            frame_rate: Frames per second
        
        Returns:
            Time in seconds
        """
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        return frames / frame_rate
    
    @staticmethod
    def seconds_to_timecode(
        seconds: float,
        frame_rate: float,
        compact: bool = False
    ) -> str:
        """
        Convert seconds to timecode string (hh:mm:ss.ff).
        
        Args:
            seconds: Time in seconds
            frame_rate: Frames per second
            compact: If True, omit leading zero components (e.g., "45.12" instead of "00:00:45.12")
        
        Returns:
            Timecode string (e.g., "01:23:45.12")
        """
        if frame_rate <= 0:
            return f"{seconds:.3f}s"
        
        if seconds < 0:
            sign = "-"
            seconds = abs(seconds)
        else:
            sign = ""
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        # Calculate frames - ensure we don't exceed frame rate
        remaining = seconds - int(seconds)
        frame = int(remaining * frame_rate)
        max_frames = int(frame_rate)
        frame = min(frame, max_frames - 1)
        frame = max(0, frame)  # Ensure non-negative
        
        if compact:
            # Smart display: omit leading zero components
            if hours > 0:
                return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}.{frame:02d}"
            elif minutes > 0:
                return f"{sign}{minutes:02d}:{secs:02d}.{frame:02d}"
            else:
                return f"{sign}{secs:02d}.{frame:02d}"
        else:
            return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}.{frame:02d}"
    
    @staticmethod
    def timecode_to_seconds(timecode: str, frame_rate: float) -> float:
        """
        Convert timecode string to seconds.
        
        Args:
            timecode: Timecode string (e.g., "01:23:45.12" or "01:23:45:12")
            frame_rate: Frames per second
        
        Returns:
            Time in seconds
        
        Raises:
            ValueError: If timecode format is invalid or frame_rate <= 0
        """
        result = TimeConverter.parse_timecode(timecode, frame_rate)
        if result is None:
            raise ValueError(f"Invalid timecode format: {timecode}")
        return result
    
    @staticmethod
    def parse_timecode(timecode: str, frame_rate: float) -> Optional[float]:
        """
        Parse a timecode string to seconds (flexible parsing).
        
        Supports formats:
        - "ss.ff" or "ss:ff" (seconds and frames)
        - "mm:ss.ff" or "mm:ss:ff" (minutes, seconds, frames)
        - "hh:mm:ss.ff" or "hh:mm:ss:ff" (full timecode)
        - Plain seconds "123.456"
        
        Args:
            timecode: Timecode string
            frame_rate: Frames per second
        
        Returns:
            Time in seconds, or None if parsing failed
        """
        if frame_rate <= 0:
            return None
        
        try:
            timecode = timecode.strip()
            
            # Check for negative
            negative = timecode.startswith("-")
            if negative:
                timecode = timecode[1:]
            
            # Handle plain numbers (no colons)
            if '.' in timecode and ':' not in timecode:
                result = float(timecode)
                return -result if negative else result
            
            # Split by : or .
            parts = timecode.replace(".", ":").split(":")
            
            if len(parts) == 2:  # ss:ff or ss.ff
                secs = int(parts[0])
                frames = int(parts[1])
                result = secs + frames / frame_rate
            elif len(parts) == 3:  # mm:ss:ff
                mins = int(parts[0])
                secs = int(parts[1])
                frames = int(parts[2])
                result = mins * 60 + secs + frames / frame_rate
            elif len(parts) == 4:  # hh:mm:ss:ff
                hours = int(parts[0])
                mins = int(parts[1])
                secs = int(parts[2])
                frames = int(parts[3])
                result = hours * 3600 + mins * 60 + secs + frames / frame_rate
            else:
                return None
            
            return -result if negative else result
            
        except (ValueError, IndexError):
            return None
    
    @staticmethod
    def seconds_to_beats(seconds: float, bpm: float) -> float:
        """
        Convert seconds to beats.
        
        Args:
            seconds: Time in seconds
            bpm: Beats per minute
        
        Returns:
            Number of beats
        """
        if bpm <= 0:
            raise ValueError(f"BPM must be positive, got {bpm}")
        return (seconds * bpm) / 60.0
    
    @staticmethod
    def beats_to_seconds(beats: float, bpm: float) -> float:
        """
        Convert beats to seconds.
        
        Args:
            beats: Number of beats
            bpm: Beats per minute
        
        Returns:
            Time in seconds
        """
        if bpm <= 0:
            raise ValueError(f"BPM must be positive, got {bpm}")
        return (beats * 60.0) / bpm
    
    @staticmethod
    def format_time(
        seconds: float,
        format_type: TimeFormat,
        frame_rate: Optional[float] = None,
        compact: bool = True
    ) -> str:
        """
        Format time according to specified format.
        
        Args:
            seconds: Time in seconds
            format_type: Desired output format
            frame_rate: Required for TIMECODE and FRAMES formats
            compact: If True, omit leading zero components in timecode
        
        Returns:
            Formatted time string
        """
        if format_type == TimeFormat.SECONDS:
            return f"{seconds:.3f}s"
        
        elif format_type == TimeFormat.MILLISECONDS:
            ms = seconds * 1000
            return f"{ms:.1f}ms"
        
        elif format_type == TimeFormat.FRAMES:
            if frame_rate is None:
                raise ValueError("frame_rate required for FRAMES format")
            frame = TimeConverter.seconds_to_frames(seconds, frame_rate)
            return f"{frame}f"
        
        elif format_type == TimeFormat.TIMECODE:
            if frame_rate is None:
                raise ValueError("frame_rate required for TIMECODE format")
            return TimeConverter.seconds_to_timecode(seconds, frame_rate, compact=compact)
        
        # Fallback
        return f"{seconds:.3f}s"

