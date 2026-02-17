"""
Timeline Interfaces

Protocol definitions for timeline integration points.
These allow the timeline to work with various audio backends and event sources.

Note: Event editing is handled via Qt signals (event_moved, event_resized, etc.)
rather than a callback interface, following Qt conventions.
"""

from typing import Protocol, List, Dict, Any, runtime_checkable


@runtime_checkable
class PlaybackInterface(Protocol):
    """
    Protocol for audio playback engines.
    
    Implement this interface to connect your audio player to the timeline.
    The timeline will query position and control playback through this interface.
    """
    
    def get_position(self) -> float:
        """
        Get current playback position in seconds.
        
        Returns:
            Current position in seconds (float)
        """
        ...
    
    def set_position(self, seconds: float) -> None:
        """
        Seek to a specific position.
        
        Args:
            seconds: Target position in seconds
        """
        ...
    
    def play(self) -> None:
        """Start or resume playback."""
        ...
    
    def pause(self) -> None:
        """Pause playback."""
        ...
    
    def stop(self) -> None:
        """Stop playback and reset position to start."""
        ...
    
    def is_playing(self) -> bool:
        """
        Check if currently playing.
        
        Returns:
            True if playing, False if paused/stopped
        """
        ...
    
    def get_duration(self) -> float:
        """
        Get total duration in seconds.
        
        Returns:
            Total duration in seconds
        """
        ...


@runtime_checkable
class EventSourceInterface(Protocol):
    """
    Protocol for event data providers.
    
    Implement this to provide events to the timeline from various sources.
    """
    
    def get_events(self) -> List[Dict[str, Any]]:
        """
        Get all events.
        
        Returns:
            List of event dictionaries with keys:
            - 'id': str (unique identifier)
            - 'time': float (start time in seconds)
            - 'duration': float (duration in seconds, 0 for markers)
            - 'classification': str (layer/category name)
            - 'metadata': dict (optional additional data)
        """
        ...
    
    def get_duration(self) -> float:
        """
        Get total timeline duration.
        
        Returns:
            Duration in seconds
        """
        ...
    
    def get_layers(self) -> List[str]:
        """
        Get list of layer/track names.
        
        Returns:
            List of layer names in display order
        """
        ...

