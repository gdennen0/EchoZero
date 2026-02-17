"""
MA3 Event Entity

Represents a single event from grandMA3 timecode system.
Provides conversion to/from EchoZero's TimelineEvent format.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

# MA3 time conversion constants
MA3_SUBFRAMES_PER_SECOND = 16777216  # MA3 uses 16777216 subframes per second


def ma3_time_to_seconds(ma3_time: float) -> float:
    """
    Convert MA3 time value to seconds.
    
    MA3 can return time in different formats:
    - Raw subframes (integer, 16777216 per second)
    - Seconds (float)
    - Timecode string (handled by Lua plugin)
    
    This function handles the case where we receive raw subframes.
    
    Args:
        ma3_time: Time value from MA3 (could be subframes or seconds)
        
    Returns:
        Time in seconds
    """
    # Values >= 1000000 are likely subframes (1 second = 16777216)
    # For integer values that are multiples of MA3_SUBFRAMES_PER_SECOND, they're definitely subframes
    if ma3_time >= 1000000:
        # Likely in subframes, convert to seconds
        return ma3_time / MA3_SUBFRAMES_PER_SECOND
    elif isinstance(ma3_time, int) and ma3_time > 0:
        # Check if it's a multiple of subframes per second (integer check)
        if ma3_time % MA3_SUBFRAMES_PER_SECOND == 0:
            return ma3_time / MA3_SUBFRAMES_PER_SECOND
    # Otherwise assume it's already in seconds
    return ma3_time


def seconds_to_ma3_time(seconds: float) -> int:
    """
    Convert seconds to MA3 subframes.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Time in MA3 subframes (integer)
    """
    return int(seconds * MA3_SUBFRAMES_PER_SECOND)


@dataclass
class MA3Event:
    """
    Represents a single MA3 timecode event.
    
    MA3 events are identified by their position in the timecode hierarchy:
    - Timecode number
    - Track group index
    - Track index
    - Event layer index
    - Event index
    
    This coordinate system uniquely identifies each event in MA3.
    """
    
    # MA3 Coordinates
    timecode_no: int
    track_group: int
    track: int
    event_layer: int
    event_index: int
    
    # Event Properties
    time: float  # Time in seconds
    event_type: str = "cmd"  # "cmd" or "fader"
    name: str = ""
    
    # Optional MA3-specific properties
    cmd: Optional[str] = None
    fade: Optional[float] = None
    delay: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Sync tracking
    last_modified: Optional[datetime] = None
    source: str = "ma3"  # "ma3" or "echozero"
    
    def __post_init__(self):
        """Validate event data."""
        if self.time < 0:
            raise ValueError(f"Event time cannot be negative: {self.time}")
        if self.timecode_no < 1:
            raise ValueError(f"Timecode number must be >= 1: {self.timecode_no}")
    
    @property
    def ma3_id(self) -> str:
        """
        Generate unique ID based on MA3 coordinates.
        
        Format: ma3_tc{tc}_tg{tg}_tr{tr}_el{el}_ev{ev}
        Example: ma3_tc101_tg1_tr1_el1_ev1
        """
        return (f"ma3_tc{self.timecode_no}_tg{self.track_group}_"
                f"tr{self.track}_el{self.event_layer}_ev{self.event_index}")
    
    @property
    def ma3_path(self) -> str:
        """
        MA3 command path for this event.
        
        Format: Timecode {tc}.{tg}.{tr}.{el}.{ev}
        Example: Timecode 101.1.1.1.1
        """
        return (f"Timecode {self.timecode_no}.{self.track_group}."
                f"{self.track}.{self.event_layer}.{self.event_index}")
    
    def to_timeline_event(self, classification: Optional[str] = None, 
                         layer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert MA3Event to EchoZero TimelineEvent format.
        
        Args:
            classification: Override classification (from routing config)
            layer_id: Override layer_id (from routing config)
            
        Returns:
            Dict compatible with TimelineEvent.from_dict()
        """
        # Determine classification
        if classification is None:
            # Default classification based on event type
            if self.event_type == "cmd":
                classification = self.name if self.name else "command"
            else:
                classification = "fader"
        
        # Build TimelineEvent dict
        event_dict = {
            'id': self.ma3_id,
            'time': self.time,
            'duration': 0.0,  # MA3 events are instant by default
            'classification': classification,
            'layer_id': layer_id,
            'user_data': {
                'ma3_source': True,
                'ma3_timecode': self.timecode_no,
                'ma3_track_group': self.track_group,
                'ma3_track': self.track,
                'ma3_event_layer': self.event_layer,
                'ma3_event_index': self.event_index,
                'ma3_type': self.event_type,
                'ma3_name': self.name,
            }
        }
        
        # Add MA3-specific properties to user_data
        if self.cmd:
            event_dict['user_data']['ma3_cmd'] = self.cmd
        if self.fade is not None:
            event_dict['user_data']['ma3_fade'] = self.fade
        if self.delay is not None:
            event_dict['user_data']['ma3_delay'] = self.delay
        
        # Include any additional metadata
        event_dict['user_data']['ma3_metadata'] = self.metadata.copy()
        
        return event_dict
    
    @classmethod
    def from_timeline_event(cls, event_dict: Dict[str, Any], 
                           timecode_no: int,
                           track_group: int,
                           track: int) -> Optional['MA3Event']:
        """
        Convert EchoZero TimelineEvent to MA3Event.
        
        Only works if the event has MA3 source metadata.
        
        Args:
            event_dict: TimelineEvent dict
            timecode_no: Target timecode number
            track_group: Target track group
            track: Target track
            
        Returns:
            MA3Event if conversion is possible, None otherwise
        """
        user_data = event_dict.get('user_data', {})
        
        # Check if this event originated from MA3
        if not user_data.get('ma3_source'):
            # Create new MA3 event from EZ event
            return cls(
                timecode_no=timecode_no,
                track_group=track_group,
                track=track,
                event_layer=1,  # Default to first event layer
                event_index=0,  # Will be assigned by MA3
                time=event_dict['time'],
                event_type='cmd',  # Default to cmd event
                name=event_dict.get('classification', ''),
                source='echozero'
            )
        
        # Reconstruct MA3Event from stored metadata
        return cls(
            timecode_no=user_data.get('ma3_timecode', timecode_no),
            track_group=user_data.get('ma3_track_group', track_group),
            track=user_data.get('ma3_track', track),
            event_layer=user_data.get('ma3_event_layer', 1),
            event_index=user_data.get('ma3_event_index', 0),
            time=event_dict['time'],
            event_type=user_data.get('ma3_type', 'cmd'),
            name=user_data.get('ma3_name', ''),
            cmd=user_data.get('ma3_cmd'),
            fade=user_data.get('ma3_fade'),
            delay=user_data.get('ma3_delay'),
            metadata=user_data.get('ma3_metadata', {}),
            source='ma3'
        )
    
    @classmethod
    def from_ma3_json(cls, data: Dict[str, Any]) -> 'MA3Event':
        """
        Create MA3Event from JSON data received via OSC.
        
        Expected format matches the JSON sent by EZ.GetAllEvents() in Lua.
        The time value should already be converted to seconds by the Lua plugin,
        but we validate and convert if needed as a safety measure.
        """
        time_value = data.get('time', 0.0)
        # Ensure time is in seconds (Lua plugin should have converted, but validate)
        if isinstance(time_value, (int, float)):
            time_value = ma3_time_to_seconds(float(time_value))
        
        return cls(
            timecode_no=data.get('timecode_no', 0),
            track_group=data.get('track_group', 0),
            track=data.get('track', 0),
            event_layer=data.get('event_layer', 0),
            event_index=data.get('event_index', 0),
            time=time_value,
            event_type=data.get('type', 'cmd'),
            name=data.get('name', ''),
            cmd=data.get('cmd'),
            fade=data.get('fade'),
            delay=data.get('delay'),
            metadata=data.copy(),
            source='ma3'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'timecode_no': self.timecode_no,
            'track_group': self.track_group,
            'track': self.track,
            'event_layer': self.event_layer,
            'event_index': self.event_index,
            'time': self.time,
            'event_type': self.event_type,
            'name': self.name,
            'metadata': self.metadata.copy(),
            'source': self.source,
        }
        
        if self.cmd:
            result['cmd'] = self.cmd
        if self.fade is not None:
            result['fade'] = self.fade
        if self.delay is not None:
            result['delay'] = self.delay
        if self.last_modified:
            result['last_modified'] = self.last_modified.isoformat()
        
        return result
    
    def __hash__(self) -> int:
        """Hash based on MA3 coordinates."""
        return hash(self.ma3_id)
    
    def __eq__(self, other) -> bool:
        """Equality based on MA3 coordinates."""
        if not isinstance(other, MA3Event):
            return False
        return self.ma3_id == other.ma3_id
