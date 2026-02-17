"""
OSC Message Domain Model

Core dataclasses for OSC message handling.
Simple, immutable, and reusable across the application.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class MessageType(Enum):
    """Types of OSC messages from MA3."""
    TRACKGROUPS = "trackgroups"
    TRACKGROUP = "trackgroup"    # Single track group (for hooks)
    TRACKS = "tracks"
    TRACK = "track"              # Single track (for hooks)
    SUBTRACK = "subtrack"        # CmdSubTrack (for hook confirmation with events)
    EVENTS = "events"
    EVENT = "event"              # Single event (for individual event notifications)
    CONNECTION = "connection"
    TIMECODES = "timecodes"
    HOOKS = "hooks"              # Hook listing
    HOOK_TEST = "hook_test"      # Hook test results
    SEQUENCE = "sequence"        # Sequence operations
    ERROR = "error"
    UNKNOWN = "unknown"


class ChangeType(Enum):
    """Types of changes in OSC messages."""
    LIST = "list"
    ALL = "all"
    ADDED = "added"
    UPDATED = "updated"
    DELETED = "deleted"
    CLEARED = "cleared"
    CHANGED = "changed"          # Hook notification: track changed
    HOOKED = "hooked"            # Hook registered
    UNHOOKED = "unhooked"        # Hook removed
    UNHOOKED_ALL = "unhooked_all"  # All hooks removed
    SUCCESS = "success"          # Hook test success
    FAILED = "failed"            # Hook test failed
    PING = "ping"
    STATUS = "status"
    ERROR = "error"
    TRACE = "trace"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OSCMessage:
    """
    Parsed OSC message from MA3.
    
    Immutable dataclass representing a structured OSC message.
    All MA3 messages follow the pipe-delimited format:
        type=X|change=Y|timestamp=Z|...data fields...
    
    Attributes:
        address: OSC address (e.g., /ez/message)
        message_type: Type of message (trackgroups, tracks, events, etc.)
        change_type: Type of change (list, added, deleted, etc.)
        timestamp: Unix timestamp from MA3
        data: Additional data fields as a dictionary
        raw: Original raw message string (for debugging)
    """
    address: str
    message_type: MessageType
    change_type: ChangeType
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    
    @property
    def type_key(self) -> str:
        """Get handler key for routing (e.g., 'trackgroups.list')."""
        return f"{self.message_type.value}.{self.change_type.value}"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dictionary."""
        return self.data.get(key, default)
    
    def __repr__(self) -> str:
        return f"OSCMessage({self.type_key}, data_keys={list(self.data.keys())})"


@dataclass
class TrackGroupData:
    """Parsed track group data from MA3."""
    no: int
    name: str
    track_count: int = 0


@dataclass
class TrackData:
    """Parsed track data from MA3."""
    no: int
    name: str


@dataclass
class EventData:
    """Parsed event data from MA3."""
    no: int
    time: float
    duration: float = 0.0
    cmd: str = ""
    name: str = ""
    cue: Optional[str] = None
    subtrack_type: str = ""
