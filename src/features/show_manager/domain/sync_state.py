"""
Sync State Types

Domain types for bidirectional synchronization between MA3 and Editor.
Uses fingerprint-based event identification to handle index shifts.

Fingerprint Format: "{time:.3f}|{duration:.3f}"
- Time: 3 decimal places (millisecond precision).  MA3 uses frame-based
  timing internally, so microsecond precision causes false mismatches after
  values round-trip through Lua / MA3 serialisation.
- Duration: 3 decimal places (millisecond precision).

name and cmd are excluded from the fingerprint because Editor uses
classification names (clap, kick, ...) while MA3 uses command names (Go+, ...),
so we match by temporal position only.

This allows matching events by content rather than index, which survives
deletions that cause index shifts in MA3.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import time as time_module


class SyncState(Enum):
    """Current sync state for a tracked layer."""
    SYNCED = auto()       # Editor and MA3 match exactly
    PENDING = auto()      # Changes queued, waiting to sync
    CONFLICT = auto()     # States diverged, user resolution needed
    DISCONNECTED = auto() # MA3 not reachable
    SYNCING = auto()      # Sync in progress


class ChangeType(Enum):
    """Type of event change."""
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"  # Time or other property changed


class ConflictType(Enum):
    """Type of sync conflict."""
    EDITOR_ONLY = "editor_only"  # Event exists in Editor but not MA3
    MA3_ONLY = "ma3_only"        # Event exists in MA3 but not Editor
    MODIFIED = "modified"        # Same base event but different properties


class Resolution(Enum):
    """User resolution for a conflict."""
    KEEP_EDITOR = "keep_editor"  # Push Editor version to MA3
    KEEP_MA3 = "keep_ma3"        # Apply MA3 version to Editor
    KEEP_BOTH = "keep_both"      # Add both versions


class SyncStrategy(Enum):
    """
    Strategy for handling sync between Editor and MA3.
    
    Determines how conflicts are resolved and which direction(s) sync flows.
    """
    # Overwrite strategies - one side wins completely
    MA3_OVERWRITES = "ma3_overwrites"
    """MA3 is source of truth - Editor mirrors MA3, no EZ->MA3 push."""
    
    EDITOR_OVERWRITES = "editor_overwrites"
    """Editor is source of truth - MA3 mirrors Editor, no MA3->EZ pull."""
    
    # Merge strategies - keep events from both sides
    MERGE_KEEP_BOTH = "merge_keep_both"
    """Keep all events from both sides (add-only, no deletes propagate)."""
    
    MERGE_MA3_PRIORITY = "merge_ma3_priority"
    """Merge both sides, MA3 wins on conflicts (same event, different properties)."""
    
    MERGE_EDITOR_PRIORITY = "merge_editor_priority"
    """Merge both sides, Editor wins on conflicts (same event, different properties)."""
    
    # Manual - require user decision for each conflict
    MANUAL = "manual"
    """Sync both directions, but pause on conflicts for user resolution."""


@dataclass
class EventFingerprint:
    """
    Content-based event identity that survives index shifts.
    
    Two events with the same fingerprint are considered the same event,
    regardless of their index position.
    """
    time: float
    cmd: str
    name: str
    duration: float = 0.0
    
    def compute(self) -> str:
        """Generate fingerprint string.

        Uses only time and duration for matching (millisecond precision).
        name and cmd are excluded because Editor uses classification names
        (clap, kick, etc.) while MA3 uses command names (Go+, etc.).

        Precision is intentionally limited to 3 decimal places to avoid
        false mismatches caused by floating-point drift when event times
        round-trip through MA3's frame-based timing.
        """
        return f"{self.time:.3f}|{self.duration:.3f}"
    
    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> 'EventFingerprint':
        """Create fingerprint from event dict."""
        return cls(
            time=float(event.get('time', 0)),
            cmd=str(event.get('cmd', '')),
            name=str(event.get('name', '')),
            duration=float(event.get('duration', 0))
        )
    
    @classmethod
    def from_string(cls, fingerprint_str: str) -> 'EventFingerprint':
        """Parse fingerprint from string format."""
        parts = fingerprint_str.split('|')
        if len(parts) >= 3:
            return cls(
                time=float(parts[0]) if parts[0] else 0.0,
                cmd=parts[1] if len(parts) > 1 else '',
                name=parts[2] if len(parts) > 2 else '',
                duration=float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
            )
        return cls(time=0.0, cmd='', name='', duration=0.0)
    
    def __hash__(self):
        return hash(self.compute())
    
    def __eq__(self, other):
        if isinstance(other, EventFingerprint):
            return self.compute() == other.compute()
        return False


@dataclass
class MA3EventData:
    """
    Full MA3 event data including extended properties.
    
    Core properties are required for sync. Extended properties
    provide additional context (cue numbers, trigger types, etc.)
    """
    # Core properties (required for sync)
    time: float
    cmd: str = ""
    name: str = ""
    duration: float = 0.0
    
    # Index/position info
    no: int = 0  # Event number in MA3
    idx: int = 0  # Index in track
    
    # Track location
    tc: int = 0  # Timecode number
    tg: int = 0  # Track group
    track: int = 0  # Track number
    
    # Extended properties (may not always be present)
    cue: Optional[str] = None  # Cue reference
    cue_no: Optional[int] = None  # Cue number
    cue_name: Optional[str] = None  # Cue name
    trigger: Optional[str] = None  # Trigger type
    event_type: Optional[str] = None  # Event type
    fade: Optional[float] = None  # Fade time
    delay: Optional[float] = None  # Delay time
    value: Optional[Any] = None  # Value
    data: Optional[Any] = None  # Additional data
    subtrack_type: str = "CmdSubTrack"  # CmdSubTrack or FaderSubTrack
    
    # Sync metadata
    fingerprint: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MA3EventData':
        """Create from event dict."""
        return cls(
            time=float(data.get('time', 0)),
            cmd=str(data.get('cmd', '')),
            name=str(data.get('name', '')),
            duration=float(data.get('duration', 0)),
            no=int(data.get('no', 0)),
            idx=int(data.get('idx', 0)),
            tc=int(data.get('tc', 0)),
            tg=int(data.get('tg', 0)),
            track=int(data.get('track', 0)),
            cue=data.get('cue'),
            cue_no=int(data['cue_no']) if data.get('cue_no') else None,
            cue_name=data.get('cue_name'),
            trigger=data.get('trigger'),
            event_type=data.get('event_type'),
            fade=float(data['fade']) if data.get('fade') else None,
            delay=float(data['delay']) if data.get('delay') else None,
            value=data.get('value'),
            data=data.get('data'),
            subtrack_type=str(data.get('subtrack_type', 'CmdSubTrack')),
            fingerprint=data.get('fingerprint', ''),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        result = {
            'time': self.time,
            'cmd': self.cmd,
            'name': self.name,
            'duration': self.duration,
            'no': self.no,
            'idx': self.idx,
            'tc': self.tc,
            'tg': self.tg,
            'track': self.track,
            'subtrack_type': self.subtrack_type,
        }
        # Include optional properties if set
        if self.cue is not None:
            result['cue'] = self.cue
        if self.cue_no is not None:
            result['cue_no'] = self.cue_no
        if self.cue_name is not None:
            result['cue_name'] = self.cue_name
        if self.trigger is not None:
            result['trigger'] = self.trigger
        if self.event_type is not None:
            result['event_type'] = self.event_type
        if self.fade is not None:
            result['fade'] = self.fade
        if self.delay is not None:
            result['delay'] = self.delay
        if self.value is not None:
            result['value'] = self.value
        if self.data is not None:
            result['data'] = self.data
        if self.fingerprint:
            result['fingerprint'] = self.fingerprint
        return result
    
    def compute_fingerprint(self) -> str:
        """Compute fingerprint for this event."""
        return EventFingerprint(
            time=self.time,
            cmd=self.cmd,
            name=self.name,
            duration=self.duration
        ).compute()


@dataclass
class SyncChange:
    """
    Represents a change to synchronize.
    
    Used for both MA3->Editor and Editor->MA3 directions.
    """
    change_type: ChangeType
    fingerprint: str
    event_data: Dict[str, Any]
    source: str  # "editor" or "ma3"
    timestamp: float = field(default_factory=time_module.time)
    
    # For updates, track what changed
    old_time: Optional[float] = None
    new_time: Optional[float] = None
    old_values: Optional[Dict[str, Any]] = None  # Previous values for any updated properties
    new_values: Optional[Dict[str, Any]] = None  # New values for any updated properties
    
    @property
    def is_add(self) -> bool:
        return self.change_type == ChangeType.ADD
    
    @property
    def is_delete(self) -> bool:
        return self.change_type == ChangeType.DELETE
    
    @property
    def is_update(self) -> bool:
        return self.change_type == ChangeType.UPDATE


@dataclass
class Conflict:
    """
    Represents a sync conflict requiring user resolution.
    
    A conflict occurs when Editor and MA3 states diverge and automatic
    resolution cannot determine which version to keep.
    """
    id: str  # Unique conflict ID
    track_coord: str  # MA3 track coordinate (e.g., "101.1.1")
    editor_layer_id: str  # Editor layer ID
    conflict_type: ConflictType
    
    # Event data (one may be None depending on conflict type)
    editor_event: Optional[Dict[str, Any]] = None
    ma3_event: Optional[Dict[str, Any]] = None
    
    # Fingerprint for matching
    fingerprint: str = ""
    
    # Resolution state
    resolved: bool = False
    resolution: Optional[Resolution] = None
    resolved_at: Optional[float] = None
    
    # Timestamps
    created_at: float = field(default_factory=time_module.time)
    
    def resolve(self, resolution: Resolution) -> None:
        """Mark conflict as resolved with given resolution."""
        self.resolved = True
        self.resolution = resolution
        self.resolved_at = time_module.time()
    
    @property
    def description(self) -> str:
        """Human-readable description of the conflict."""
        if self.conflict_type == ConflictType.EDITOR_ONLY:
            name = self.editor_event.get('name', 'Event') if self.editor_event else 'Event'
            time = self.editor_event.get('time', 0) if self.editor_event else 0
            return f"'{name}' at {time:.2f}s exists in Editor but not in MA3"
        elif self.conflict_type == ConflictType.MA3_ONLY:
            name = self.ma3_event.get('name', 'Event') if self.ma3_event else 'Event'
            time = self.ma3_event.get('time', 0) if self.ma3_event else 0
            return f"'{name}' at {time:.2f}s exists in MA3 but not in Editor"
        elif self.conflict_type == ConflictType.MODIFIED:
            name = self.editor_event.get('name', 'Event') if self.editor_event else 'Event'
            return f"'{name}' differs between Editor and MA3"
        return "Unknown conflict"


@dataclass
class ComparisonResult:
    """
    Result of comparing Editor and MA3 event states.
    
    Used during resync to identify what needs to be synchronized.
    """
    # Matched events (same fingerprint in both)
    matched: List[tuple] = field(default_factory=list)  # (editor_event, ma3_event) pairs
    
    # Events only in one system
    editor_only: List[Dict[str, Any]] = field(default_factory=list)
    ma3_only: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detected conflicts
    conflicts: List[Conflict] = field(default_factory=list)
    
    @property
    def is_synced(self) -> bool:
        """True if no differences between Editor and MA3."""
        return (
            len(self.editor_only) == 0 and
            len(self.ma3_only) == 0 and
            len(self.conflicts) == 0
        )
    
    @property
    def total_differences(self) -> int:
        """Total number of differences found."""
        return len(self.editor_only) + len(self.ma3_only) + len(self.conflicts)
    
    def summary(self) -> str:
        """Human-readable summary of comparison."""
        if self.is_synced:
            return f"Synced ({len(self.matched)} events match)"
        parts = []
        if self.editor_only:
            parts.append(f"{len(self.editor_only)} Editor-only")
        if self.ma3_only:
            parts.append(f"{len(self.ma3_only)} MA3-only")
        if self.conflicts:
            parts.append(f"{len(self.conflicts)} conflicts")
        return f"Out of sync: {', '.join(parts)}"


@dataclass
class TrackedLayer:
    """
    State for a layer being synchronized.
    
    Maintains the mapping between Editor layer and MA3 track,
    along with sync state and cached event data.
    """
    editor_layer_id: str
    editor_block_id: str
    ma3_coord: str  # Format: "tc.tg.track" (e.g., "101.1.1")
    
    # Sync state
    sync_state: SyncState = SyncState.SYNCED
    
    # Sync strategy (how conflicts are resolved)
    sync_strategy: SyncStrategy = SyncStrategy.MA3_OVERWRITES
    
    # Last known event states (for change detection)
    last_editor_events: List[Dict[str, Any]] = field(default_factory=list)
    last_ma3_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pending changes waiting to sync
    pending_changes: List[SyncChange] = field(default_factory=list)
    
    # Active conflicts for this layer
    conflicts: List[Conflict] = field(default_factory=list)
    
    # Change source tracking (for loop prevention)
    change_source: str = ""  # "editor", "ma3", or "" (none)
    sync_paused: bool = False
    
    # Timestamps
    last_sync_at: Optional[float] = None
    hooked_at: Optional[float] = None
    
    @property
    def tc(self) -> int:
        """Extract timecode number from coord."""
        parts = self.ma3_coord.split('.')
        return int(parts[0]) if parts else 0
    
    @property
    def tg(self) -> int:
        """Extract track group from coord."""
        parts = self.ma3_coord.split('.')
        return int(parts[1]) if len(parts) > 1 else 0
    
    @property
    def track(self) -> int:
        """Extract track number from coord."""
        parts = self.ma3_coord.split('.')
        return int(parts[2]) if len(parts) > 2 else 0
    
    @property
    def has_conflicts(self) -> bool:
        """True if there are unresolved conflicts."""
        return any(not c.resolved for c in self.conflicts)
    
    @property
    def unresolved_conflict_count(self) -> int:
        """Count of unresolved conflicts."""
        return sum(1 for c in self.conflicts if not c.resolved)
    
    def clear_resolved_conflicts(self) -> int:
        """Remove resolved conflicts, return count removed."""
        original_count = len(self.conflicts)
        self.conflicts = [c for c in self.conflicts if not c.resolved]
        return original_count - len(self.conflicts)


def compute_fingerprint(event: Dict[str, Any]) -> str:
    """
    Compute fingerprint for an event dict.
    
    Convenience function that creates EventFingerprint and returns string.
    """
    return EventFingerprint.from_event(event).compute()


def events_match(event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
    """
    Check if two events match by fingerprint.
    
    Returns True if events have the same fingerprint (same content).
    """
    return compute_fingerprint(event1) == compute_fingerprint(event2)


__all__ = [
    'SyncState',
    'ChangeType',
    'ConflictType',
    'Resolution',
    'EventFingerprint',
    'MA3EventData',
    'SyncChange',
    'Conflict',
    'ComparisonResult',
    'TrackedLayer',
    'compute_fingerprint',
    'events_match',
]
