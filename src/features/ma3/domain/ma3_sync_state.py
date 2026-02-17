"""
MA3 Sync State Entity

Tracks synchronization state between MA3 and EchoZero.
Manages conflict detection and resolution.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum, auto


class SyncDirection(Enum):
    """Direction of synchronization."""
    MA3_TO_EZ = auto()  # MA3 is source, EZ is target
    EZ_TO_MA3 = auto()  # EZ is source, MA3 is target
    BIDIRECTIONAL = auto()  # Both directions active


class ConflictResolution(Enum):
    """How to resolve conflicts."""
    USE_MA3 = auto()  # MA3 version wins
    USE_EZ = auto()  # EchoZero version wins
    MERGE = auto()  # Attempt to merge changes
    SKIP = auto()  # Skip this event, leave as-is
    PROMPT_USER = auto()  # Ask user to decide


class ChangeType(Enum):
    """Type of change detected."""
    ADDED = auto()
    MODIFIED = auto()
    DELETED = auto()
    MOVED = auto()  # Time changed


@dataclass
class ConflictRecord:
    """
    Records a conflict between MA3 and EchoZero versions of an event.
    
    A conflict occurs when both sides have modified the same event
    since the last sync.
    """
    
    # Event identification
    event_id: str  # MA3 ID or EZ ID
    
    # Conflicting versions
    ma3_version: Optional[Dict[str, Any]] = None  # MA3Event.to_dict()
    ez_version: Optional[Dict[str, Any]] = None  # TimelineEvent dict
    
    # Conflict metadata
    detected_at: datetime = field(default_factory=datetime.now)
    conflict_type: str = "modification"  # "modification", "deletion", "both_added"
    
    # Resolution
    resolution: Optional[ConflictResolution] = None
    resolved_at: Optional[datetime] = None
    resolved_by: str = "user"  # "user", "auto", "policy"
    
    # Differences
    differences: List[str] = field(default_factory=list)  # List of differing properties
    
    def __post_init__(self):
        """Calculate differences."""
        if self.ma3_version and self.ez_version:
            self._calculate_differences()
    
    def _calculate_differences(self):
        """Calculate which properties differ between versions."""
        if not self.ma3_version or not self.ez_version:
            return
        
        # Compare time
        ma3_time = self.ma3_version.get('time', 0)
        ez_time = self.ez_version.get('time', 0)
        if abs(ma3_time - ez_time) > 0.001:  # 1ms tolerance
            self.differences.append(f"time (MA3: {ma3_time:.3f}s, EZ: {ez_time:.3f}s)")
        
        # Compare classification/name
        ma3_name = self.ma3_version.get('name', '')
        ez_class = self.ez_version.get('classification', '')
        if ma3_name != ez_class:
            self.differences.append(f"name/classification (MA3: '{ma3_name}', EZ: '{ez_class}')")
        
        # Compare MA3-specific properties
        ma3_cmd = self.ma3_version.get('cmd')
        if ma3_cmd:
            self.differences.append(f"MA3 command: {ma3_cmd}")
    
    @property
    def is_resolved(self) -> bool:
        """Whether this conflict has been resolved."""
        return self.resolution is not None
    
    @property
    def summary(self) -> str:
        """Human-readable summary of the conflict."""
        if not self.differences:
            return f"Conflict on event {self.event_id}"
        return f"Conflict on event {self.event_id}: {', '.join(self.differences[:3])}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'ma3_version': self.ma3_version,
            'ez_version': self.ez_version,
            'detected_at': self.detected_at.isoformat(),
            'conflict_type': self.conflict_type,
            'resolution': self.resolution.name if self.resolution else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'differences': self.differences,
        }


@dataclass
class EventSyncState:
    """
    Tracks sync state for a single event.
    
    Maintains checksums and timestamps to detect changes.
    """
    
    event_id: str  # MA3 ID or EZ ID
    
    # Last known state
    last_ma3_checksum: Optional[str] = None
    last_ez_checksum: Optional[str] = None
    
    # Timestamps
    last_ma3_sync: Optional[datetime] = None
    last_ez_sync: Optional[datetime] = None
    last_bidirectional_sync: Optional[datetime] = None
    
    # Change tracking
    pending_ma3_changes: List[ChangeType] = field(default_factory=list)
    pending_ez_changes: List[ChangeType] = field(default_factory=list)
    
    # Conflict tracking
    has_conflict: bool = False
    conflict_record: Optional[ConflictRecord] = None
    
    def mark_synced(self, direction: SyncDirection, checksum: str):
        """Mark event as synced in given direction."""
        now = datetime.now()
        
        if direction == SyncDirection.MA3_TO_EZ:
            self.last_ma3_checksum = checksum
            self.last_ma3_sync = now
            self.pending_ma3_changes.clear()
        elif direction == SyncDirection.EZ_TO_MA3:
            self.last_ez_checksum = checksum
            self.last_ez_sync = now
            self.pending_ez_changes.clear()
        elif direction == SyncDirection.BIDIRECTIONAL:
            self.last_ma3_checksum = checksum
            self.last_ez_checksum = checksum
            self.last_bidirectional_sync = now
            self.pending_ma3_changes.clear()
            self.pending_ez_changes.clear()
        
        # Clear conflict if resolved
        if not self.pending_ma3_changes and not self.pending_ez_changes:
            self.has_conflict = False
    
    def detect_change(self, source: str, new_checksum: str) -> bool:
        """
        Detect if event has changed.
        
        Args:
            source: "ma3" or "ez"
            new_checksum: Current checksum
            
        Returns:
            True if change detected
        """
        if source == "ma3":
            if self.last_ma3_checksum is None:
                return True  # First time seeing this event
            return new_checksum != self.last_ma3_checksum
        else:  # ez
            if self.last_ez_checksum is None:
                return True
            return new_checksum != self.last_ez_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'last_ma3_checksum': self.last_ma3_checksum,
            'last_ez_checksum': self.last_ez_checksum,
            'last_ma3_sync': self.last_ma3_sync.isoformat() if self.last_ma3_sync else None,
            'last_ez_sync': self.last_ez_sync.isoformat() if self.last_ez_sync else None,
            'last_bidirectional_sync': self.last_bidirectional_sync.isoformat() if self.last_bidirectional_sync else None,
            'has_conflict': self.has_conflict,
            'conflict_record': self.conflict_record.to_dict() if self.conflict_record else None,
        }


@dataclass
class MA3SyncState:
    """
    Overall synchronization state between MA3 and EchoZero.
    
    Tracks all events and their sync status.
    """
    
    # Configuration
    timecode_no: int
    editor_block_id: str
    sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    
    # Event tracking
    event_states: Dict[str, EventSyncState] = field(default_factory=dict)
    
    # Conflict management
    conflicts: List[ConflictRecord] = field(default_factory=list)
    
    # Statistics
    last_full_sync: Optional[datetime] = None
    total_syncs: int = 0
    total_conflicts: int = 0
    total_conflicts_resolved: int = 0
    
    def get_or_create_event_state(self, event_id: str) -> EventSyncState:
        """Get existing event state or create new one."""
        if event_id not in self.event_states:
            self.event_states[event_id] = EventSyncState(event_id=event_id)
        return self.event_states[event_id]
    
    def add_conflict(self, conflict: ConflictRecord):
        """Add a new conflict."""
        self.conflicts.append(conflict)
        self.total_conflicts += 1
        
        # Update event state
        event_state = self.get_or_create_event_state(conflict.event_id)
        event_state.has_conflict = True
        event_state.conflict_record = conflict
    
    def resolve_conflict(self, event_id: str, resolution: ConflictResolution):
        """Mark conflict as resolved."""
        event_state = self.event_states.get(event_id)
        if event_state and event_state.conflict_record:
            event_state.conflict_record.resolution = resolution
            event_state.conflict_record.resolved_at = datetime.now()
            event_state.has_conflict = False
            self.total_conflicts_resolved += 1
    
    @property
    def pending_conflicts(self) -> List[ConflictRecord]:
        """Get all unresolved conflicts."""
        return [c for c in self.conflicts if not c.is_resolved]
    
    @property
    def has_pending_conflicts(self) -> bool:
        """Whether there are unresolved conflicts."""
        return len(self.pending_conflicts) > 0
    
    def mark_full_sync(self):
        """Mark a full synchronization completed."""
        self.last_full_sync = datetime.now()
        self.total_syncs += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timecode_no': self.timecode_no,
            'editor_block_id': self.editor_block_id,
            'sync_direction': self.sync_direction.name,
            'event_states': {k: v.to_dict() for k, v in self.event_states.items()},
            'conflicts': [c.to_dict() for c in self.conflicts],
            'last_full_sync': self.last_full_sync.isoformat() if self.last_full_sync else None,
            'total_syncs': self.total_syncs,
            'total_conflicts': self.total_conflicts,
            'total_conflicts_resolved': self.total_conflicts_resolved,
        }
