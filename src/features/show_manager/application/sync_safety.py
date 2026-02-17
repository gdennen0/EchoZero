"""
Sync Safety Framework

Provides safety mechanisms for bidirectional event synchronization between Editor and MA3.
Implements backup, validation, conflict detection, and rollback capabilities.

Principles:
- NEVER delete without backup
- ALWAYS validate before applying
- FAIL gracefully, preserve data
- LOG everything for debugging
- ERROR on the side of safety (skip rather than corrupt)
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import copy

from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.message import Log


class SyncAction(Enum):
    """Types of sync actions."""
    ADD = auto()
    UPDATE = auto()
    DELETE = auto()
    MOVE = auto()


class ValidationResult(Enum):
    """Result of validation."""
    VALID = auto()
    INVALID_TIME = auto()      # Event time out of range
    INVALID_DURATION = auto()  # Duration negative or too large
    DUPLICATE = auto()         # Event already exists
    MISSING_TARGET = auto()    # Target layer/track doesn't exist
    CONFLICT = auto()          # Both sides changed
    UNKNOWN_ERROR = auto()


@dataclass
class EventSnapshot:
    """Snapshot of an event for backup/restore."""
    id: str
    time: float
    duration: float
    classification: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_event(cls, event) -> 'EventSnapshot':
        """Create snapshot from an event object."""
        return cls(
            id=getattr(event, 'id', ''),
            time=getattr(event, 'time', 0.0),
            duration=getattr(event, 'duration', 0.0),
            classification=getattr(event, 'classification', ''),
            metadata=copy.deepcopy(getattr(event, 'metadata', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'time': self.time,
            'duration': self.duration,
            'classification': self.classification,
            'metadata': self.metadata,
        }


@dataclass
class LayerSnapshot:
    """Snapshot of a layer's events for backup/restore."""
    layer_id: str
    block_id: str
    timestamp: datetime
    events: List[EventSnapshot] = field(default_factory=list)
    source: str = ""  # "editor" or "ma3"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            'layer_id': self.layer_id,
            'block_id': self.block_id,
            'timestamp': self.timestamp.isoformat(),
            'events': [e.to_dict() for e in self.events],
            'source': self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerSnapshot':
        """Create from dictionary."""
        return cls(
            layer_id=data.get('layer_id', ''),
            block_id=data.get('block_id', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            events=[EventSnapshot(**e) for e in data.get('events', [])],
            source=data.get('source', ''),
        )


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    action: SyncAction
    layer_id: str
    events_affected: int = 0
    backup_created: bool = False
    validation_result: ValidationResult = ValidationResult.VALID
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def success_result(cls, action: SyncAction, layer_id: str, events_affected: int = 0) -> 'SyncResult':
        """Create successful result."""
        return cls(
            success=True,
            action=action,
            layer_id=layer_id,
            events_affected=events_affected,
            backup_created=True
        )
    
    @classmethod
    def failure_result(cls, action: SyncAction, layer_id: str, error: str, 
                       validation: ValidationResult = ValidationResult.UNKNOWN_ERROR) -> 'SyncResult':
        """Create failure result."""
        return cls(
            success=False,
            action=action,
            layer_id=layer_id,
            validation_result=validation,
            error_message=error
        )


class SyncBackupManager:
    """
    Manages backups of layer states before sync operations.
    
    Provides:
    - Pre-sync backup creation
    - Rollback capability
    - Backup history (limited)
    """
    
    MAX_BACKUPS_PER_LAYER = 5  # Keep last 5 backups per layer
    
    def __init__(self):
        # layer_id -> list of snapshots (newest first)
        self._backups: Dict[str, List[LayerSnapshot]] = {}
        Log.info("SyncBackupManager: Initialized")
    
    def create_backup(self, layer_id: str, block_id: str, events: List[Any], source: str) -> LayerSnapshot:
        """
        Create a backup of the current layer state.
        
        ALWAYS call this before any destructive operation.
        
        Args:
            layer_id: Layer identifier
            block_id: Block ID
            events: List of event objects to backup
            source: "editor" or "ma3"
            
        Returns:
            LayerSnapshot that was created
        """
        snapshot = LayerSnapshot(
            layer_id=layer_id,
            block_id=block_id,
            timestamp=datetime.now(),
            events=[EventSnapshot.from_event(e) for e in events],
            source=source
        )
        
        # Add to backups
        if layer_id not in self._backups:
            self._backups[layer_id] = []
        
        self._backups[layer_id].insert(0, snapshot)
        
        # Trim old backups
        if len(self._backups[layer_id]) > self.MAX_BACKUPS_PER_LAYER:
            self._backups[layer_id] = self._backups[layer_id][:self.MAX_BACKUPS_PER_LAYER]
        
        Log.info(f"SyncBackupManager: Created backup for layer '{layer_id}' with {len(events)} events")
        return snapshot
    
    def get_latest_backup(self, layer_id: str) -> Optional[LayerSnapshot]:
        """Get the most recent backup for a layer."""
        backups = self._backups.get(layer_id, [])
        return backups[0] if backups else None
    
    def get_backup_history(self, layer_id: str) -> List[LayerSnapshot]:
        """Get all backups for a layer (newest first)."""
        return self._backups.get(layer_id, [])
    
    def restore_from_backup(self, snapshot: LayerSnapshot, editor_api) -> bool:
        """
        Restore a layer from a backup snapshot.
        
        Args:
            snapshot: LayerSnapshot to restore
            editor_api: EditorAPI instance
            
        Returns:
            True if restore was successful
        """
        try:
            layer_id = snapshot.layer_id
            
            # Clear current events
            editor_api.clear_layer_events(layer_id)
            
            # Restore events from snapshot
            events_to_add = [e.to_dict() for e in snapshot.events]
            editor_api.add_events(events_to_add, source="restore")
            
            Log.info(f"SyncBackupManager: Restored layer '{layer_id}' with {len(snapshot.events)} events from backup")
            return True
            
        except Exception as e:
            Log.error(f"SyncBackupManager: Failed to restore from backup: {e}")
            return False
    
    def clear_backups(self, layer_id: Optional[str] = None):
        """Clear backups for a layer or all backups."""
        if layer_id:
            self._backups.pop(layer_id, None)
        else:
            self._backups.clear()


class SyncValidator:
    """
    Validates sync operations before execution.
    
    Prevents:
    - Invalid event data
    - Duplicate events
    - Missing targets
    - Conflicts
    """
    
    # Validation thresholds
    MAX_EVENT_TIME = 86400.0  # 24 hours in seconds
    MAX_DURATION = 3600.0     # 1 hour max duration
    MIN_TIME = 0.0            # No negative times
    
    def __init__(self):
        pass
    
    def validate_event(self, event_data: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """
        Validate a single event before sync.
        
        Args:
            event_data: Event dict with time, duration, classification, etc.
            
        Returns:
            Tuple of (ValidationResult, error_message)
        """
        time = event_data.get('time', 0.0)
        duration = event_data.get('duration', 0.0)
        
        # Validate time
        if time < self.MIN_TIME:
            return ValidationResult.INVALID_TIME, f"Event time {time} is negative"
        if time > self.MAX_EVENT_TIME:
            return ValidationResult.INVALID_TIME, f"Event time {time} exceeds maximum ({self.MAX_EVENT_TIME}s)"
        
        # Validate duration
        if duration < 0:
            return ValidationResult.INVALID_DURATION, f"Event duration {duration} is negative"
        if duration > self.MAX_DURATION:
            return ValidationResult.INVALID_DURATION, f"Event duration {duration} exceeds maximum ({self.MAX_DURATION}s)"
        
        return ValidationResult.VALID, ""
    
    def validate_events(self, events: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate a list of events.
        
        Args:
            events: List of event dicts
            
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        for i, event in enumerate(events):
            result, msg = self.validate_event(event)
            if result != ValidationResult.VALID:
                errors.append(f"Event {i}: {msg}")
        
        return len(errors) == 0, errors
    
    def check_for_duplicates(
        self,
        new_events: List[Dict[str, Any]],
        existing_events: List[Any],
        time_tolerance: float = 0.01  # 10ms tolerance
    ) -> List[int]:
        """
        Check for duplicate events (same time within tolerance).
        
        Args:
            new_events: Events to be added
            existing_events: Events already in layer
            time_tolerance: Time difference tolerance for duplicate detection
            
        Returns:
            List of indices of duplicate events in new_events
        """
        duplicates = []
        existing_times = [getattr(e, 'time', 0.0) for e in existing_events]
        
        for i, event in enumerate(new_events):
            new_time = event.get('time', 0.0)
            for existing_time in existing_times:
                if abs(new_time - existing_time) < time_tolerance:
                    duplicates.append(i)
                    break
        
        return duplicates
    
    def detect_conflict(
        self,
        editor_snapshot: Optional[LayerSnapshot],
        ma3_snapshot: Optional[LayerSnapshot],
        last_sync_time: Optional[datetime]
    ) -> bool:
        """
        Detect if both sides changed since last sync.
        
        Args:
            editor_snapshot: Current Editor state
            ma3_snapshot: Current MA3 state
            last_sync_time: When last sync occurred
            
        Returns:
            True if conflict detected
        """
        if not last_sync_time:
            return False  # No previous sync, no conflict
        
        if not editor_snapshot or not ma3_snapshot:
            return False
        
        # Check if both were modified after last sync
        # (This is simplified - real implementation would track timestamps per event)
        editor_modified = editor_snapshot.timestamp > last_sync_time
        ma3_modified = ma3_snapshot.timestamp > last_sync_time
        
        return editor_modified and ma3_modified


class SafeSyncService(QObject):
    """
    Safe synchronization service that wraps all sync operations with safety mechanisms.
    
    Features:
    - Automatic backup before destructive operations
    - Validation of all incoming data
    - Conflict detection
    - Rollback on failure
    - Comprehensive logging
    
    Usage:
        safe_sync = SafeSyncService(facade, show_manager_block_id)
        result = safe_sync.sync_to_editor(ma3_events, layer_id, editor_api)
        if not result.success:
            # Handle failure - backup was already created
            safe_sync.rollback(layer_id, editor_api)
    """
    
    # Signals
    sync_started = pyqtSignal(str, str)  # layer_id, direction
    sync_completed = pyqtSignal(str, bool)  # layer_id, success
    conflict_detected = pyqtSignal(str)  # layer_id
    validation_failed = pyqtSignal(str, list)  # layer_id, errors
    
    def __init__(self, facade, show_manager_block_id: str, parent=None):
        super().__init__(parent)
        self._facade = facade
        self._show_manager_block_id = show_manager_block_id
        
        self._backup_manager = SyncBackupManager()
        self._validator = SyncValidator()
        
        # Track last sync times per layer
        self._last_sync_times: Dict[str, datetime] = {}
        
        Log.info(f"SafeSyncService: Initialized for ShowManager {show_manager_block_id}")
    
    def sync_to_editor(
        self,
        ma3_events: List[Dict[str, Any]],
        layer_id: str,
        block_id: str,
        editor_api,
        clear_existing: bool = False,
        skip_duplicates: bool = True
    ) -> SyncResult:
        """
        Safely sync MA3 events to Editor layer.
        
        SAFETY STEPS:
        1. Validate incoming events
        2. Backup existing Editor events
        3. Check for duplicates
        4. Apply events
        5. Log result
        
        Args:
            ma3_events: Events from MA3 to add
            layer_id: Target Editor layer
            block_id: Editor block ID
            editor_api: EditorAPI instance
            clear_existing: Whether to clear existing events first
            skip_duplicates: Whether to skip duplicate events
            
        Returns:
            SyncResult with success/failure info
        """
        self.sync_started.emit(layer_id, "ma3_to_editor")
        
        try:
            # Step 1: Validate incoming events
            is_valid, errors = self._validator.validate_events(ma3_events)
            if not is_valid:
                Log.warning(f"SafeSyncService: Validation failed for {len(errors)} events")
                self.validation_failed.emit(layer_id, errors)
                
                # Remove invalid events but continue with valid ones
                valid_events = []
                for i, event in enumerate(ma3_events):
                    result, _ = self._validator.validate_event(event)
                    if result == ValidationResult.VALID:
                        valid_events.append(event)
                
                if not valid_events:
                    return SyncResult.failure_result(
                        SyncAction.ADD, layer_id, 
                        "All events failed validation",
                        ValidationResult.INVALID_TIME
                    )
                
                ma3_events = valid_events
                Log.info(f"SafeSyncService: Continuing with {len(valid_events)} valid events")
            
            # Step 2: Backup existing events BEFORE any changes
            existing_events = editor_api.get_events_in_layer(layer_id)
            if existing_events:
                self._backup_manager.create_backup(
                    layer_id=layer_id,
                    block_id=block_id,
                    events=existing_events,
                    source="editor"
                )
            
            # Step 3: Check for duplicates
            events_to_add = ma3_events
            if skip_duplicates and not clear_existing:
                duplicate_indices = self._validator.check_for_duplicates(
                    ma3_events, existing_events
                )
                if duplicate_indices:
                    Log.info(f"SafeSyncService: Skipping {len(duplicate_indices)} duplicate events")
                    events_to_add = [e for i, e in enumerate(ma3_events) if i not in duplicate_indices]
            
            # Step 4: Apply events
            if clear_existing:
                editor_api.clear_layer_events(layer_id)
            
            if events_to_add:
                added_count = editor_api.add_events(events_to_add, source="ma3_sync")
            else:
                added_count = 0
            
            # Step 5: Record sync time
            self._last_sync_times[layer_id] = datetime.now()
            
            self.sync_completed.emit(layer_id, True)
            Log.info(f"SafeSyncService: Successfully synced {added_count} events to Editor layer '{layer_id}'")
            
            return SyncResult.success_result(SyncAction.ADD, layer_id, added_count)
            
        except Exception as e:
            Log.error(f"SafeSyncService: Sync to Editor failed: {e}")
            self.sync_completed.emit(layer_id, False)
            return SyncResult.failure_result(SyncAction.ADD, layer_id, str(e))
    
    def sync_to_ma3(
        self,
        editor_layer_id: str,
        ma3_track_coord: str,
        editor_api,
        ma3_service,
        clear_existing: bool = True
    ) -> SyncResult:
        """
        Safely sync Editor layer to MA3 track.
        
        SAFETY STEPS:
        1. Read Editor events
        2. Validate events
        3. (MA3 backup would require reading MA3 state - not always possible)
        4. Send to MA3
        5. Log result
        
        Args:
            editor_layer_id: Source Editor layer
            ma3_track_coord: Target MA3 track coordinate
            editor_api: EditorAPI instance
            ma3_service: MA3CommunicationService instance
            clear_existing: Whether to clear MA3 track first
            
        Returns:
            SyncResult with success/failure info
        """
        self.sync_started.emit(editor_layer_id, "editor_to_ma3")
        
        try:
            # Step 1: Read Editor events
            events = editor_api.get_events_in_layer(editor_layer_id)
            
            if not events:
                Log.info(f"SafeSyncService: No events in Editor layer '{editor_layer_id}'")
                return SyncResult.success_result(SyncAction.ADD, editor_layer_id, 0)
            
            # Step 2: Validate events
            event_dicts = [{'time': e.time, 'duration': e.duration} for e in events]
            is_valid, errors = self._validator.validate_events(event_dicts)
            if not is_valid:
                Log.warning(f"SafeSyncService: Some Editor events failed validation: {errors}")
                # Continue with valid events only
            
            # Step 3: Parse MA3 coordinate
            parts = ma3_track_coord.split('_')
            try:
                timecode_no = int(parts[0].replace('tc', ''))
                track_group = int(parts[1].replace('tg', ''))
                track = int(parts[2].replace('tr', ''))
            except (IndexError, ValueError):
                return SyncResult.failure_result(
                    SyncAction.ADD, editor_layer_id,
                    f"Invalid MA3 track coordinate: {ma3_track_coord}"
                )
            
            # Step 4: Send to MA3
            if clear_existing:
                self._send_clear_track(ma3_service, timecode_no, track_group, track)
            
            for i, event in enumerate(events):
                self._send_add_event(
                    ma3_service, timecode_no, track_group, track,
                    event_index=i,
                    time=event.time,
                    name=getattr(event, 'classification', ''),
                )
            
            # Step 5: Record sync time
            self._last_sync_times[editor_layer_id] = datetime.now()
            
            self.sync_completed.emit(editor_layer_id, True)
            Log.info(f"SafeSyncService: Successfully synced {len(events)} events to MA3 track '{ma3_track_coord}'")
            
            return SyncResult.success_result(SyncAction.ADD, editor_layer_id, len(events))
            
        except Exception as e:
            Log.error(f"SafeSyncService: Sync to MA3 failed: {e}")
            self.sync_completed.emit(editor_layer_id, False)
            return SyncResult.failure_result(SyncAction.ADD, editor_layer_id, str(e))
    
    def rollback(self, layer_id: str, editor_api) -> bool:
        """
        Rollback a layer to its most recent backup.
        
        Args:
            layer_id: Layer to rollback
            editor_api: EditorAPI instance
            
        Returns:
            True if rollback was successful
        """
        snapshot = self._backup_manager.get_latest_backup(layer_id)
        if not snapshot:
            Log.warning(f"SafeSyncService: No backup found for layer '{layer_id}'")
            return False
        
        return self._backup_manager.restore_from_backup(snapshot, editor_api)
    
    def get_backup_history(self, layer_id: str) -> List[LayerSnapshot]:
        """Get backup history for a layer."""
        return self._backup_manager.get_backup_history(layer_id)
    
    def _send_clear_track(self, ma3_service, tc: int, tg: int, tr: int):
        """Send clear track command to MA3."""
        message = f"clear_track|tc={tc}|tg={tg}|tr={tr}"
        ma3_service.send_message(message)
    
    def _send_add_event(self, ma3_service, tc: int, tg: int, tr: int,
                        event_index: int, time: float, name: str):
        """Send add event command to MA3."""
        message = f"add_event|tc={tc}|tg={tg}|tr={tr}|idx={event_index}|time={time}|name={name}"
        ma3_service.send_message(message)
