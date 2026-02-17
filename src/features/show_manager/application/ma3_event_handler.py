"""
MA3 Event Handler

Handles incoming event change notifications from MA3.
Uses EditorAPI to update corresponding Editor layers.

MA3 Message Formats (from MA3 Lua hooks):
- type=event|change=added|tc=101|tg=1|tr=1|idx=5|time=1.5|name=Kick|cmd=Go+
- type=event|change=modified|tc=101|tg=1|tr=1|idx=5|time=2.0|...
- type=event|change=deleted|tc=101|tg=1|tr=1|idx=5
- type=event|change=moved|tc=101|tg=1|tr=1|idx=5|old_time=1.5|new_time=2.0
- type=track|change=hooked|tc=101|tg=1|tr=1|name=Kicks
- type=track|change=unhooked|tc=101|tg=1|tr=1
"""
from typing import TYPE_CHECKING, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum, auto

from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.message import Log
from src.application.settings.show_manager_settings import ShowManagerSettingsManager
from src.features.show_manager.application.sync_system_manager import _ma3_event_defaults

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.features.blocks.application.editor_api import EditorAPI
    from src.features.show_manager.application.sync_system_manager import SyncSystemManager


class MA3ChangeType(Enum):
    """Types of changes from MA3."""
    EVENT_ADDED = auto()
    EVENT_MODIFIED = auto()
    EVENT_DELETED = auto()
    EVENT_MOVED = auto()
    TRACK_HOOKED = auto()
    TRACK_UNHOOKED = auto()
    TRACK_RENAMED = auto()


@dataclass
class MA3EventChange:
    """
    Represents an event change notification from MA3.
    
    Attributes:
        change_type: Type of change
        timecode_no: Timecode number
        track_group: Track group index
        track: Track index
        event_index: Event index within track
        time: Event time (for add/modify)
        old_time: Original time (for move)
        new_time: New time (for move)
        name: Event name
        cmd: MA3 command string
        metadata: Additional data
    """
    change_type: MA3ChangeType
    timecode_no: int
    track_group: int
    track: int
    event_index: int = 0
    time: float = 0.0
    old_time: Optional[float] = None
    new_time: Optional[float] = None
    name: str = ""
    cmd: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def track_coord(self) -> str:
        """Get MA3 track coordinate string."""
        return f"tc{self.timecode_no}_tg{self.track_group}_tr{self.track}"
    
    @classmethod
    def from_message(cls, message) -> Optional['MA3EventChange']:
        """
        Create from MA3Message object.
        
        Args:
            message: MA3Message from MA3CommunicationService
            
        Returns:
            MA3EventChange or None if invalid
        """
        try:
            # Get change_type from message (stored in message.change_type)
            change_str = getattr(message, 'change_type', '') or ''
            
            # Map change string to enum
            change_map = {
                'added': MA3ChangeType.EVENT_ADDED,
                'modified': MA3ChangeType.EVENT_MODIFIED, 
                'updated': MA3ChangeType.EVENT_MODIFIED,  # Alias
                'deleted': MA3ChangeType.EVENT_DELETED,
                'moved': MA3ChangeType.EVENT_MOVED,
                'hooked': MA3ChangeType.TRACK_HOOKED,
                'unhooked': MA3ChangeType.TRACK_UNHOOKED,
                'renamed': MA3ChangeType.TRACK_RENAMED,
                'changed': MA3ChangeType.EVENT_MODIFIED,  # General change
                'list': None,  # Query response, not a change
                'all': None,   # Query response, not a change
            }
            
            change_type = change_map.get(change_str)
            if not change_type:
                return None
            
            # Data is in message.data dict
            data = getattr(message, 'data', {}) or {}
            
            return cls(
                change_type=change_type,
                timecode_no=int(data.get('tc', 0)),
                track_group=int(data.get('tg', 0)),
                track=int(data.get('track', data.get('tr', 0))),  # Support both 'track' and 'tr'
                event_index=int(data.get('idx', 0)),
                time=float(data.get('time', 0.0)),
                old_time=float(data['old_time']) if 'old_time' in data else None,
                new_time=float(data['new_time']) if 'new_time' in data else None,
                name=str(data.get('name', '')),
                cmd=str(data.get('cmd', '')),
                metadata=data
            )
        except (ValueError, KeyError, AttributeError) as e:
            Log.warning(f"MA3EventChange: Failed to parse message: {e}")
            return None


class MA3EventHandler(QObject):
    """
    Handles incoming event change notifications from MA3.
    
    Routes events to SyncSystemManager for bidirectional synchronization.
    
    Signals:
        event_received: Emitted when any MA3 event change is received
        sync_required: Emitted when a change requires sync to Editor
    """
    
    # Signals
    event_received = pyqtSignal(object)  # MA3EventChange
    sync_required = pyqtSignal(str, object)  # track_coord, MA3EventChange
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        show_manager_block_id: str,
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self._facade = facade
        self._show_manager_block_id = show_manager_block_id
        self._settings_manager = ShowManagerSettingsManager(facade, show_manager_block_id)
        
        # SyncSystemManager reference (set via set_sync_manager)
        self._sync_manager: Optional["SyncSystemManager"] = None
        
        # EditorAPI cache by block_id
        self._editor_apis: Dict[str, "EditorAPI"] = {}
        
        # Track which tracks are being monitored
        self._hooked_tracks: Dict[str, Dict[str, Any]] = {}  # coord -> track info
        
        # Cache for received events (from GetEvents/GetAllEvents responses)
        self._cached_events: Dict[str, Dict[str, Any]] = {}  # coord -> {events, timestamp}
        # Suppress MA3->Editor adds during Editor->MA3 pushes (coord -> remaining count)
        self._suppress_events: Dict[str, int] = {}
        
        Log.info(f"MA3EventHandler: Initialized for ShowManager {show_manager_block_id}")

    def _apply_updates_enabled(self, track_coord: str) -> bool:
        """Check if MA3 updates should be applied to Editor for a track."""
        try:
            if self._sync_manager:
                entity = self._sync_manager.get_synced_layer_by_ma3_coord(track_coord)
                if not entity and "." in track_coord:
                    # Normalize coord format: "101.1.12" -> "tc101_tg1_tr12"
                    parts = track_coord.split(".")
                    if len(parts) == 3:
                        normalized = f"tc{parts[0]}_tg{parts[1]}_tr{parts[2]}"
                        entity = self._sync_manager.get_synced_layer_by_ma3_coord(normalized)
                if entity and entity.settings:
                    return bool(getattr(entity.settings, "apply_updates_enabled", True))
        except Exception:
            pass
        return True

    def set_event_suppression(self, track_coord: str, count: int) -> None:
        """Suppress next N MA3 event adds for a coord."""
        if count <= 0:
            return
        self._suppress_events[track_coord] = int(count)
    
    def set_sync_manager(self, sync_manager: "SyncSystemManager") -> None:
        """Set the SyncSystemManager reference."""
        self._sync_manager = sync_manager
        Log.debug(f"MA3EventHandler: Connected to SyncSystemManager")
    
    # Legacy methods for backward compatibility (no-op)
    def set_controller(self, controller: Any) -> None:
        """Deprecated: Use set_sync_manager instead."""
        pass
    
    def set_sync_engine(self, sync_engine: Any) -> None:
        """Deprecated: Use set_sync_manager instead."""
        pass
    
    def register_editor_api(self, block_id: str, editor_api: "EditorAPI") -> None:
        """Register an EditorAPI for a block."""
        self._editor_apis[block_id] = editor_api
    
    def get_editor_api(self, block_id: str) -> Optional["EditorAPI"]:
        """Get EditorAPI for a block."""
        if block_id in self._editor_apis:
            return self._editor_apis[block_id]
        
        # Create new EditorAPI if not cached
        from src.features.blocks.application.editor_api import create_editor_api
        api = create_editor_api(self._facade, block_id)
        self._editor_apis[block_id] = api
        return api
    
    # =========================================================================
    # Message Handlers (called by MA3CommunicationService)
    # =========================================================================
    
    def handle_event_message(self, message) -> None:
        """
        Handle incoming event message from MA3.
        
        Args:
            message: MA3Message from communication service
        """
        change = MA3EventChange.from_message(message)
        if not change:
            Log.debug(f"MA3EventHandler: Ignoring non-change event message: {getattr(message, 'change_type', 'unknown')}")
            return
        
        self.event_received.emit(change)
        
        # Route to appropriate handler
        if change.change_type == MA3ChangeType.EVENT_ADDED:
            self._handle_event_added(change)
        elif change.change_type == MA3ChangeType.EVENT_MODIFIED:
            self._handle_event_modified(change)
        elif change.change_type == MA3ChangeType.EVENT_DELETED:
            self._handle_event_deleted(change)
        elif change.change_type == MA3ChangeType.EVENT_MOVED:
            self._handle_event_moved(change)
    
    def handle_track_message(self, message) -> None:
        """
        Handle incoming track message from MA3.
        
        Args:
            message: MA3Message from communication service
        """
        change = MA3EventChange.from_message(message)
        if not change:
            Log.debug(f"MA3EventHandler: Ignoring non-change track message: {getattr(message, 'change_type', 'unknown')}")
            return
        
        if change.change_type == MA3ChangeType.TRACK_HOOKED:
            self._handle_track_hooked(change)
        elif change.change_type == MA3ChangeType.TRACK_UNHOOKED:
            self._handle_track_unhooked(change)
        elif change.change_type == MA3ChangeType.TRACK_RENAMED:
            self._handle_track_renamed(change)
    
    def handle_events_list(self, message) -> None:
        """
        Handle incoming events list from MA3 (response to GetEvents or GetAllEvents).
        
        Args:
            message: MA3Message from communication service
        """
        data = getattr(message, 'data', {}) or {}
        events = data.get('events', [])
        tc = int(data.get('tc', 0))
        tg = int(data.get('tg', 0))
        track = int(data.get('track', 0))
        
        if not events:
            Log.info(f"MA3EventHandler: Received empty events list for {tc}.{tg}.{track}")
        else:
            Log.info(f"MA3EventHandler: Received {len(events)} events for TC {tc}")
        
        # Store events for sync operations
        coord = f"tc{tc}_tg{tg}_tr{track}"
        self._cached_events[coord] = {
            'tc': tc,
            'tg': tg,
            'track': track,
            'events': events,
            'timestamp': data.get('timestamp', 0)
        }
        
        # Route to SyncSystemManager
        if self._sync_manager:
            self._sync_manager.on_track_events_received(coord, events)
    
    def handle_track_changed(self, message) -> None:
        """
        Handle track change notification (from hooked CmdSubTrack).
        
        This is called when a hooked track's events change. The Lua plugin
        performs change detection and includes:
        - events: Full current event list
        - changes: Summary with added_count, deleted_count, moved_count
        - added: List of added events with fingerprints
        - deleted: List of deleted events with fingerprints
        - moved: List of moved events with old/new positions
        
        Args:
            message: MA3Message from communication service
        """
        data = getattr(message, 'data', {}) or {}
        tc = int(data.get('tc', 0))
        tg = int(data.get('tg', 0))
        track = int(data.get('track', 0))
        track_name = str(data.get('name', '') or '')
        track_note = str(data.get('note', '') or '')
        events = data.get('events', [])
        
        coord = f"{tc}.{tg}.{track}"
        
        # Parse change detection results from Lua plugin
        changes = data.get('changes', {})
        added_count = changes.get('added_count', 0)
        deleted_count = changes.get('deleted_count', 0)
        moved_count = changes.get('moved_count', 0)
        
        Log.info(
            f"MA3EventHandler: Track {coord} changed - "
            f"{added_count} added, {deleted_count} deleted, {moved_count} moved, "
            f"{len(events)} total events"
        )
        
        # Remap track index if MA3 shifted indices (e.g., track deletion)
        if self._sync_manager and track_name:
            self._sync_manager.remap_ma3_track_by_name(tc, tg, track, track_name, track_note=track_note)
        
        # Normalize coord for SyncSystemManager
        normalized_coord = f"tc{tc}_tg{tg}_tr{track}"
        
        # Update cached events
        self._cached_events[normalized_coord] = {
            'tc': tc,
            'tg': tg,
            'track': track,
            'events': events,
            'timestamp': data.get('timestamp', 0)
        }
        
        # Route to SyncSystemManager
        if self._sync_manager:
            if not self._apply_updates_enabled(normalized_coord):
                Log.info(f"MA3EventHandler: Updates paused - skipping apply for {normalized_coord}")
                return
            # SyncSystemManager handles divergence detection and sync
            self._sync_manager.on_track_events_received(normalized_coord, events)
        
        # Signal that sync may be required (for legacy handlers)
        self.sync_required.emit(coord, message)
    
    def get_cached_events(self, tc: int, tg: int, track: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached events for a track (from last fetch or hook notification)."""
        coord = f"{tc}.{tg}.{track}"
        cached = self._cached_events.get(coord)
        if not cached:
            return None
        return cached.get('events', [])
    
    # =========================================================================
    # Event Change Handlers
    # =========================================================================
    
    def _handle_event_added(self, change: MA3EventChange) -> None:
        """Handle event added in MA3."""
        Log.info(f"MA3EventHandler: Event added in MA3 track {change.track_coord} at {change.time}s")
        suppress_remaining = self._suppress_events.get(change.track_coord)
        if suppress_remaining:
            new_remaining = suppress_remaining - 1
            if new_remaining > 0:
                self._suppress_events[change.track_coord] = new_remaining
            else:
                self._suppress_events.pop(change.track_coord, None)
            return
        if not self._apply_updates_enabled(change.track_coord):
            Log.info(f"MA3EventHandler: Updates paused - ignoring add for {change.track_coord}")
            return
        
        # Find synced Editor layer for this track
        editor_info = self._find_synced_editor_layer(change.track_coord)
        if not editor_info:
            Log.debug(f"MA3EventHandler: Track {change.track_coord} not synced to any Editor layer")
            return
        
        editor_block_id, layer_name = editor_info
        
        # Use EditorAPI to add the event
        editor_api = self.get_editor_api(editor_block_id)
        if not editor_api:
            Log.warning(f"MA3EventHandler: Could not get EditorAPI for block {editor_block_id}")
            return
        
        # Add single event with normalized MA3 defaults
        live_metadata = {
            'source': 'ma3_live',
            'ma3_coord': change.track_coord,
            'ma3_event_index': change.event_index,
            'ma3_name': change.name,
            'ma3_cmd': change.cmd,
        }
        norm_duration, live_metadata = _ma3_event_defaults(0.0, live_metadata)
        editor_api.add_event(
            time=change.time,
            classification=layer_name,
            duration=norm_duration,
            metadata=live_metadata,
            source="ma3_live"
        )
        
        Log.info(f"MA3EventHandler: Added event to Editor layer '{layer_name}' at {change.time}s")
    
    def _handle_event_modified(self, change: MA3EventChange) -> None:
        """Handle event modified in MA3."""
        Log.info(f"MA3EventHandler: Event modified in MA3 track {change.track_coord}")
        if not self._apply_updates_enabled(change.track_coord):
            Log.info(f"MA3EventHandler: Updates paused - ignoring modify for {change.track_coord}")
            return
        
        editor_info = self._find_synced_editor_layer(change.track_coord)
        if not editor_info:
            return
        
        editor_block_id, layer_name = editor_info
        editor_api = self.get_editor_api(editor_block_id)
        if not editor_api:
            return
        
        # Find matching event by MA3 metadata
        events = editor_api.get_events_in_layer(layer_name)
        for event in events:
            if event.metadata.get('ma3_event_index') == change.event_index:
                # Get data item IDs to find which item contains this event
                data_item_ids = editor_api.get_data_item_ids()
                if data_item_ids:
                    # Update event (time, name, etc.)
                    editor_api.update_event(
                        event_id=event.id,
                        data_item_id=data_item_ids[0],  # Use first data item
                        time=change.time,
                        metadata={
                            **event.metadata,
                            'ma3_name': change.name,
                            'ma3_cmd': change.cmd,
                        }
                    )
                    Log.info(f"MA3EventHandler: Updated event in layer '{layer_name}'")
                break
    
    def _handle_event_deleted(self, change: MA3EventChange) -> None:
        """Handle event deleted in MA3."""
        Log.info(f"MA3EventHandler: Event deleted in MA3 track {change.track_coord}")
        if not self._apply_updates_enabled(change.track_coord):
            Log.info(f"MA3EventHandler: Updates paused - ignoring delete for {change.track_coord}")
            return
        
        editor_info = self._find_synced_editor_layer(change.track_coord)
        if not editor_info:
            return
        
        editor_block_id, layer_name = editor_info
        editor_api = self.get_editor_api(editor_block_id)
        if not editor_api:
            return
        
        # Find and delete matching event
        events = editor_api.get_events_in_layer(layer_name)
        for event in events:
            if event.metadata.get('ma3_event_index') == change.event_index:
                data_item_ids = editor_api.get_data_item_ids()
                if data_item_ids:
                    editor_api.delete_event(
                        event_id=event.id,
                        data_item_id=data_item_ids[0],
                        layer_name=layer_name
                    )
                    Log.info(f"MA3EventHandler: Deleted event from layer '{layer_name}'")
                break
    
    def _handle_event_moved(self, change: MA3EventChange) -> None:
        """Handle event moved in MA3."""
        Log.info(f"MA3EventHandler: Event moved in MA3 track {change.track_coord}: {change.old_time} -> {change.new_time}")
        if not self._apply_updates_enabled(change.track_coord):
            Log.info(f"MA3EventHandler: Updates paused - ignoring move for {change.track_coord}")
            return
        
        editor_info = self._find_synced_editor_layer(change.track_coord)
        if not editor_info:
            return
        
        editor_block_id, layer_name = editor_info
        editor_api = self.get_editor_api(editor_block_id)
        if not editor_api:
            return
        
        # Find and move matching event
        events = editor_api.get_events_in_layer(layer_name)
        for event in events:
            if event.metadata.get('ma3_event_index') == change.event_index:
                data_item_ids = editor_api.get_data_item_ids()
                if data_item_ids and change.new_time is not None:
                    editor_api.move_event(
                        event_id=event.id,
                        data_item_id=data_item_ids[0],
                        new_time=change.new_time,
                        layer_name=layer_name
                    )
                    Log.info(f"MA3EventHandler: Moved event in layer '{layer_name}' to {change.new_time}s")
                break
    
    # =========================================================================
    # Track Change Handlers
    # =========================================================================
    
    def _handle_track_hooked(self, change: MA3EventChange) -> None:
        """Handle track hooked (subscribed) in MA3."""
        Log.info(f"MA3EventHandler: Track hooked: {change.track_coord} ({change.name})")
        
        self._hooked_tracks[change.track_coord] = {
            'name': change.name,
            'timecode_no': change.timecode_no,
            'track_group': change.track_group,
            'track': change.track,
        }
    
    def _handle_track_unhooked(self, change: MA3EventChange) -> None:
        """Handle track unhooked (unsubscribed) in MA3."""
        Log.info(f"MA3EventHandler: Track unhooked: {change.track_coord}")
        
        if change.track_coord in self._hooked_tracks:
            del self._hooked_tracks[change.track_coord]
    
    def _handle_track_renamed(self, change: MA3EventChange) -> None:
        """Handle track renamed in MA3."""
        Log.info(f"MA3EventHandler: Track renamed: {change.track_coord} -> {change.name}")
        
        if change.track_coord in self._hooked_tracks:
            self._hooked_tracks[change.track_coord]['name'] = change.name
        
        # Update corresponding Editor layer if synced
        editor_info = self._find_synced_editor_layer(change.track_coord)
        if editor_info:
            editor_block_id, layer_name = editor_info
            editor_api = self.get_editor_api(editor_block_id)
            if editor_api:
                # Optionally rename the layer to match
                # For now, just update metadata
                Log.debug(f"MA3EventHandler: Track {change.track_coord} renamed, Editor layer '{layer_name}' unchanged")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _find_synced_editor_layer(self, ma3_track_coord: str) -> Optional[tuple]:
        """
        Find the Editor layer synced to an MA3 track.
        
        Args:
            ma3_track_coord: MA3 track coordinate
            
        Returns:
            Tuple of (editor_block_id, layer_name) or None
        """
        if not self._sync_manager:
            return None
        
        # Find synced entity by MA3 coord
        entity = self._sync_manager.get_synced_layer_by_ma3_coord(ma3_track_coord)
        if entity and entity.editor_layer_id and entity.editor_block_id:
            return (entity.editor_block_id, entity.editor_layer_id)
        
        return None
    
    def get_hooked_tracks(self) -> Dict[str, Dict[str, Any]]:
        """Get currently hooked MA3 tracks."""
        return self._hooked_tracks.copy()
    
    def cleanup(self) -> None:
        """Clean up handler resources."""
        self._hooked_tracks.clear()
        self._cached_events.clear()
        self._editor_apis.clear()
        self._sync_manager = None
        Log.debug("MA3EventHandler: Cleaned up")
