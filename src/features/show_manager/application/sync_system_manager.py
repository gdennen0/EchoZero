"""
Sync System Manager

Single orchestration point for all sync operations between EchoZero and MA3.
Replaces LayerSyncController and merges SyncEngine functionality.

Architecture:
- Lua plugin sends raw data (no computation)
- This manager stores all MA3 state locally
- This manager does all diff computation
- UI displays only, no data management
"""
from typing import Optional, Dict, Any, List, Callable, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from PyQt6.QtCore import QObject, pyqtSignal

from src.features.show_manager.domain.sync_layer_entity import (
    SyncLayerEntity,
    SyncLayerSettings,
    SyncSource,
    SyncStatus,
    SyncDirection,
    ConflictStrategy,
)
from src.features.show_manager.application.sync_layer_manager import (
    SyncLayerManager,
    SyncLayerComparison,
)
from src.features.show_manager.application.ma3_track_resolver import normalize_track_name
from src.utils.message import Log
from src.utils.paths import get_debug_log_path

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.application.settings.show_manager_settings import ShowManagerSettingsManager
    from src.features.ma3.application.ma3_communication_service import MA3CommunicationService
    from src.features.blocks.application.editor_api import EditorAPI

# =============================================================================
# MA3 Event Defaults
# =============================================================================

MA3_MIN_EVENT_DURATION = 0.3  # Minimum duration for MA3 events (seconds)

def _ma3_event_defaults(
    duration: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Normalize MA3 event defaults at creation time.
    
    Enforces minimum duration and sets render_as_marker=True as default
    display mode (user can override via context menu; the override persists
    in the database and is respected on reload).
    
    Args:
        duration: Raw duration from MA3 (typically 0.0)
        metadata: Event metadata dict (modified in place if provided)
        
    Returns:
        Tuple of (normalized_duration, metadata_with_defaults)
    """
    duration = max(duration, MA3_MIN_EVENT_DURATION)
    metadata = metadata or {}
    metadata.setdefault('render_as_marker', True)
    return duration, metadata

# =============================================================================
# MA3 State Data Classes
# =============================================================================

@dataclass
class MA3TrackGroupInfo:
    """Information about an MA3 track group."""
    timecode_no: int
    track_group_no: int
    name: str
    track_count: int = 0

@dataclass
class MA3TrackInfo:
    """Information about an MA3 track."""
    timecode_no: int
    track_group_no: int
    track_no: int
    name: str
    coord: str = ""  # Computed: tc{n}_tg{n}_tr{n}
    event_count: int = 0
    sequence_no: Optional[int] = None  # Sequence number assigned to track (if any)
    note: str = ""  # EZ identity stored in MA3 .note property (e.g. "ez:Drums")
    
    def __post_init__(self):
        if not self.coord:
            self.coord = f"tc{self.timecode_no}_tg{self.track_group_no}_tr{self.track_no}"

@dataclass
class MA3EventInfo:
    """Information about an MA3 event."""
    time: float
    name: str
    cmd: str = ""
    idx: int = 0
    tc: int = 0
    tg: int = 0
    track: int = 0
    
    def fingerprint(self) -> str:
        """Generate fingerprint for event matching."""
        return f"{self.time:.6f}||{self.name}"
    
    def content_fingerprint(self) -> str:
        """Generate content-only fingerprint for move detection."""
        return f"{self.cmd}|{self.name}"

class SyncSystemManager(QObject):
    """
    Single orchestration point for all sync operations.
    
    This manager:
    - Maintains a list of SyncLayerEntity instances
    - Provides methods for sync/unsync/resync operations
    - Handles reconnection and divergence checking
    - Emits signals for UI updates
    - Uses existing commands for operations
    
    Design principle: The UI should be thin - just call methods on this manager
    and update based on signals.
    """
    
    # Signals for UI
    entities_changed = pyqtSignal()  # Emitted when entity list changes
    entity_updated = pyqtSignal(str)  # Emitted when specific entity updated (entity_id)
    sync_status_changed = pyqtSignal(str, str)  # entity_id, new_status
    divergence_detected = pyqtSignal(str, object)  # entity_id, SyncLayerComparison
    error_occurred = pyqtSignal(str, str)  # entity_id, error_message
    ma3_connection_changed = pyqtSignal(bool)  # is_connected (legacy, kept for compat)
    connection_state_changed = pyqtSignal(str)  # "connected", "disconnected", "stale"
    # Emitted when track already exists and user must choose: overwrite, merge, or cancel
    # Args: (editor_layer_id, track_name, editor_event_count, ma3_event_count)
    track_conflict_prompt = pyqtSignal(str, str, int, int)
    
    # Emitted when a similar MA3 track exists and user must choose: use_existing, create_new, or cancel
    # Args: (editor_layer_id, editor_name, existing_ma3_track_name, ma3_event_count, tc, tg, seq)
    existing_track_prompt = pyqtSignal(str, str, str, int, int, int, int)
    
    # Emitted when divergence is detected and user must choose which events to keep
    # Args: (editor_layer_id, editor_name, ma3_track_name, editor_event_count, ma3_event_count, tc, tg, seq, track_no)
    reconciliation_prompt = pyqtSignal(str, str, str, int, int, int, int, int, int)
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        show_manager_block_id: str,
        settings_manager: "ShowManagerSettingsManager",
        parent: Optional[QObject] = None,
    ):
        """
        Initialize the sync system manager.
        
        Args:
            facade: Application facade for accessing services
            show_manager_block_id: The ShowManager block ID
            settings_manager: Settings manager for persistence
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self._facade = facade
        self._show_manager_block_id = show_manager_block_id
        self._settings_manager = settings_manager
        
        # In-memory storage of sync layers
        self._synced_layers: Dict[str, SyncLayerEntity] = {}
        
        # Track MA3 connection state
        self._ma3_connected = False
        self._connection_state: str = "disconnected"  # "connected", "disconnected", "stale"
        
        # Ping-based connection monitoring (moved from panel)
        self._ping_timer: Optional["QTimer"] = None  # Created in start_connection_monitoring()
        self._missed_pings: int = 0
        self._awaiting_ping_response: bool = False
        
        # Hooked MA3 tracks (coord -> callback)
        self._hooked_tracks: Dict[str, Callable] = {}
        self._hooked_track_groups: Dict[str, bool] = {}
        
        # Sync guards to prevent recursive/concurrent updates
        self._syncing_from_ma3: Dict[str, bool] = {}  # coord -> is_syncing
        self._last_ma3_push_time: Dict[str, float] = {}  # coord -> timestamp
        self._last_editor_push_time: Dict[str, float] = {}  # coord -> timestamp
        self._ma3_empty_ignore_window_s: float = 0.5  # seconds
        self._force_apply_to_ez: Dict[str, bool] = {}  # coord -> force apply flag
        self._ma3_apply_cooldown_until: Dict[str, float] = {}  # coord -> timestamp
        self._ma3_apply_cooldown_s: float = 0.5
        
        # MA3 events request session state
        self._ma3_events_in_flight: Dict[str, int] = {}  # coord -> request_id
        self._ma3_events_request_timeout_s: float = 1.5
        
        # Multi-track sync diagnostics: track.changed arrival timestamps
        self._diag_track_changed_ts: Dict[str, float] = {}  # coord -> timestamp
        self._diag_first_track_changed_ts: float = 0.0  # earliest track.changed in current batch
        self._events_request_retries: Dict[str, int] = {}  # coord -> retry count
        
        # Skip MA3->Editor sync flag (set during Editor->MA3 push to prevent feedback loop)
        self._skip_initial_hook: Dict[str, bool] = {}  # coord -> should_skip
        
        # =================================================================
        # MA3 State Storage (single source of truth for MA3 data)
        # =================================================================
        # What we think exists in MA3 - populated by fetch operations
        
        # Track groups by timecode: {timecode_no: [MA3TrackGroupInfo, ...]}
        self._ma3_track_groups: Dict[int, List[MA3TrackGroupInfo]] = {}
        
        # Tracks by coord: {coord: MA3TrackInfo}
        self._ma3_tracks: Dict[str, MA3TrackInfo] = {}
        self._ma3_tracks_version: int = 0
        
        # Events by track coord: {coord: [MA3EventInfo, ...]}
        self._ma3_track_events: Dict[str, List[MA3EventInfo]] = {}
        
        # Pending sequence existence checks: {seq_no: bool or None}
        # None = check pending, True = exists, False = doesn't exist
        self._pending_sequence_checks: Dict[int, Optional[bool]] = {}
        
        # Configured timecode number for this ShowManager
        # Will be loaded from settings in _load_from_settings()
        self._configured_timecode: int = 1
        
        # Cached editor references (avoid repeated connection traversal)
        self._cached_editor_block_id: Optional[str] = None
        self._cached_editor_api: Optional[Any] = None
        self._cached_editor_api_block_id: Optional[str] = None  # block_id the API was created for
        
        # Load synced layers and settings
        self._load_from_settings()
        
        # Subscribe to Editor changes via EventBus
        self._subscribe_to_editor_changes()
        
        # Register handler for sequence.exists messages
        self._register_ma3_handlers()
        
        # Push sync state to TimelineLayer objects whenever entities change.
        # This is the single path for updating the Editor timeline's sync icons.
        # Uses QTimer.singleShot(0) so the push runs AFTER any pending events
        # (e.g., BlockUpdated -> Editor reload that creates fresh layer objects).
        self.entities_changed.connect(self._schedule_sync_state_push)
        self.sync_status_changed.connect(lambda *_: self._schedule_sync_state_push())
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def ma3_comm_service(self) -> Optional["MA3CommunicationService"]:
        """Get MA3 communication service."""
        # Try direct attribute first
        comm = getattr(self._facade, "ma3_comm_service", None)
        if comm:
            return comm
        # Try via show_manager_listener_service
        listener_service = getattr(self._facade, "show_manager_listener_service", None)
        if listener_service:
            return getattr(listener_service, "_ma3_comm", None)
        return None
    
    def _get_ma3_target(self) -> Tuple[str, int]:
        """
        Get current MA3 target IP and port from settings.
        
        Returns:
            Tuple of (ip, port) from ShowManager settings
        """
        if self._settings_manager:
            ip = self._settings_manager.ma3_ip or "127.0.0.1"
            port = self._settings_manager.ma3_port or 9001
            return (ip, port)
        return ("127.0.0.1", 9001)
    
    def _send_lua_command_with_target(self, lua_code: str) -> bool:
        """
        Send Lua command to MA3 using current IP/port from settings.
        
        This ensures we always use the current settings, even if service
        defaults haven't been updated yet.
        
        Args:
            lua_code: Lua command to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        comm = self.ma3_comm_service
        if not comm:
            Log.error("SyncSystemManager: MA3 communication service not available")
            return False
        
        ip, port = self._get_ma3_target()
        
        # Always pass explicit IP/port to ensure we use current settings
        success = comm.send_lua_command(lua_code, target_ip=ip, target_port=port)
        
        if not success:
            Log.warning(f"SyncSystemManager: Failed to send Lua command to {ip}:{port}: {lua_code[:50]}")
        
        return success
    
    # =========================================================================
    # Connection Monitoring (centralized ping-based detection)
    # =========================================================================
    
    @property
    def connection_state(self) -> str:
        """Current MA3 connection state: 'connected', 'disconnected', or 'stale'."""
        return self._connection_state
    
    def start_connection_monitoring(self) -> None:
        """Start ping-based connection monitoring.
        
        Called when the OSC listener starts. Sends pings every 3s and detects
        disconnect by counting consecutive missed pings.
        
        Connection is NOT set to "connected" optimistically. Instead, we wait
        for the first actual message from MA3 (via on_ma3_message_received)
        before triggering the connected flow. This ensures structure fetching
        and auto-reconnect only happen when MA3 is confirmed alive.
        
        Also subscribes to MA3OscInbound events on the facade event bus so the
        SSM receives incoming messages even when the ShowManagerPanel is not open.
        """
        from PyQt6.QtCore import QTimer
        
        
        # Subscribe to raw OSC inbound events so the SSM can track connection
        # health independently of the ShowManagerPanel being open
        self._subscribe_to_osc_inbound()
        
        if self._ping_timer is None:
            self._ping_timer = QTimer()
            self._ping_timer.timeout.connect(self._on_ping_tick)
        
        self._missed_pings = 0
        self._awaiting_ping_response = False
        self._ping_timer.start(3000)
        
        # Send an immediate ping to get a fast first response
        QTimer.singleShot(200, self._on_ping_tick)
        
        Log.info("SyncSystemManager: Connection monitoring started (3s ping interval, awaiting first response)")
    
    def stop_connection_monitoring(self) -> None:
        """Stop ping-based connection monitoring.
        
        Called when the OSC listener stops.
        """
        if self._ping_timer is not None:
            self._ping_timer.stop()
        
        self._missed_pings = 0
        self._awaiting_ping_response = False
        
        # Unsubscribe from OSC inbound events
        self._unsubscribe_from_osc_inbound()
        
        self.on_ma3_disconnected()
        self._set_connection_state("disconnected")
        
        Log.info("SyncSystemManager: Connection monitoring stopped")
    
    # =========================================================================
    # OSC Event Bus Subscription (independent of panel)
    # =========================================================================
    
    def _subscribe_to_osc_inbound(self) -> None:
        """Subscribe to MA3OscInbound events on the facade event bus.
        
        This allows the SSM to track connection health even when the
        ShowManagerPanel is not open.
        """
        if not hasattr(self, '_osc_subscribed') or not self._osc_subscribed:
            event_bus = getattr(self._facade, 'event_bus', None)
            if event_bus:
                try:
                    from src.application.events.events import MA3OscInbound
                    event_bus.subscribe(MA3OscInbound, self._on_osc_inbound_event)
                    self._osc_subscribed = True
                    Log.info("SyncSystemManager: Subscribed to MA3OscInbound on event bus")
                except Exception as e:
                    Log.warning(f"SyncSystemManager: Failed to subscribe to MA3OscInbound: {e}")
    
    def _unsubscribe_from_osc_inbound(self) -> None:
        """Unsubscribe from MA3OscInbound events."""
        if getattr(self, '_osc_subscribed', False):
            event_bus = getattr(self._facade, 'event_bus', None)
            if event_bus:
                try:
                    from src.application.events.events import MA3OscInbound
                    event_bus.unsubscribe(MA3OscInbound, self._on_osc_inbound_event)
                except Exception:
                    pass
            self._osc_subscribed = False
    
    def _on_osc_inbound_event(self, event) -> None:
        """Handle raw OSC inbound event from the event bus.
        
        Any OSC message arriving from MA3 means the connection is alive.
        Also parses EZ plugin messages (trackgroups, tracks, events) so the
        SSM can function even when the ShowManagerPanel is not open.
        """
        self.on_ma3_message_received()
        
        # Parse EZ plugin messages for structure data
        data = getattr(event, "data", {}) or {}
        address = data.get("address")
        if address == "/ez/message":
            args = data.get("args") or []
            if args:
                self._handle_ez_message_internal(args[0] if isinstance(args, list) else str(args))
    
    def _handle_ez_message_internal(self, message_str: str) -> None:
        """Parse and route EZ plugin messages that the SSM needs.
        
        Handles: trackgroups.list, tracks.list, events.list, events.all
        This allows the SSM to process MA3 responses even without the panel.
        """
        if not message_str:
            return
        try:
            from src.features.ma3.infrastructure.osc_parser import get_osc_parser
            parser = get_osc_parser()
            message = parser.parse_message(message_str)
            if not message:
                return
            
            type_key = f"{message.message_type.value}.{message.change_type.value}" if message.message_type and message.change_type else ""
            
            
            if type_key == "trackgroups.list":
                tc = message.get('tc', 0)
                trackgroups = parser.parse_trackgroups(message)
                groups_data = [
                    {'no': tg.no, 'name': tg.name, 'track_count': tg.track_count}
                    for tg in trackgroups
                ]
                self.on_track_groups_received(tc, groups_data)
                # Auto-fetch tracks for each group
                for tg in trackgroups:
                    cmd = f"EZ.GetTracks({tc}, {tg.no})"
                    self._send_lua_command_with_target(cmd)
            
            elif type_key == "tracks.list":
                tc = message.get('tc', 0)
                tg = message.get('tg', 0)
                tracks = parser.parse_tracks(message)
                tracks_data = [
                    {'no': track.no, 'name': track.name or f"Track {track.no}", 'event_count': 0}
                    for track in tracks
                ]
                self.on_tracks_received(tc, tg, tracks_data)
            
            elif type_key in ("events.list", "events.all"):
                tc = message.get('tc', 0)
                tg = message.get('tg', 0)
                track = message.get('track', 0)
                coord = f"tc{tc}_tg{tg}_tr{track}"
                events = message.data.get('events', []) if hasattr(message, 'data') else []
                self.on_track_events_received(coord, events)
                
        except Exception as e:
            Log.debug(f"SyncSystemManager: Error parsing EZ message: {e}")
    
    def on_ma3_message_received(self) -> None:
        """Called when ANY message arrives from MA3.
        
        Resets the missed ping counter and, if previously disconnected,
        triggers the full connection flow: state transition, structure fetch,
        and auto-reconnect of synced layers.
        
        This is the ONLY path that transitions to "connected" -- we never
        assume MA3 is alive until we receive an actual message.
        """
        self._awaiting_ping_response = False
        was_disconnected = self._connection_state != "connected"
        
        
        if self._missed_pings > 0:
            self._missed_pings = 0
        
        if was_disconnected:
            Log.info(f"SyncSystemManager: MA3 connection confirmed (was {self._connection_state})")
            self._set_connection_state("connected")
            self.on_ma3_connected()
            # Fetch MA3 structure to populate available layers and trigger auto-reconnect
            # This sends EZ.GetTrackGroups -> response flows to on_tracks_received ->
            # auto_reconnect_layers -> resync of existing layers
            self._fetch_ma3_structure()
    
    def _on_ping_tick(self) -> None:
        """Periodic ping tick (every 3s).
        
        Sends a ping to MA3 and tracks whether the previous ping was answered:
        - 1 miss (3s): stale warning
        - 2 misses (6s): confirmed disconnect
        """
        # Check if the previous ping was answered
        if self._awaiting_ping_response:
            self._missed_pings += 1
            
            if self._missed_pings == 1:
                self._set_connection_state("stale")
            elif self._missed_pings >= 2:
                if self._ma3_connected:
                    Log.info(f"SyncSystemManager: MA3 disconnected (no response to {self._missed_pings} pings)")
                    self.on_ma3_disconnected()
                    self._set_connection_state("disconnected")
                # Keep pinging so we detect reconnection
        
        # Send a new ping using the configured target
        success = self._send_lua_command_with_target("EZ.Ping()")
        if success:
            self._awaiting_ping_response = True
    
    def _set_connection_state(self, new_state: str) -> None:
        """Update connection state and emit signal if changed."""
        if new_state != self._connection_state:
            old_state = self._connection_state
            self._connection_state = new_state
            Log.info(f"SyncSystemManager: Connection state {old_state} -> {new_state}")
            self.connection_state_changed.emit(new_state)
    
    def _fetch_ma3_structure(self) -> None:
        """Fetch MA3 track groups for the current timecode.
        
        Sends the EZ.GetTrackGroups Lua command to MA3. The response will
        flow through the OSC handler chain and eventually call on_tracks_received().
        """
        tc = self.configured_timecode
        if not tc:
            Log.warning("SyncSystemManager: Cannot fetch structure - no timecode configured")
            return
        
        cmd = f"EZ.GetTrackGroups({tc})"
        success = self._send_lua_command_with_target(cmd)
        if success:
            Log.info(f"SyncSystemManager: Fetching MA3 structure for TC{tc}")
        else:
            Log.warning(f"SyncSystemManager: Failed to fetch MA3 structure for TC{tc}")
    
    @property
    def is_ma3_connected(self) -> bool:
        """Check if MA3 is connected."""
        return self._ma3_connected
    
    @property
    def configured_timecode(self) -> int:
        """Get the configured timecode number from settings."""
        # Always return the current value from settings manager if available
        if self._settings_manager:
            return self._settings_manager.target_timecode
        return self._configured_timecode
    
    @configured_timecode.setter
    def configured_timecode(self, value: int) -> None:
        """Set the configured timecode and save to settings."""
        if value != self._configured_timecode:
            self._configured_timecode = value
            if self._settings_manager:
                self._settings_manager.target_timecode = value
            Log.info(f"SyncSystemManager: Updated configured_timecode to {value}")
    
    def _subscribe_to_editor_changes(self) -> None:
        """Subscribe to Editor block change events and project lifecycle events."""
        try:
            from src.application.events.events import BlockUpdated, ProjectLoaded
            
            # Use the facade's event_bus (shared instance) not a new EventBus()
            if hasattr(self._facade, 'event_bus') and self._facade.event_bus:
                self._facade.event_bus.subscribe(BlockUpdated, self._on_editor_block_updated)
                self._facade.event_bus.subscribe(ProjectLoaded, self._on_project_changed)
                # Also subscribe to string-based event names for compatibility
                self._facade.event_bus.subscribe("project.loaded", self._on_project_changed)
                self._facade.event_bus.subscribe("project.created", self._on_project_changed)
                Log.info("SyncSystemManager: Subscribed to BlockUpdated and project events on shared EventBus")
            else:
                Log.warning("SyncSystemManager: No event_bus on facade, cannot subscribe to Editor changes")
        except ImportError as e:
            Log.warning(f"SyncSystemManager: Could not subscribe to events: {e}")
    
    def _register_ma3_handlers(self) -> None:
        """Register handlers for MA3 OSC messages."""
        comm = self.ma3_comm_service
        if not comm:
            return
        
        # Register handler for sequence.exists messages
        comm.register_handler("sequence", "exists", self._on_sequence_exists_response)
        comm.register_handler("trackgroup", "changed", self._on_trackgroup_changed)
        Log.debug("SyncSystemManager: Registered handler for sequence.exists messages")
    
    def _on_sequence_exists_response(self, message) -> None:
        """Handle sequence.exists OSC response from MA3."""
        seq_no = message.data.get("no")
        exists = message.data.get("exists", False)
        
        if seq_no is not None:
            self._pending_sequence_checks[seq_no] = exists
            Log.debug(f"SyncSystemManager: Sequence {seq_no} exists: {exists}")
    
    def _on_editor_block_updated(self, event: "BlockUpdated") -> None:
        """
        Handle Editor block updates.
        
        When an Editor block is updated, check if any synced layers need to
        push their changes to MA3.
        """
        
        if not hasattr(event, 'data') or not event.data:
            return
        
        block_id = event.data.get("id")
        events_updated = event.data.get("events_updated", False)
        update_source = event.data.get("source", "")
        
        Log.debug(f"SyncSystemManager: BlockUpdated - block_id={block_id}, events_updated={events_updated}, source={update_source}")
        
        # Only process if events were updated
        if not events_updated:
            Log.debug(f"SyncSystemManager: Skipping - events_updated=False")
            return
        
        # IMPORTANT: Skip if the update came from MA3 sync (avoid feedback loop)
        if update_source == "ma3_sync":
            Log.debug(f"SyncSystemManager: Skipping Editor→MA3 push - source was ma3_sync")
            return
        
        # Get layer names that changed (if provided)
        changed_layer_names = event.data.get("layer_names", [])
        
        
        # Check if any synced layers use this block
        matching_entities = [e for e in self._synced_layers.values() 
                           if e.editor_block_id == block_id and e.sync_status == SyncStatus.SYNCED]
        
        # If layer_names provided, only push entities for those specific layers
        if changed_layer_names:
            matching_entities = [e for e in matching_entities 
                               if e.editor_layer_id in changed_layer_names]
        else:
            # If layer_names not provided, log a warning (this shouldn't happen for user edits)
            # But still push all matching entities (fallback behavior)
            if update_source == "editor":
                Log.warning(f"SyncSystemManager: BlockUpdated from Editor without layer_names - will push all matching entities for block {block_id}")
        
        
        Log.debug(f"SyncSystemManager: Found {len(matching_entities)} synced entities for block {block_id} (layers: {changed_layer_names})")
        
        
        import time
        for entity in matching_entities:
            if entity.ma3_coord and self._syncing_from_ma3.get(entity.ma3_coord, False):
                Log.debug(f"SyncSystemManager: Skipping Editor→MA3 push for {entity.editor_layer_id} (syncing from MA3)")
                continue
            if entity.ma3_coord:
                cooldown_until = self._ma3_apply_cooldown_until.get(entity.ma3_coord, 0)
                if time.time() < cooldown_until:
                    Log.debug(f"SyncSystemManager: Skipping Editor→MA3 push for {entity.editor_layer_id} (cooldown)")
                    continue
            # Push to MA3 for the specific synced layer that changed
            # Use coalescing to avoid spamming MA3 during drags
            Log.info(f"SyncSystemManager: Scheduling Editor→MA3 push for {entity.editor_layer_id} -> {entity.ma3_coord}")
            self._schedule_push_editor_to_ma3(entity)
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def sync_layer(
        self,
        source: str,
        source_id: str,
        target_id: Optional[str] = None,
        auto_create: bool = False,
        event_authority: Optional[str] = None,
    ) -> Optional[str]:
        """
        Start syncing a layer.
        
        Args:
            source: "ma3" or "editor" - where the layer originates
            source_id: MA3 coord or Editor layer_id
            target_id: Optional target (Editor layer_id or MA3 coord)
            auto_create: If True, auto-create target without prompting
            event_authority: When mapping to existing target, which side's events
                to keep. "editor" pushes Editor->MA3, "ma3" pushes MA3->Editor,
                None uses default behavior (follow source direction).
            
        Returns:
            Entity ID if successful, None otherwise
        """
        Log.info(f"SyncSystemManager: sync_layer source={source} source_id={source_id}")
        
        try:
            if source == "ma3":
                return self._sync_from_ma3(source_id, target_id, auto_create, event_authority)
            elif source == "editor":
                return self._sync_from_editor(source_id, target_id, auto_create, event_authority)
            else:
                Log.error(f"SyncSystemManager: Invalid source: {source}")
                return None
        except Exception as e:
            Log.error(f"SyncSystemManager: sync_layer failed: {e}")
            self.error_occurred.emit(source_id, str(e))
            return None
    
    def unsync_layer(self, entity_id: str) -> bool:
        """
        Stop syncing a layer with asymmetric behavior.
        
        - MA3-sourced: Delete the synced copy from Editor
        - Editor-sourced: Leave the synced copy in MA3
        
        Args:
            entity_id: The sync layer entity ID
            
        Returns:
            True if successful
        """
        Log.info(f"SyncSystemManager: unsync_layer entity_id={entity_id}")
        
        entity = self._synced_layers.get(entity_id)
        if not entity:
            Log.warning(f"SyncSystemManager: Entity not found: {entity_id}")
            return False
        
        try:
            # Unhook MA3 track if hooked
            if entity.ma3_coord and entity.ma3_coord in self._hooked_tracks:
                self._unhook_ma3_track(entity.ma3_coord)
            
            # Capture deletion info before removing entity from storage
            should_delete_editor = (
                entity.source == SyncSource.MA3
                and entity.editor_layer_id
                and entity.editor_block_id
            )
            editor_block_id = entity.editor_block_id
            editor_layer_id = entity.editor_layer_id
            
            # Remove from storage FIRST so that notify_editor_layer_deleted()
            # won't find this entity and re-create a DISCONNECTED entry
            del self._synced_layers[entity_id]
            self._save_to_settings()
            
            # Now apply asymmetric unsync behavior
            if should_delete_editor:
                # MA3-sourced: Delete from Editor (entity already removed from storage)
                self._delete_editor_layer(editor_block_id, editor_layer_id)
            # Editor-sourced: Leave MA3 copy (do nothing)
            
            self.entities_changed.emit()
            return True
            
        except Exception as e:
            Log.error(f"SyncSystemManager: unsync_layer failed: {e}")
            self.error_occurred.emit(entity_id, str(e))
            return False
    
    def detach_layer(self, entity_id: str) -> bool:
        """
        Detach a sync layer, keeping the Editor layer as a standalone derived layer.
        
        Unlike unsync_layer (which deletes MA3-sourced editor layers), this always
        preserves the Editor layer and its data. The layer becomes a regular,
        independently editable layer with the derived_from_ma3 flag set.
        
        Steps:
        1. Unhooks the MA3 track
        2. Clears sync metadata on the Editor layer (is_synced=False, derived_from_ma3=True)
        3. Removes the sync entity from storage (freeing the MA3 coord for re-sync)
        4. Saves and emits signals
        
        Args:
            entity_id: The sync layer entity ID
            
        Returns:
            True if successful
        """
        Log.info(f"SyncSystemManager: detach_layer entity_id={entity_id}")
        
        entity = self._synced_layers.get(entity_id)
        if not entity:
            Log.warning(f"SyncSystemManager: Entity not found: {entity_id}")
            return False
        
        try:
            # Step 1: Unhook MA3 track if hooked
            if entity.ma3_coord and entity.ma3_coord in self._hooked_tracks:
                self._unhook_ma3_track(entity.ma3_coord)
            
            # Step 2: Clear sync metadata on the Editor layer, mark as derived
            if entity.editor_layer_id and entity.editor_block_id:
                editor_api = self._get_editor_api(entity.editor_block_id)
                if editor_api:
                    try:
                        editor_api.update_layer(
                            entity.editor_layer_id,
                            is_synced=False,
                            show_manager_block_id=None,
                            ma3_track_coord=None,
                            derived_from_ma3=True,
                        )
                        Log.info(
                            f"SyncSystemManager: Cleared sync metadata on Editor layer "
                            f"'{entity.editor_layer_id}' and marked as derived_from_ma3"
                        )
                    except Exception as e:
                        Log.warning(f"SyncSystemManager: Failed to clear sync metadata via EditorAPI: {e}")
                        # Fall back to direct layer object update via panel
                        self._clear_sync_metadata_direct(entity.editor_block_id, entity.editor_layer_id)
                else:
                    # EditorAPI not available, try direct update
                    self._clear_sync_metadata_direct(entity.editor_block_id, entity.editor_layer_id)
            
            # Step 3: Remove from storage (frees MA3 coord for re-sync)
            del self._synced_layers[entity_id]
            self._save_to_settings()
            
            # Step 4: Emit signals
            self.entities_changed.emit()
            
            Log.info(
                f"SyncSystemManager: Detached sync entity '{entity.name}' - "
                f"Editor layer preserved as derived layer"
            )
            return True
            
        except Exception as e:
            Log.error(f"SyncSystemManager: detach_layer failed: {e}")
            self.error_occurred.emit(entity_id, str(e))
            return False
    
    def _clear_sync_metadata_direct(self, editor_block_id: str, editor_layer_name: str) -> None:
        """Clear sync metadata directly on the Editor layer via UI state.
        
        Fallback when EditorAPI is not available.
        """
        try:
            result = self._facade.get_ui_state(
                state_type='editor_layers',
                entity_id=editor_block_id
            )
            if result.success and result.data:
                layers = result.data.get('layers', [])
                for layer in layers:
                    if layer.get('name') == editor_layer_name:
                        layer['is_synced'] = False
                        layer['show_manager_block_id'] = None
                        layer['ma3_track_coord'] = None
                        layer['derived_from_ma3'] = True
                        break
                self._facade.set_ui_state(
                    state_type='editor_layers',
                    entity_id=editor_block_id,
                    data={'layers': layers}
                )
                Log.info(f"SyncSystemManager: Cleared sync metadata directly for '{editor_layer_name}'")
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to clear sync metadata directly: {e}")

    def resync_layer(self, entity_id: str) -> bool:
        """
        Re-check sync status for an existing synced layer.
        
        This does NOT auto-apply changes. It:
        1. Rehooks the MA3 track to ensure fresh data
        2. Compares MA3 events vs Editor events
        3. If diverged: marks DIVERGED (user resolves in UI)
        4. If clean: marks SYNCED
        
        Resolution is deferred to the user via the synced layers table
        (Apply to EZ / Apply to MA3 buttons).
        
        Args:
            entity_id: The sync layer entity ID
            
        Returns:
            True if check completed successfully
        """
        Log.info(f"SyncSystemManager: resync_layer entity_id={entity_id}")
        
        entity = self._synced_layers.get(entity_id)
        if not entity:
            Log.warning(f"SyncSystemManager: Entity not found: {entity_id}")
            return False
        
        try:
            # Rehook to ensure fresh data from MA3
            if entity.ma3_coord:
                rehook_success = self._rehook_ma3_track(entity.ma3_coord)
                if not rehook_success:
                    Log.warning(f"SyncSystemManager: Rehook failed for {entity.ma3_coord}")
            
            # Compare MA3 vs Editor
            comparison = self._compare_entity(entity)
            if comparison and comparison.diverged:
                entity.mark_diverged()
                Log.info(
                    f"SyncSystemManager: resync_layer divergence detected for '{entity.name}': "
                    f"MA3={comparison.ma3_count}, Editor={comparison.editor_count}, "
                    f"Matched={comparison.matched_count}"
                )
            else:
                entity.mark_synced()
                Log.info(f"SyncSystemManager: resync_layer '{entity.name}' is in sync")
            
            self._save_to_settings()
            self.entities_changed.emit()
            self._push_sync_state_to_all_layers()
            return True
            
        except Exception as e:
            Log.error(f"SyncSystemManager: resync_layer failed: {e}")
            entity.mark_error(str(e))
            self.error_occurred.emit(entity_id, str(e))
            return False
    
    def apply_to_ma3(self, entity_id: str) -> bool:
        """Apply the Editor layer events to the MA3 track."""
        Log.info(f"SyncSystemManager: apply_to_ma3 entity_id={entity_id}")
        entity = self._synced_layers.get(entity_id)
        if not entity:
            Log.warning(f"SyncSystemManager: Entity not found: {entity_id}")
            return False
        if not entity.editor_layer_id or not entity.ma3_coord:
            Log.warning(f"SyncSystemManager: Missing editor_layer_id or ma3_coord for {entity_id}")
            return False
        
        try:
            self._push_editor_to_ma3(entity)
            entity.mark_synced()
            self._save_to_settings()
            self.entity_updated.emit(entity_id)
            self.sync_status_changed.emit(entity_id, entity.sync_status.value)
            return True
        except Exception as e:
            Log.error(f"SyncSystemManager: apply_to_ma3 failed: {e}")
            entity.mark_error(str(e))
            self.error_occurred.emit(entity_id, str(e))
            return False
    
    def apply_to_ez(self, entity_id: str) -> bool:
        """Apply the MA3 track events to the Editor layer (fresh fetch)."""
        Log.info(f"SyncSystemManager: apply_to_ez entity_id={entity_id}")
        entity = self._synced_layers.get(entity_id)
        if not entity:
            Log.warning(f"SyncSystemManager: Entity not found: {entity_id}")
            return False
        if not entity.ma3_coord or not entity.editor_layer_id:
            Log.warning(f"SyncSystemManager: Missing ma3_coord or editor_layer_id for {entity_id}")
            return False
        
        self._force_apply_to_ez[entity.ma3_coord] = True
        self._request_ma3_events(entity.ma3_coord)
        return True

    def record_track_changed_timestamp(self, coord: str, timestamp: float) -> None:
        """Record when a track.changed message arrived for diagnostic purposes."""
        import time
        self._diag_track_changed_ts[coord] = timestamp
        # Track the earliest track.changed in a batch (reset if >2s gap)
        if self._diag_first_track_changed_ts == 0.0 or (timestamp - self._diag_first_track_changed_ts) > 2.0:
            self._diag_first_track_changed_ts = timestamp

    def request_ma3_events(self, coord: str) -> bool:
        """Request MA3 events for a track coordinate."""
        import time
        from PyQt6.QtCore import QTimer
        
        if not hasattr(self, "_pending_events_requests"):
            self._pending_events_requests: Dict[str, float] = {}
        if not hasattr(self, "_pending_events_timers"):
            self._pending_events_timers: Dict[str, QTimer] = {}
        if not hasattr(self, "_pending_events_request_ids"):
            self._pending_events_request_ids: Dict[str, int] = {}
        if not hasattr(self, "_ma3_request_counter"):
            self._ma3_request_counter = 0
        if not hasattr(self, "_last_events_request_id"):
            self._last_events_request_id: Dict[str, int] = {}
        
        existing_request = self._pending_events_requests.get(coord)
        was_deduplicated = False
        if existing_request and (time.time() - existing_request) < self._ma3_events_request_timeout_s:
            was_deduplicated = True
            Log.info(f"[MULTITRACK-DIAG] request_ma3_events DEDUPLICATED: coord={coord}, reason=pending_request")
            Log.debug(f"SyncSystemManager: Request already pending for {coord}, skipping duplicate")
            return False
        if coord in self._ma3_events_in_flight and existing_request:
            if (time.time() - existing_request) < self._ma3_events_request_timeout_s:
                was_deduplicated = True
                Log.info(f"[MULTITRACK-DIAG] request_ma3_events DEDUPLICATED: coord={coord}, reason=in_flight")
                Log.debug(f"SyncSystemManager: In-flight request exists for {coord}, skipping duplicate")
                return False
        
        requested_at = time.time()
        self._pending_events_requests[coord] = requested_at
        self._ma3_request_counter += 1
        request_id = self._ma3_request_counter
        self._pending_events_request_ids[coord] = request_id
        self._last_events_request_id[coord] = request_id
        self._ma3_events_in_flight[coord] = request_id
        
        Log.info(f"[MULTITRACK-DIAG] request_ma3_events SENT: coord={coord}, request_id={request_id}, in_flight_count={len(self._ma3_events_in_flight)}")
        
        if coord not in self._pending_events_timers:
            timer = QTimer()
            timer.setSingleShot(True)
            self._pending_events_timers[coord] = timer
        else:
            timer = self._pending_events_timers[coord]
        
        MAX_RETRIES = 2
        
        def on_timeout():
            last_request = self._pending_events_requests.get(coord, 0)
            if last_request == requested_at:
                retry_count = self._events_request_retries.get(coord, 0)
                if retry_count < MAX_RETRIES:
                    self._events_request_retries[coord] = retry_count + 1
                    Log.warning(f"[MULTITRACK-DIAG] events.list TIMEOUT for {coord}, retrying (attempt {retry_count + 1}/{MAX_RETRIES})")
                    # Clear the in-flight state so the retry is not deduplicated
                    self._pending_events_requests.pop(coord, None)
                    self._pending_events_request_ids.pop(coord, None)
                    self._ma3_events_in_flight.pop(coord, None)
                    # Re-request events for this coord
                    self.request_ma3_events(coord)
                else:
                    Log.error(f"[MULTITRACK-DIAG] events.list FAILED for {coord} after {MAX_RETRIES} retries")
                    self._pending_events_requests.pop(coord, None)
                    self._pending_events_request_ids.pop(coord, None)
                    self._ma3_events_in_flight.pop(coord, None)
                    self._events_request_retries.pop(coord, None)
        
        timer.timeout.disconnect() if timer.receivers(timer.timeout) else None
        timer.timeout.connect(on_timeout)
        timer.start(int(self._ma3_events_request_timeout_s * 1000))
        
        # Clear retry counter on fresh (non-retry) request
        if coord not in self._events_request_retries:
            self._events_request_retries[coord] = 0
        
        return self._request_ma3_events(coord, request_id=request_id)

    def get_latest_events_request_id(self, coord: str) -> Optional[int]:
        """Get latest MA3 events request ID for a track coord."""
        if hasattr(self, "_last_events_request_id"):
            return self._last_events_request_id.get(coord)
        return None
    
    # =========================================================================
    # Reconnection Handling
    # =========================================================================
    
    def on_ma3_connected(self) -> None:
        """
        Called internally when MA3 connection is established.
        
        Triggers reconnection flow:
        1. Fetch track groups from configured timecode
        2. Reconcile saved sync layers with MA3 state
        3. Hook tracks that still exist
        4. Flag missing/diverged layers for user attention
        """
        if self._ma3_connected:
            return  # Already connected, avoid duplicate processing
        
        Log.info("SyncSystemManager: MA3 connected")
        self._ma3_connected = True
        self.ma3_connection_changed.emit(True)
        
        # Clear cached events (may be stale) but keep track metadata
        # so the available layers list doesn't flicker during reconnect.
        # on_tracks_received() will replace stale track data per TC/TG
        # when fresh data arrives from MA3.
        self._ma3_track_events.clear()
        
        # Fetch MA3 structure (track groups -> tracks -> auto_reconnect)
        # This is self-contained: no panel involvement needed.
        # _fetch_ma3_structure() sends EZ.GetTrackGroups(tc) which triggers the
        # full pipeline: on_tracks_received() -> auto_reconnect_layers() -> push state
        self._fetch_ma3_structure()
        
        # Hook all MA3 tracks that are synced
        # (hooks will be validated when tracks data arrives)
        for entity_id, entity in self._synced_layers.items():
            if entity.ma3_coord:
                self._hook_ma3_track(entity.ma3_coord)
    
    def on_ma3_disconnected(self) -> None:
        """Called internally when MA3 connection is lost.
        
        Transitions SYNCED and DIVERGED entities to AWAITING_CONNECTION
        and pushes the updated sync state to editor timeline icons.
        """
        if not self._ma3_connected:
            return  # Already disconnected, avoid duplicate processing
        
        Log.info("SyncSystemManager: MA3 disconnected")
        self._ma3_connected = False
        self._hooked_tracks.clear()
        self._hooked_track_groups.clear()
        self.ma3_connection_changed.emit(False)
        
        # Mark all active entities as AWAITING_CONNECTION (no MA3 link)
        downgraded = 0
        for entity_id, entity in self._synced_layers.items():
            if entity.sync_status in (SyncStatus.SYNCED, SyncStatus.DIVERGED):
                entity.sync_status = SyncStatus.AWAITING_CONNECTION
                downgraded += 1
        if downgraded:
            Log.info(f"SyncSystemManager: Downgraded {downgraded} entities to AWAITING_CONNECTION on disconnect")
        self._save_to_settings()
        self.entities_changed.emit()
        self._push_sync_state_to_all_layers()
    
    def _request_ma3_track_groups(self) -> None:
        """Request track groups from MA3 to rebuild local state."""
        if not self.ma3_comm_service:
            return
        
        tc = self._configured_timecode
        lua_cmd = f"EZ.GetTrackGroups({tc})"
        self._send_lua_command_with_target(lua_cmd)
        Log.info(f"SyncSystemManager: Requested track groups for TC{tc}")
    
    def reconcile_synced_layers(self) -> Dict[str, str]:
        """
        Reconcile saved sync layers with current MA3 state.
        
        Called after MA3 track data is received on reconnection.
        
        Returns:
            Dict mapping entity_id to status: "ok", "missing", "diverged"
        """
        Log.info("SyncSystemManager: Reconciling synced layers")
        results = {}
        
        for entity_id, entity in self._synced_layers.items():
            if not entity.ma3_coord:
                results[entity_id] = "no_ma3_coord"
                continue
            
            # Check if MA3 track still exists
            if entity.ma3_coord not in self._ma3_tracks:
                # Track missing from MA3
                Log.warning(f"SyncSystemManager: MA3 track missing: {entity.ma3_coord}")
                entity.mark_error("MA3 track not found after reconnect")
                results[entity_id] = "missing"
                self.error_occurred.emit(entity_id, "MA3 track not found")
                continue
            
            # Track exists - check for divergence
            comparison = self._compare_entity(entity)
            if comparison and comparison.diverged:
                Log.info(f"SyncSystemManager: Divergence detected for {entity.ma3_coord}")
                entity.mark_diverged()
                results[entity_id] = "diverged"
                self.divergence_detected.emit(entity_id, comparison)
            else:
                # Re-hook and mark synced
                entity.mark_synced()
                self._hook_ma3_track(entity.ma3_coord)
                results[entity_id] = "ok"
        
        self._save_to_settings()
        self.entities_changed.emit()
        
        Log.info(f"SyncSystemManager: Reconciliation complete: {results}")
        return results
    
    def check_divergences(self) -> List[str]:
        """
        Check all synced layers for divergence.
        
        Returns:
            List of entity IDs that have diverged
        """
        Log.info("SyncSystemManager: Checking divergences")
        diverged_ids = []
        
        for entity_id, entity in self._synced_layers.items():
            if not entity.is_synced:
                continue
            
            comparison = self._compare_entity(entity)
            if comparison and comparison.diverged:
                entity.mark_diverged()
                diverged_ids.append(entity_id)
                self.divergence_detected.emit(entity_id, comparison)
        
        if diverged_ids:
            self._save_to_settings()
            self.entities_changed.emit()
        
        return diverged_ids
    
    def resolve_divergence(
        self,
        entity_id: str,
        strategy: str,
    ) -> bool:
        """
        Resolve divergence for a specific entity.
        
        Args:
            entity_id: The entity ID
            strategy: "ma3_wins", "ez_wins", or "merge"
            
        Returns:
            True if resolved successfully
        """
        Log.info(f"SyncSystemManager: resolve_divergence entity_id={entity_id} strategy={strategy}")
        
        entity = self._synced_layers.get(entity_id)
        if not entity:
            return False
        
        try:
            if strategy == "ma3_wins":
                self._push_ma3_to_editor(entity)
            elif strategy == "ez_wins":
                self._push_editor_to_ma3(entity)
            elif strategy == "merge":
                self._merge_events(entity)
            else:
                Log.error(f"Unknown resolution strategy: {strategy}")
                return False
            
            entity.mark_synced()
            self._save_to_settings()
            self._push_sync_state_to_all_layers()
            
            self.entity_updated.emit(entity_id)
            self.entities_changed.emit()
            self.sync_status_changed.emit(entity_id, entity.sync_status.value)
            Log.info(f"SyncSystemManager: Divergence resolved for '{entity.name}' via {strategy}")
            return True
            
        except Exception as e:
            Log.error(f"SyncSystemManager: resolve_divergence failed: {e}")
            entity.mark_error(str(e))
            self.error_occurred.emit(entity_id, str(e))
            return False
    
    # =========================================================================
    # Data Access for UI
    # =========================================================================
    
    def get_synced_layers(self) -> List[SyncLayerEntity]:
        """Get all currently synced layers."""
        return list(self._synced_layers.values())
    
    def get_synced_layer(self, entity_id: str) -> Optional[SyncLayerEntity]:
        """Get a specific synced layer by ID."""
        return self._synced_layers.get(entity_id)
    
    def get_synced_layer_by_ma3_coord(self, coord: str) -> Optional[SyncLayerEntity]:
        """Get synced layer by MA3 coordinate."""
        for entity in self._synced_layers.values():
            if entity.ma3_coord == coord:
                Log.debug(f"SyncSystemManager: Found entity for coord '{coord}' -> editor_layer_id={entity.editor_layer_id}")
                return entity
        # Debug: no match found
        Log.debug(f"SyncSystemManager: No entity found for coord '{coord}' in {len(self._synced_layers)} synced layers")
        return None
    
    def remap_ma3_track_by_name(
        self,
        timecode_no: int,
        track_group_no: int,
        track_no: int,
        track_name: str,
        track_note: str = "",
    ) -> Optional[SyncLayerEntity]:
        """
        Remap a synced layer to a new MA3 track index based on EZ ID or track name.
        
        Matching priority:
        1. EZ ID: If track_note starts with "ez:", match by entity.ez_track_id
        2. Normalized name: Match by normalized track name (strip ez_/ma3_ prefix)
        
        This is used when MA3 track indices shift (e.g., deletions), but
        the track object/name remains the same.
        """
        # Priority 1: Match by EZ ID from track .note
        if track_note and track_note.startswith("ez:"):
            for entity in self._synced_layers.values():
                if entity.ma3_timecode_no != timecode_no or entity.ma3_track_group != track_group_no:
                    continue
                if entity.ez_track_id == track_note:
                    if entity.ma3_track != track_no:
                        self._remap_entity_ma3_coord(entity, track_no)
                    return entity
        
        # Priority 2: Match by normalized name
        if not track_name:
            return None
        
        target_name = normalize_track_name(track_name)
        if not target_name:
            return None
        
        for entity in self._synced_layers.values():
            if entity.ma3_timecode_no != timecode_no or entity.ma3_track_group != track_group_no:
                continue
            if normalize_track_name(entity.name) != target_name:
                continue
            if entity.ma3_track != track_no:
                self._remap_entity_ma3_coord(entity, track_no)
            return entity
        
        return None

    def _remap_entity_ma3_coord(self, entity: SyncLayerEntity, new_track_no: int) -> None:
        """Update entity MA3 coord and move per-track state to new key."""
        if not entity.ma3_timecode_no or not entity.ma3_track_group:
            return
        
        old_coord = entity.ma3_coord
        new_coord = f"tc{entity.ma3_timecode_no}_tg{entity.ma3_track_group}_tr{new_track_no}"
        if not old_coord or old_coord == new_coord:
            return
        
        def _move(mapping: Dict[str, Any], old_key: str, new_key: str) -> None:
            if old_key in mapping:
                value = mapping.pop(old_key)
                mapping[new_key] = value
        
        _move(self._hooked_tracks, old_coord, new_coord)
        _move(self._syncing_from_ma3, old_coord, new_coord)
        _move(self._last_ma3_push_time, old_coord, new_coord)
        _move(self._last_editor_push_time, old_coord, new_coord)
        _move(self._skip_initial_hook, old_coord, new_coord)
        _move(self._ma3_events_in_flight, old_coord, new_coord)
        _move(self._force_apply_to_ez, old_coord, new_coord)
        _move(self._ma3_apply_cooldown_until, old_coord, new_coord)
        _move(self._ma3_track_events, old_coord, new_coord)
        
        track_info = self._ma3_tracks.pop(old_coord, None)
        if track_info:
            track_info.track_no = new_track_no
            track_info.coord = new_coord
            self._ma3_tracks[new_coord] = track_info
        
        entity.ma3_track = new_track_no
        entity.ma3_coord = new_coord
        
        Log.info(f"SyncSystemManager: Remapped MA3 coord {old_coord} -> {new_coord}")
        self._save_to_settings()
        self.entity_updated.emit(entity.id)
        self.entities_changed.emit()

    def auto_reconnect_layers(self) -> Dict[str, str]:
        """Attempt to reconnect disconnected/unmapped sync entities to MA3 tracks.
        
        Uses a two-priority matching strategy:
        1. EZ ID: entity.ez_track_id matches track.note (exact, most reliable)
        2. Normalized name: entity name matches track name (strip ez_/ma3_ prefix)
        
        Data safety rules:
        - Never auto-delete: unmatched entities stay as DISCONNECTED
        - Never auto-overwrite: ambiguous matches (multiple candidates) are skipped
        
        Returns:
            Dict of {entity_id: "reconnected" | "ambiguous" | "no_match"}
        """
        results: Dict[str, str] = {}
        
        # Collect entities that need reconnection.
        # AWAITING_CONNECTION entities were downgraded from SYNCED on disconnect
        # or project load without MA3. PENDING entities are mid-reconnect
        # waiting for divergence check.
        reconnectable = [
            entity for entity in self._synced_layers.values()
            if entity.sync_status in (
                SyncStatus.DISCONNECTED, SyncStatus.UNMAPPED,
                SyncStatus.ERROR, SyncStatus.PENDING,
                SyncStatus.AWAITING_CONNECTION,
            )
            and entity.has_ma3_side
        ]
        
        
        if not reconnectable:
            return results
        
        # Build lookup tables from current MA3 tracks
        # EZ ID index: note -> [MA3TrackInfo, ...]
        ez_id_index: Dict[str, List[MA3TrackInfo]] = {}
        # Normalized name index: normalized_name -> [MA3TrackInfo, ...]
        name_index: Dict[str, List[MA3TrackInfo]] = {}
        
        for track_info in self._ma3_tracks.values():
            if track_info.note and track_info.note.startswith("ez:"):
                ez_id_index.setdefault(track_info.note, []).append(track_info)
            norm_name = normalize_track_name(track_info.name)
            if norm_name:
                name_index.setdefault(norm_name, []).append(track_info)
        
        # Track which MA3 coords have already been claimed (avoid double-mapping)
        claimed_coords: set = set()
        for entity in self._synced_layers.values():
            if entity.sync_status in (SyncStatus.SYNCED, SyncStatus.DIVERGED, SyncStatus.PENDING, SyncStatus.AWAITING_CONNECTION):
                if entity.ma3_coord:
                    claimed_coords.add(entity.ma3_coord)
        
        for entity in reconnectable:
            match: Optional[MA3TrackInfo] = None
            match_method = ""
            
            # The entity's own current coord should not block itself from reconnecting
            own_coord = entity.ma3_coord
            
            # Priority 0: If entity already has a valid coord that exists in MA3 tracks,
            # reconnect directly to it (no matching needed)
            if own_coord and own_coord in self._ma3_tracks:
                match = self._ma3_tracks[own_coord]
                match_method = "existing_coord"
            
            # Priority 1: Match by EZ ID
            if not match and entity.ez_track_id:
                candidates = ez_id_index.get(entity.ez_track_id, [])
                # Exclude claimed coords BUT allow entity's own coord
                unclaimed = [c for c in candidates if c.coord not in claimed_coords or c.coord == own_coord]
                if len(unclaimed) == 1:
                    match = unclaimed[0]
                    match_method = "ez_id"
                elif len(unclaimed) > 1:
                    results[entity.id] = "ambiguous"
                    Log.warning(
                        f"SyncSystemManager.auto_reconnect: Ambiguous EZ ID match for "
                        f"entity '{entity.name}' (ez_track_id='{entity.ez_track_id}'), "
                        f"{len(unclaimed)} candidates"
                    )
                    continue
            
            # Priority 2: Match by normalized name
            if not match:
                entity_norm_name = normalize_track_name(entity.name)
                if entity_norm_name:
                    candidates = name_index.get(entity_norm_name, [])
                    # Exclude claimed coords BUT allow entity's own coord
                    unclaimed = [c for c in candidates if c.coord not in claimed_coords or c.coord == own_coord]
                    if len(unclaimed) == 1:
                        match = unclaimed[0]
                        match_method = "name"
                    elif len(unclaimed) > 1:
                        results[entity.id] = "ambiguous"
                        Log.warning(
                            f"SyncSystemManager.auto_reconnect: Ambiguous name match for "
                            f"entity '{entity.name}' (normalized='{entity_norm_name}'), "
                            f"{len(unclaimed)} candidates"
                        )
                        continue
            
            if match:
                # Reconnect: update entity coord to the matched track
                old_coord = entity.ma3_coord
                new_coord = match.coord
                
                if old_coord != new_coord:
                    self._remap_entity_ma3_coord(entity, match.track_no)
                
                # Update EZ track ID if the track has one and entity doesn't
                if match.note and match.note.startswith("ez:") and not entity.ez_track_id:
                    entity.ez_track_id = match.note
                
                # Mark as PENDING (not SYNCED) until divergence check completes.
                # When MA3 events arrive via on_track_events_received(), the
                # comparison will promote to SYNCED or flag as DIVERGED.
                entity.sync_status = SyncStatus.PENDING
                entity.error_message = None
                claimed_coords.add(new_coord)
                results[entity.id] = "reconnected"
                
                Log.info(
                    f"SyncSystemManager.auto_reconnect: Reconnected entity '{entity.name}' "
                    f"to {new_coord} via {match_method} (PENDING until divergence check)"
                )
                
                # Re-hook the track for live updates and request events
                # (triggers on_track_events_received -> divergence check)
                if self.ma3_comm_service:
                    self._hook_ma3_track(new_coord)
                    self._request_ma3_events(new_coord)
            else:
                results[entity.id] = "no_match"
        
        # Save if any reconnections happened
        reconnected_count = sum(1 for v in results.values() if v == "reconnected")
        if reconnected_count > 0:
            self._save_to_settings()
            self.entities_changed.emit()
            # Update editor icons from "disconnected" (orange) to "active" (green)
            self._push_sync_state_to_all_layers()
            Log.info(
                f"SyncSystemManager.auto_reconnect: {reconnected_count} reconnected, "
                f"{sum(1 for v in results.values() if v == 'ambiguous')} ambiguous, "
                f"{sum(1 for v in results.values() if v == 'no_match')} no match"
            )
        
        return results

    def get_synced_layer_by_editor_layer(self, layer_id: str) -> Optional[SyncLayerEntity]:
        """Get synced layer by Editor layer ID."""
        for entity in self._synced_layers.values():
            if entity.editor_layer_id == layer_id:
                return entity
        return None

    def notify_editor_layer_deleted(self, editor_layer_name: str) -> bool:
        """Handle notification that an Editor layer has been deleted.
        
        Finds the matching sync entity, marks it as DISCONNECTED,
        clears the stale Editor references, and persists the change.
        The entity is kept so the user can decide in the ShowManager
        Layer Sync tab whether to reconnect or remove it.
        
        Args:
            editor_layer_name: The name (layer_id) of the deleted Editor layer.
            
        Returns:
            True if a matching entity was found and updated, False otherwise.
        """
        entity = self.get_synced_layer_by_editor_layer(editor_layer_name)
        if not entity:
            return False

        # Unhook MA3 track to stop pushing stale data
        if entity.ma3_coord and entity.ma3_coord in self._hooked_tracks:
            self._unhook_ma3_track(entity.ma3_coord)

        entity.mark_disconnected(
            f"Editor layer '{editor_layer_name}' was deleted. "
            "Reconnect to another Editor layer or remove this sync entry."
        )
        entity.unlink_editor()

        self._save_to_settings()
        self.entity_updated.emit(entity.id)
        self.sync_status_changed.emit(entity.id, SyncStatus.DISCONNECTED.value)
        self.entities_changed.emit()

        Log.info(
            f"SyncSystemManager: Marked entity '{entity.name}' as DISCONNECTED "
            f"(Editor layer '{editor_layer_name}' was deleted)"
        )
        return True

    def get_available_ma3_tracks(self) -> List[Dict[str, Any]]:
        """
        Get MA3 tracks that are not currently actively synced.
        
        Only excludes tracks with SYNCED, PENDING, DIVERGED, or
        AWAITING_CONNECTION status. DISCONNECTED, ERROR, and UNMAPPED
        tracks are considered available for (re-)syncing.
        
        Returns:
            List of track info dicts
        """
        # Get all MA3 tracks from local state
        all_tracks = self._get_all_ma3_tracks()
        
        # Only filter out tracks that are actively synced (not disconnected/error/unmapped)
        active_statuses = (SyncStatus.SYNCED, SyncStatus.PENDING, SyncStatus.DIVERGED, SyncStatus.AWAITING_CONNECTION)
        synced_coords = {
            e.ma3_coord for e in self._synced_layers.values()
            if e.ma3_coord and e.sync_status in active_statuses
        }
        
        return [t for t in all_tracks if t.get("coord") not in synced_coords]
    
    def get_available_editor_layers(self) -> List[Dict[str, Any]]:
        """
        Get Editor layers that are not currently synced.
        
        Returns:
            List of layer info dicts
        """
        # Get all Editor layers
        all_layers = self._get_all_editor_layers()
        
        # Filter out already synced
        synced_layer_ids = {e.editor_layer_id for e in self._synced_layers.values() if e.editor_layer_id}
        
        return [l for l in all_layers if l.get("layer_id") not in synced_layer_ids]
    
    # =========================================================================
    # Backward Compatibility Methods
    # =========================================================================
    # These methods provide a similar API to the old LayerSyncController
    # to ease migration of legacy UI code.
    
    def is_synced(self, entity_type: str, source_id: str) -> bool:
        """
        Check if an entity is currently synced.
        
        Args:
            entity_type: "ma3" or "editor"
            source_id: MA3 coord or Editor layer_id
            
        Returns:
            True if the entity is in the synced list
        """
        if entity_type == "ma3":
            return self.get_synced_layer_by_ma3_coord(source_id) is not None
        elif entity_type == "editor":
            return self.get_synced_layer_by_editor_layer(source_id) is not None
        return False
    
    def get_ma3_sourced_layers(self) -> List[SyncLayerEntity]:
        """Get all synced layers that originated from MA3."""
        return [e for e in self._synced_layers.values() if e.source == SyncSource.MA3]
    
    def get_editor_sourced_layers(self) -> List[SyncLayerEntity]:
        """Get all synced layers that originated from Editor."""
        return [e for e in self._synced_layers.values() if e.source == SyncSource.EDITOR]
    
    # =========================================================================
    # Sequence Management
    # =========================================================================
    
    def set_sequence(self, entity_id: str, sequence_no: int) -> bool:
        """
        Update sequence assignment for a synced layer.
        
        Args:
            entity_id: The entity ID
            sequence_no: New sequence number
            
        Returns:
            True if updated successfully
        """
        entity = self._synced_layers.get(entity_id)
        if not entity:
            return False
        
        entity.settings.sequence_no = sequence_no
        self._save_to_settings()
        
        # Update MA3 track.target if connected
        if entity.ma3_coord and self._ma3_connected:
            self._update_ma3_sequence(entity)
        
        self.entity_updated.emit(entity_id)
        return True
    
    def set_track_group(self, entity_id: str, track_group_no: int) -> bool:
        """
        Update track group assignment for a synced layer.
        
        Args:
            entity_id: The entity ID
            track_group_no: New track group number (1-99999)
            
        Returns:
            True if updated successfully
        """
        entity = self._synced_layers.get(entity_id)
        if not entity:
            return False
        
        entity.settings.track_group_no = track_group_no
        self._save_to_settings()
        
        self.entity_updated.emit(entity_id)
        return True
    
    # =========================================================================
    # Private: Sync Operations
    # =========================================================================
    
    def _sync_from_ma3(
        self,
        ma3_coord: str,
        editor_layer_id: Optional[str],
        auto_create: bool,
        event_authority: Optional[str] = None,
    ) -> Optional[str]:
        """Sync from MA3 track to Editor layer.
        
        Args:
            ma3_coord: MA3 track coordinate
            editor_layer_id: Optional Editor layer to map to (None = create new)
            auto_create: If True, auto-create Editor layer
            event_authority: When mapping to existing, controls event direction:
                "editor" = push Editor events to MA3
                "ma3" = push MA3 events to Editor (default for this direction)
                None = default behavior (push MA3 to Editor)
        """
        
        # Check if already synced
        existing = self.get_synced_layer_by_ma3_coord(ma3_coord)
        if existing:
            Log.info(f"SyncSystemManager: MA3 coord {ma3_coord} already synced")
            return existing.id
        
        # Get MA3 track info
        track_info = self._get_ma3_track_info(ma3_coord)
        if not track_info:
            Log.error(f"SyncSystemManager: MA3 track not found: {ma3_coord}")
            return None
        
        # Create entity
        entity_id = str(uuid.uuid4())
        sequence_no = track_info.get("sequence_no")
        
        # Check if MA3 track already has an EZ ID in its .note
        existing_note = track_info.get("note", "")
        existing_ez_id = existing_note if existing_note.startswith("ez:") else None
        
        entity = SyncLayerEntity.from_ma3_track(
            id=entity_id,
            coord=ma3_coord,
            timecode_no=track_info.get("timecode_no", 1),
            track_group=track_info.get("track_group", 1),
            track=track_info.get("track", 1),
            name=track_info.get("name", ""),
            group_name=track_info.get("group_name"),
            event_count=track_info.get("event_count", 0),
            sequence_no=sequence_no,
            ez_track_id=existing_ez_id,
        )
        
        # Determine Editor layer
        mapping_to_existing = False
        if not editor_layer_id:
            # Create with proper group info from MA3 track
            group_id = f"tc{track_info.get('timecode_no', 1)}"
            group_name = f"TC {track_info.get('timecode_no', 1)}"
            editor_layer_id = self._create_editor_layer(
                name=entity.name,
                group_id=group_id,
                group_name=group_name,
                is_synced=True,
                ma3_track_coord=ma3_coord,
            )
        else:
            # Will update existing Editor layer AFTER entity is saved to settings,
            # so the sync indicator lookup finds the entity on the BlockUpdated reload
            mapping_to_existing = True
        
        if editor_layer_id:
            editor_block_id = self._get_editor_block_id()
            entity.link_to_editor(editor_layer_id, editor_block_id)
        
        # Stamp EZ track ID on the MA3 track .note property for persistent identity.
        # Only write if the track doesn't already have an ez: prefix in .note.
        if editor_layer_id and not existing_ez_id:
            # Derive EZ ID from the editor layer name
            ez_track_id = f"ez:{editor_layer_id}"
            entity.ez_track_id = ez_track_id
            # Send command to MA3 to write the .note property
            tc_no = track_info.get("timecode_no", 1)
            tg_no = track_info.get("track_group", 1)
            tr_no = track_info.get("track", 1)
            self._send_lua_command_with_target(
                f'EZ.SetTrackNote({tc_no}, {tg_no}, {tr_no}, "{ez_track_id}")'
            )
            Log.info(f"SyncSystemManager: Stamped EZ ID '{ez_track_id}' on MA3 track {ma3_coord}")
        
        # Hook MA3 track for changes and handle initial event sync
        if self.ma3_comm_service:
            self._hook_ma3_track(ma3_coord)
            
            if event_authority == "editor" and mapping_to_existing:
                # User chose to keep Editor events: push Editor -> MA3
                self._push_editor_to_ma3(entity)
                Log.info(f"SyncSystemManager: Hooked track, pushed Editor events to MA3 for {ma3_coord}")
            else:
                # Default: request MA3 events (will push MA3 -> Editor asynchronously)
                self._request_ma3_events(ma3_coord)
                Log.info(f"SyncSystemManager: Hooked track and requested MA3 events for {ma3_coord}")
        
        entity.mark_synced()
        
        # Store and save
        self._synced_layers[entity_id] = entity
        self._save_to_settings()
        
        # Update existing Editor layer's sync properties AFTER the entity is persisted,
        # so the BlockUpdated -> Editor reload -> sync indicator lookup finds the entity.
        if mapping_to_existing and editor_layer_id:
            editor_block_id = self._get_editor_block_id()
            if editor_block_id:
                try:
                    self._update_existing_editor_layer_sync(
                        editor_block_id, editor_layer_id, ma3_coord
                    )
                except Exception as _dbg_exc2:
                    raise
        
        self.entities_changed.emit()
        return entity_id
    
    def _sync_from_editor(
        self,
        editor_layer_id: str,
        ma3_coord: Optional[str],
        auto_create: bool,
        event_authority: Optional[str] = None,
    ) -> Optional[str]:
        """Sync from Editor layer to MA3 track.
        
        Args:
            editor_layer_id: Editor layer ID
            ma3_coord: Optional MA3 coord to map to (None = auto-create)
            auto_create: If True, auto-create MA3 track
            event_authority: When mapping to existing, controls event direction:
                "editor" = push Editor events to MA3 (default for this direction)
                "ma3" = push MA3 events to Editor
                None = default behavior (push Editor to MA3)
        """
        # Check if already synced
        existing = self.get_synced_layer_by_editor_layer(editor_layer_id)
        if existing:
            Log.info(f"SyncSystemManager: Editor layer {editor_layer_id} already synced")
            return existing.id
        
        # Get Editor layer info
        layer_info = self._get_editor_layer_info(editor_layer_id)
        if not layer_info:
            Log.error(f"SyncSystemManager: Editor layer not found: {editor_layer_id}")
            return None
        
        # Create entity
        entity_id = str(uuid.uuid4())
        entity = SyncLayerEntity.from_editor_layer(
            id=entity_id,
            layer_id=editor_layer_id,
            block_id=layer_info.get("block_id", self._get_editor_block_id()),
            name=layer_info.get("name", editor_layer_id),
            group_name=layer_info.get("group_name"),
            event_count=layer_info.get("event_count", 0),
        )
        
        # Determine MA3 track
        if ma3_coord:
            # Get full track info from cache (has sequence_no, track_group, etc.)
            track_info = self._get_ma3_track_info(ma3_coord)
            parts = self._parse_ma3_coord(ma3_coord)
            if parts:
                entity.link_to_ma3(
                    coord=ma3_coord,
                    timecode_no=parts["timecode_no"],
                    track_group=parts["track_group"],
                    track=parts["track"],
                )
                
                # Set accurate sequence/track group on entity settings
                if track_info:
                    seq_no = track_info.get("sequence_no")
                    entity.settings.sequence_no = seq_no if seq_no is not None else 1
                    entity.settings.track_group_no = track_info.get("track_group", parts["track_group"])
                else:
                    entity.settings.track_group_no = parts["track_group"]
                
                # Stamp EZ track ID on the MA3 track .note property for persistent identity.
                if editor_layer_id and self._ma3_connected:
                    ez_track_id = f"ez:{editor_layer_id}"
                    entity.ez_track_id = ez_track_id
                    self._send_lua_command_with_target(
                        f'EZ.SetTrackNote({parts["timecode_no"]}, {parts["track_group"]}, {parts["track"]}, "{ez_track_id}")'
                    )
                    Log.info(f"SyncSystemManager: Stamped EZ ID '{ez_track_id}' on MA3 track {ma3_coord}")
                
                # Hook MA3 track
                if self._ma3_connected:
                    self._hook_ma3_track(ma3_coord)
                
                # Handle initial event sync based on event_authority
                if event_authority == "ma3":
                    # User chose to keep MA3 events: pull MA3 -> Editor
                    self._request_ma3_events(ma3_coord)
                    Log.info(f"SyncSystemManager: Linked to MA3, requesting MA3 events for {ma3_coord}")
                else:
                    # Default: push Editor events to MA3
                    self._push_editor_to_ma3(entity)
                    Log.info(f"SyncSystemManager: Linked to MA3, pushed Editor events for {ma3_coord}")
        
        entity.mark_synced()
        
        # Store and save
        self._synced_layers[entity_id] = entity
        self._save_to_settings()
        
        # Update the Editor layer's sync properties so the timeline shows the green icon.
        # Done AFTER entity is persisted so the sync indicator lookup finds it.
        editor_block_id = self._get_editor_block_id()
        if editor_block_id and ma3_coord:
            self._update_existing_editor_layer_sync(
                editor_block_id, editor_layer_id, ma3_coord
            )
        
        self.entities_changed.emit()
        return entity_id
    
    # =========================================================================
    # Private: Data Operations
    # =========================================================================
    
    # Constant for MA3->Editor coalesce window
    _COALESCE_DELAY_MS = 300  # Wait 300ms after last change before pushing
    
    def _schedule_push_ma3_to_editor(self, entity: SyncLayerEntity) -> None:
        """
        Schedule a push from MA3 to Editor with GLOBAL multi-track coalescing.
        
        Instead of per-coord independent timers, this uses a single global
        coalesce window that accumulates all changed coords. When any track
        reports new events, it is added to the pending set and the global
        timer is reset. After 300ms of quiet (no new track arrivals), ALL
        pending tracks are pushed in a single atomic batch.
        
        This ensures that when a user drags events across N tracks in MA3,
        all N tracks update simultaneously in the Editor rather than in a
        staggered fashion.
        
        IMPORTANT: Uses QTimer (main thread) instead of threading.Timer
        because database operations must happen on the main thread.
        """
        from PyQt6.QtCore import QTimer
        import time
        
        coord = entity.ma3_coord
        if not coord:
            return
        
        
        # Initialize global coalescing state if needed
        if not hasattr(self, "_global_coalesce_timer"):
            self._global_coalesce_timer: Optional[QTimer] = None
        if not hasattr(self, "_global_coalesce_pending"):
            self._global_coalesce_pending: Dict[str, SyncLayerEntity] = {}
        if not hasattr(self, "_global_coalesce_last_arrival"):
            self._global_coalesce_last_arrival: float = 0.0
        
        # Add this coord to the pending set and record arrival time
        self._global_coalesce_pending[coord] = entity
        self._global_coalesce_last_arrival = time.time()
        
        Log.info(f"[MULTITRACK-DIAG] _schedule_push_ma3_to_editor: coord={coord}, timer_start={time.time():.4f}, pending_coords_count={len(self._global_coalesce_pending)}, pending_coords={list(self._global_coalesce_pending.keys())}")
        
        # Create the global timer if it doesn't exist
        if self._global_coalesce_timer is None:
            self._global_coalesce_timer = QTimer()
            self._global_coalesce_timer.setSingleShot(True)
            self._global_coalesce_timer.timeout.connect(self._do_global_coalesce_push)
        
        # (Re)start the global timer -- each new coord arrival resets the window
        self._global_coalesce_timer.start(self._COALESCE_DELAY_MS)
        
        Log.debug(f"SyncSystemManager: Global coalesce timer (re)started for {self._COALESCE_DELAY_MS}ms (pending: {list(self._global_coalesce_pending.keys())})")
    
    def _do_global_coalesce_push(self) -> None:
        """
        Execute the global coalesce batch push.
        
        Called by the global coalesce QTimer. Checks if the quiet window
        has elapsed, and if so, pushes ALL pending coords to the Editor
        in a single batch.
        """
        import time
        
        now = time.time()
        elapsed_ms = (now - self._global_coalesce_last_arrival) * 1000
        
        if elapsed_ms < self._COALESCE_DELAY_MS - 10:  # 10ms tolerance
            # A new coord arrived during the window, reschedule
            remaining = int(self._COALESCE_DELAY_MS - elapsed_ms)
            Log.debug(f"SyncSystemManager: Global coalesce rescheduled, waiting {remaining}ms more")
            if self._global_coalesce_timer:
                self._global_coalesce_timer.start(remaining)
            return
        
        # Take a snapshot of all pending entities and clear the pending set
        pending = dict(self._global_coalesce_pending)
        self._global_coalesce_pending.clear()
        
        if not pending:
            return
        
        Log.info(f"[MULTITRACK-DIAG] Global coalesce batch push: {len(pending)} coords: {list(pending.keys())}")
        
        
        # Push all pending coords in sequence
        for push_coord, push_entity in pending.items():
            try:
                self._push_ma3_to_editor(push_entity)
                push_entity.mark_synced()
                self.entity_updated.emit(push_entity.id)
                Log.info(f"SyncSystemManager: Batch push complete for {push_coord}")
            except Exception as e:
                Log.error(f"SyncSystemManager: Batch push failed for {push_coord}: {e}")
        
        # Save settings once after all pushes
        self._save_to_settings()
        
        
        # Reset diagnostic batch tracker
        self._diag_first_track_changed_ts = 0.0
    
    def _push_ma3_to_editor(self, entity: SyncLayerEntity) -> None:
        """
        Push MA3 events to Editor layer using bulletproof full-replace strategy.
        
        This method:
        1. Gets or creates a dedicated EventDataItem for MA3 sync events
        2. Clears ALL events from that data item directly
        3. Adds ALL events from MA3 with ma3_idx in metadata
        4. Stores the data_item_id in the entity for future updates
        
        NOTE: This is called by _schedule_push_ma3_to_editor after coalescing.
        """
        import time
        import hashlib
        
        coord = entity.ma3_coord
        if not coord or not entity.editor_layer_id:
            Log.warning("SyncSystemManager: Cannot push - missing ma3_coord or editor_layer_id")
            return
        
        # Sync guard: prevent concurrent/recursive pushes for same track
        already_syncing = self._syncing_from_ma3.get(coord, False)
        
        if already_syncing:
            Log.debug(f"SyncSystemManager: Skipping push for {coord} - already syncing")
            return
        
        # Multi-track diagnostic: latency from first track.changed
        import time as _time_diag
        _push_ts = _time_diag.time()
        _since_first_ms = ((_push_ts - self._diag_first_track_changed_ts) * 1000) if self._diag_first_track_changed_ts else -1
        _since_this_track_ms = ((_push_ts - self._diag_track_changed_ts.get(coord, 0)) * 1000) if self._diag_track_changed_ts.get(coord) else -1
        Log.info(f"[MULTITRACK-DIAG] _push_ma3_to_editor: coord={coord}, since_first_track_changed_ms={_since_first_ms:.1f}, since_this_track_changed_ms={_since_this_track_ms:.1f}")
        
        # Get MA3 events first to compute fingerprint
        ma3_events = self._get_ma3_events(coord)
        
        
        # Compute a fingerprint of the MA3 events to detect actual changes
        event_fingerprint = hashlib.md5(
            str([(e.get("time", 0), e.get("name", "")) for e in ma3_events]).encode()
        ).hexdigest()
        
        # Skip if we've already pushed this exact event set
        if not hasattr(self, "_last_push_fingerprint"):
            self._last_push_fingerprint: Dict[str, str] = {}
        
        last_fingerprint = self._last_push_fingerprint.get(coord)
        fingerprint_match = last_fingerprint == event_fingerprint
        
        
        force_apply = bool(self._force_apply_to_ez.get(coord))
        if fingerprint_match and not force_apply:
            Log.debug(f"SyncSystemManager: Skipping push for {coord} - events unchanged")
            return
        
        # Set sync guard
        self._syncing_from_ma3[coord] = True
        self._last_ma3_push_time[coord] = time.time()
        self._last_push_fingerprint[coord] = event_fingerprint
        
        try:
            layer_name = entity.editor_layer_id  # e.g., "clap"
            Log.info(f"SyncSystemManager: _push_ma3_to_editor - layer={layer_name}, events={len(ma3_events)}")
            
            self._apply_ma3_events_via_editor_api(entity, ma3_events, layer_name, coord)
            if force_apply:
                self._force_apply_to_ez.pop(coord, None)
            self._ma3_apply_cooldown_until[coord] = time.time() + self._ma3_apply_cooldown_s
        except Exception as e:
            Log.error(f"SyncSystemManager: Failed to push MA3 to Editor: {e}")
            import traceback
            Log.debug(traceback.format_exc())
        finally:
            # Release sync guard
            self._syncing_from_ma3[coord] = False

    def _apply_ma3_events_via_editor_api(
        self,
        entity: SyncLayerEntity,
        ma3_events: List[Dict[str, Any]],
        layer_name: str,
        coord: str
    ) -> None:
        """Apply MA3 events to Editor using EditorAPI (clear + add)."""
        if not entity.editor_block_id:
            Log.warning("SyncSystemManager: Cannot apply via EditorAPI - missing editor_block_id")
            return
        
        from src.features.blocks.application.editor_api import EditorAPI
        
        api = EditorAPI(self._facade, entity.editor_block_id, None)
        
        # Use a single EventDataItem per MA3 timecode for MA3-sourced layers
        source = "ma3_sync"
        if entity.source == SyncSource.MA3:
            timecode_data_item_id = self._get_or_create_ma3_timecode_data_item(entity, coord)
            if timecode_data_item_id:
                entity.editor_data_item_id = timecode_data_item_id
                source = "ma3"

        # Use the original data item's source when available to avoid creating a new data item
        if self._facade.data_item_repo:
            if entity.editor_data_item_id:
                item = self._facade.data_item_repo.get(entity.editor_data_item_id)
                if item and hasattr(item, "metadata"):
                    item_source = (item.metadata or {}).get("source")
                    if item_source:
                        source = item_source
            if source == "ma3_sync":
                # Fallback: find the data item that already owns this layer
                from src.shared.domain.entities import EventDataItem
                items = self._facade.data_item_repo.list_by_block(entity.editor_block_id)
                for item in items:
                    if isinstance(item, EventDataItem):
                        if any(getattr(e, "classification", None) == layer_name for e in item.get_events()):
                            item_source = (item.metadata or {}).get("source")
                            if item_source:
                                source = item_source
                            if not entity.editor_data_item_id:
                                entity.editor_data_item_id = item.id
                            break
        
        events_to_add: List[Dict[str, Any]] = []
        for event in ma3_events:
            try:
                time_val = float(event.get("time", 0.0))
            except (ValueError, TypeError):
                time_val = 0.0
            idx_val = event.get("idx") or event.get("no")
            try:
                idx_int = int(idx_val) if idx_val is not None else 0
            except (ValueError, TypeError):
                idx_int = 0
            
            raw_duration = event.get("duration", 0.0) or 0.0
            evt_metadata = {
                "_synced_from_ma3": True,
                "_ma3_track_coord": coord,
                "_ma3_idx": idx_int,
                "_show_manager_block_id": self._show_manager_block_id,
            }
            norm_duration, evt_metadata = _ma3_event_defaults(raw_duration, evt_metadata)
            events_to_add.append({
                "time": time_val,
                "duration": norm_duration,
                "classification": layer_name,
                "metadata": evt_metadata,
            })
        
        added = api.apply_layer_snapshot(
            layer_name=layer_name,
            events=events_to_add,
            source=source,
            update_source="ma3_sync",
            data_item_id=entity.editor_data_item_id
        )
        Log.info(
            f"SyncSystemManager: EditorAPI applied MA3 events for '{layer_name}' "
            f"(added={added}, source={source})"
        )

    def _get_or_create_ma3_timecode_data_item(
        self,
        entity: SyncLayerEntity,
        coord: str
    ) -> Optional[str]:
        """Get or create the shared EventDataItem for a MA3 timecode."""
        from src.shared.domain.entities import EventDataItem
        from src.application.commands.data_item_commands import CreateEventDataItemCommand

        editor_block_id = entity.editor_block_id or self._get_editor_block_id()
        if not editor_block_id or not self._facade.data_item_repo:
            return None

        timecode_no = entity.ma3_timecode_no
        if not timecode_no:
            parts = self._parse_ma3_coord(coord)
            if parts:
                timecode_no = parts.get("timecode_no")

        if not timecode_no:
            return None

        item_name = f"MA3_TC_{timecode_no}"

        items = self._facade.data_item_repo.list_by_block(editor_block_id)
        for item in items:
            if isinstance(item, EventDataItem):
                if item.name == item_name:
                    return item.id
                meta_tc = (item.metadata or {}).get("_ma3_timecode_no")
                meta_source = (item.metadata or {}).get("source")
                if meta_tc == timecode_no and meta_source in {"ma3", "ma3_sync"}:
                    return item.id

        create_cmd = CreateEventDataItemCommand(
            facade=self._facade,
            block_id=editor_block_id,
            name=item_name,
            metadata={
                "source": "ma3",
                "_synced_from_ma3": True,
                "_ma3_timecode_no": timecode_no,
                "group_id": f"tc_{timecode_no}",
                "group_name": f"TC {timecode_no}",
                "_show_manager_block_id": self._show_manager_block_id,
            }
        )
        self._facade.command_bus.execute(create_cmd)
        return create_cmd.created_data_item_id
    
    def _get_or_create_sync_data_item(self, entity: SyncLayerEntity) -> Optional[str]:
        """Get the EventDataItem for synced layer events.
        
        For Editor-sourced layers, we use the ORIGINAL data item that contains
        the Editor layer's events. This ensures MA3 changes update the existing
        layer rather than creating a new one.
        """
        from src.shared.domain.entities import EventDataItem
        
        # If we already have a stored data_item_id (from original layer), use it
        if entity.editor_data_item_id:
            if self._facade.data_item_repo:
                existing = self._facade.data_item_repo.get(entity.editor_data_item_id)
                if existing:
                    Log.debug(f"SyncSystemManager: Using stored data item: {entity.editor_data_item_id}")
                    return entity.editor_data_item_id
        
        # For Editor-sourced entities, try to find original data item by group_name
        if entity.source == SyncSource.EDITOR and entity.group_name:
            data_item_id = self._find_data_item_for_layer(entity.group_name)
            if data_item_id:
                Log.info(f"SyncSystemManager: Found original data item by group_name: {data_item_id}")
                return data_item_id
        
        # For MA3-sourced entities, we may need to create a new data item
        # (This is the case when syncing from MA3 to Editor, not Editor to MA3)
        editor_block_id = entity.editor_block_id or self._get_editor_block_id()
        if not editor_block_id or not self._facade.data_item_repo:
            return None
        
        # Check if a sync data item already exists for this MA3 track
        items = self._facade.data_item_repo.list_by_block(editor_block_id)
        for item in items:
            if isinstance(item, EventDataItem):
                source = item.metadata.get("source", "")
                ma3_coord = item.metadata.get("_ma3_track_coord", "")
                if source == "ma3" and ma3_coord == entity.ma3_coord:
                    return item.id
        
        # Only create new data item for MA3-sourced entities (not Editor-sourced)
        if entity.source == SyncSource.MA3:
            from src.application.commands.data_item_commands import CreateEventDataItemCommand
            
            editor_result = self._facade.describe_block(editor_block_id)
            editor_name = editor_result.data.name if editor_result.success and editor_result.data else "Editor"
            
            create_cmd = CreateEventDataItemCommand(
                facade=self._facade,
                block_id=editor_block_id,
                name=f"{editor_name}_ma3_sync_{entity.name}",
                metadata={
                    "source": "ma3",
                    "_synced_from_ma3": True,
                    "_ma3_track_coord": entity.ma3_coord,
                    "_show_manager_block_id": self._show_manager_block_id,
                }
            )
            self._facade.command_bus.execute(create_cmd)
            return create_cmd.created_data_item_id
        
        Log.warning(f"SyncSystemManager: Could not find data item for Editor layer: {entity.editor_layer_id}")
        return None
    
    def _replace_data_item_events(self, data_item_id: str, events: List[Dict[str, Any]], layer_name: str) -> None:
        """Replace events for a SPECIFIC LAYER in a data item (preserves other layers).
        
        IMPORTANT: Data items can contain multiple layers (e.g., kick, clap, snare, hihat).
        This method only replaces events for the target layer, preserving all other layers.
        """
        from src.shared.domain.entities import EventDataItem, EventLayer
        
        if not self._facade.data_item_repo:
            return
        
        data_item = self._facade.data_item_repo.get(data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            Log.warning(f"SyncSystemManager: Data item not found: {data_item_id}")
            return
        
        try:
            # Remove ONLY the target layer (preserves other layers like kick, snare, hihat)
            existing_layer = data_item.get_layer_by_name(layer_name)
            if existing_layer:
                data_item.remove_layer(layer_name)
                Log.debug(f"SyncSystemManager: Removed existing layer '{layer_name}' before replacement")
            
            # Add new events for this layer
            for event in events:
                data_item.add_event(
                    time=event.get("time", 0.0),
                    duration=event.get("duration", 0.0),
                    classification=event.get("classification", layer_name),
                    metadata=event.get("metadata", {}),
                    layer_name=layer_name,
                )
            
            # Ensure the EventLayer exists even if there are no events
            # This is important for 0-event layers to appear in the timeline
            if not events:
                existing_layer = data_item.get_layer_by_name(layer_name)
                if not existing_layer:
                    empty_layer = EventLayer(name=layer_name, events=[])
                    data_item.add_layer(empty_layer)
                    Log.debug(f"SyncSystemManager: Created empty EventLayer '{layer_name}' for 0-event layer")
            
            # Save the updated data item
            self._facade.data_item_repo.update(data_item)
        except Exception as e:
            Log.error(f"SyncSystemManager: Exception in _replace_data_item_events: {e}")
            raise
        
        Log.info(f"SyncSystemManager: Replaced {len(events)} events for layer '{layer_name}' in data item {data_item_id}")

    def _apply_ma3_diff_to_data_item(
        self,
        data_item_id: str,
        layer_name: str,
        ma3_events: List[Dict[str, Any]],
        coord: str
    ) -> Dict[str, int]:
        """Apply MA3→Editor in-place diff using ma3_idx as the primary key."""
        from src.shared.domain.entities import EventDataItem, EventLayer

        result = {"added": 0, "updated": 0, "deleted": 0}

        if not self._facade.data_item_repo:
            return result

        data_item = self._facade.data_item_repo.get(data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            Log.warning(f"SyncSystemManager: Data item not found: {data_item_id}")
            return result

        layer = data_item.get_layer_by_name(layer_name)
        if not layer:
            layer = EventLayer(name=layer_name, events=[])
            data_item.add_layer(layer)

        # Build MA3 index map
        ma3_by_idx: Dict[int, Dict[str, Any]] = {}
        for i, event in enumerate(ma3_events):
            idx_val = event.get("idx") or event.get("no")
            if idx_val is None:
                continue
            try:
                idx_int = int(idx_val)
            except (ValueError, TypeError):
                continue
            ma3_by_idx[idx_int] = event

        # Build Editor index map (by metadata._ma3_idx)
        editor_by_idx: Dict[int, Any] = {}
        editor_no_idx: List[Any] = []
        for evt in list(layer.events):
            metadata = getattr(evt, "metadata", {}) or {}
            ma3_idx = metadata.get("_ma3_idx")
            if ma3_idx is None:
                editor_no_idx.append(evt)
                continue
            try:
                idx_int = int(ma3_idx)
            except (ValueError, TypeError):
                editor_no_idx.append(evt)
                continue
            editor_by_idx[idx_int] = evt

        time_tolerance = 0.001
        matched_no_idx = set()

        for idx_int, ma3_event in ma3_by_idx.items():
            ma3_time_raw = ma3_event.get("time", 0.0)
            try:
                ma3_time = float(ma3_time_raw) if ma3_time_raw is not None else 0.0
            except (ValueError, TypeError):
                ma3_time = 0.0

            existing = editor_by_idx.get(idx_int)
            if not existing:
                # Fallback: match by time if no ma3_idx on editor event
                for candidate in editor_no_idx:
                    if candidate in matched_no_idx:
                        continue
                    try:
                        candidate_time = float(getattr(candidate, "time", 0.0))
                    except (ValueError, TypeError):
                        candidate_time = 0.0
                    if abs(candidate_time - ma3_time) <= time_tolerance:
                        existing = candidate
                        matched_no_idx.add(candidate)
                        break

            if existing:
                updated = False
                if abs(getattr(existing, "time", 0.0) - ma3_time) > time_tolerance:
                    existing.time = ma3_time
                    updated = True

                new_metadata = dict(getattr(existing, "metadata", {}) or {})
                new_metadata.update({
                    "_synced_from_ma3": True,
                    "_ma3_track_coord": coord,
                    "_ma3_idx": idx_int,
                    "_show_manager_block_id": self._show_manager_block_id,
                })
                if new_metadata != getattr(existing, "metadata", {}):
                    existing.metadata = new_metadata
                    updated = True

                if updated:
                    result["updated"] += 1
            else:
                raw_dur = ma3_event.get("duration", 0.0) or 0.0
                add_metadata = {
                    "_synced_from_ma3": True,
                    "_ma3_track_coord": coord,
                    "_ma3_idx": idx_int,
                    "_show_manager_block_id": self._show_manager_block_id,
                }
                norm_dur, add_metadata = _ma3_event_defaults(raw_dur, add_metadata)
                data_item.add_event(
                    time=ma3_time,
                    classification=layer_name,
                    duration=norm_dur,
                    metadata=add_metadata,
                    layer_name=layer_name,
                )
                result["added"] += 1

        # Delete editor events that no longer exist in MA3 (by ma3_idx)
        to_delete = []
        for evt in list(layer.events):
            metadata = getattr(evt, "metadata", {}) or {}
            ma3_idx = metadata.get("_ma3_idx")
            if ma3_idx is None:
                continue
            try:
                idx_int = int(ma3_idx)
            except (ValueError, TypeError):
                continue
            if idx_int not in ma3_by_idx:
                to_delete.append(evt)

        for evt in to_delete:
            layer.remove_event(evt)
        result["deleted"] += len(to_delete)

        # Update event count and persist
        data_item.event_count = sum(len(l.events) for l in data_item.get_layers())
        self._facade.data_item_repo.update(data_item)

        return result
    
    def _refresh_editor_timeline(self, editor_block_id: Optional[str], layer_name: Optional[str] = None) -> None:
        """Trigger a timeline refresh in the Editor panel."""
        if not editor_block_id:
            return
        
        # Emit BlockUpdated event to trigger UI refresh
        # IMPORTANT: EditorPanel._on_block_updated expects data["id"] to contain the block ID
        # Include layer_name so UI can refresh only the affected layer
        from src.application.events.events import BlockUpdated
        if hasattr(self._facade, 'event_bus') and self._facade.event_bus:
            event_data = {
                "id": editor_block_id,
                "events_updated": True,
                "source": "ma3_sync"  # This will be skipped by _on_editor_block_updated
            }
            # Include layer_names if provided (for UI refresh, not sync)
            if layer_name:
                event_data["layer_names"] = [layer_name]
            self._facade.event_bus.publish(BlockUpdated(
                data=event_data
            ))
    
    def _schedule_push_editor_to_ma3(self, entity: SyncLayerEntity) -> None:
        """
        Schedule a push from Editor to MA3 with coalescing.
        
        Similar to _schedule_push_ma3_to_editor, this waits for drag operations
        to complete before pushing to MA3.
        """
        from PyQt6.QtCore import QTimer
        import time
        
        COALESCE_DELAY_MS = 300  # Wait 300ms after last change before pushing
        
        
        coord = entity.ma3_coord
        if not coord:
            return
        
        # Initialize coalescing state if needed (reuse existing dicts with different prefix)
        if not hasattr(self, "_coalesce_ez_timers"):
            self._coalesce_ez_timers: Dict[str, QTimer] = {}
        if not hasattr(self, "_coalesce_ez_entities"):
            self._coalesce_ez_entities: Dict[str, SyncLayerEntity] = {}
        if not hasattr(self, "_coalesce_ez_schedule_time"):
            self._coalesce_ez_schedule_time: Dict[str, float] = {}
        
        # Store the entity and current schedule time
        self._coalesce_ez_entities[coord] = entity
        schedule_time = time.time()
        self._coalesce_ez_schedule_time[coord] = schedule_time
        
        # If there's already a timer running for this coord, let it handle
        if coord in self._coalesce_ez_timers and self._coalesce_ez_timers[coord].isActive():
            Log.debug(f"SyncSystemManager: EZ→MA3 timer already running for {coord}, updating entity")
            return
        
        def do_push():
            # Check if we were rescheduled while waiting
            current_schedule = self._coalesce_ez_schedule_time.get(coord, 0)
            elapsed_ms = (time.time() - current_schedule) * 1000
            
            if elapsed_ms < COALESCE_DELAY_MS - 10:  # 10ms tolerance
                # We were rescheduled, wait more
                remaining = int(COALESCE_DELAY_MS - elapsed_ms)
                Log.debug(f"SyncSystemManager: EZ→MA3 rescheduled, waiting {remaining}ms more for {coord}")
                self._coalesce_ez_timers[coord].start(remaining)
                return
            
            # Time to push
            entity_to_push = self._coalesce_ez_entities.get(coord)
            if entity_to_push:
                try:
                    self._push_editor_to_ma3(entity_to_push)
                    entity_to_push.mark_synced()
                    self._save_to_settings()
                    self.entity_updated.emit(entity_to_push.id)
                    Log.info(f"SyncSystemManager: Coalesced EZ→MA3 push complete for {coord}")
                except Exception as e:
                    Log.error(f"SyncSystemManager: Coalesced EZ→MA3 push failed: {e}")
            
            # Cleanup
            self._coalesce_ez_timers.pop(coord, None)
            self._coalesce_ez_entities.pop(coord, None)
            self._coalesce_ez_schedule_time.pop(coord, None)
        
        # Create and start timer
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(do_push)
        self._coalesce_ez_timers[coord] = timer
        timer.start(COALESCE_DELAY_MS)
        
        Log.debug(f"SyncSystemManager: Scheduled EZ→MA3 push for {coord} in {COALESCE_DELAY_MS}ms")
    
    def _push_editor_to_ma3(self, entity: SyncLayerEntity) -> None:
        """Push Editor events to MA3 track by clearing and rebuilding.
        
        This approach clears all events in MA3 and re-adds all Editor events.
        This is simpler and more reliable than reconciliation, avoiding:
        - Event diff calculation issues
        - Stale cache problems
        - Index shifting issues
        - Race conditions
        """
        if not entity.editor_layer_id or not entity.ma3_coord:
            Log.warning("SyncSystemManager: Cannot push - missing editor_layer_id or ma3_coord")
            return
        
        if not self.ma3_comm_service:
            Log.warning("SyncSystemManager: Cannot push - no MA3 comm service")
            return
        
        # Get events from Editor
        editor_events = self._get_editor_events(entity.editor_layer_id, entity.editor_block_id)
        if not editor_events:
            Log.warning(f"SyncSystemManager: No Editor events to push for {entity.editor_layer_id}")
            return
        
        
        # Parse coord
        parts = self._parse_ma3_coord(entity.ma3_coord)
        if not parts:
            Log.error(f"SyncSystemManager: Invalid MA3 coord: {entity.ma3_coord}")
            return
        
        tc_no = parts["timecode_no"]
        tg_no = parts["track_group"]
        tr_no = parts["track"]
        
        # Record Editor->MA3 push time for empty-callback ignore window
        import time
        self._last_editor_push_time[entity.ma3_coord] = time.time()
        
        # Clear all events from MA3 track first
        Log.info(f"SyncSystemManager: Clearing all events from MA3 track {tc_no}.{tg_no}.{tr_no}")
        clear_success = self._clear_ma3_track(tc_no, tg_no, tr_no)
        if not clear_success:
            Log.error(f"SyncSystemManager: Failed to clear MA3 track {tc_no}.{tg_no}.{tr_no}")
            return
        
        # Add all Editor events to MA3
        result = self._add_all_editor_events_to_ma3(
            editor_events=editor_events,
            tc_no=tc_no,
            tg_no=tg_no,
            tr_no=tr_no
        )
        
        
        # Update local cache directly with what we sent (don't request fresh events)
        # This keeps Editor state static - we know what we sent, so update cache to match
        # Only update Editor when MA3 changes independently (not from our push)
        if result.get("added", 0) > 0:
            # Update cache directly with Editor events (what we just sent to MA3)
            # Convert Editor events to MA3EventInfo format for cache
            cache_events = []
            for i, event in enumerate(editor_events):
                # Extract time and name from Editor event
                if isinstance(event, dict):
                    time_val = event.get("time", 0.0)
                    name_val = event.get("name") or event.get("classification", "")
                else:
                    time_val = getattr(event, 'time', 0.0)
                    name_val = getattr(event, 'classification', None) or getattr(event, 'name', None) or ""
                
                # Ensure time is float
                try:
                    time_float = float(time_val) if time_val is not None else 0.0
                except (ValueError, TypeError):
                    time_float = 0.0
                
                # Ensure name is string
                if name_val is None:
                    name_str = ""
                elif isinstance(name_val, str):
                    name_str = name_val
                elif isinstance(name_val, bytes):
                    name_str = name_val.decode('utf-8', errors='replace')
                else:
                    name_str = str(name_val)
                
                cache_events.append(MA3EventInfo(
                    time=time_float,
                    name=name_str,
                    cmd="",  # We don't store cmd in Editor events
                    idx=i + 1,  # MA3 uses 1-based indexing
                    tc=tc_no,
                    tg=tg_no,
                    track=tr_no,
                ))
            
            # Update cache directly (don't trigger MA3->Editor sync)
            self._ma3_track_events[entity.ma3_coord] = cache_events
            if entity.ma3_coord in self._ma3_tracks:
                self._ma3_tracks[entity.ma3_coord].event_count = len(cache_events)
            
            Log.debug(f"SyncSystemManager: Updated cache directly for {entity.ma3_coord} with {len(cache_events)} events (skipped fresh request to keep Editor static)")
        
    
    def _add_all_editor_events_to_ma3(
        self,
        editor_events: List[Any],
        tc_no: int,
        tg_no: int,
        tr_no: int,
    ) -> Dict[str, int]:
        """
        Add all Editor events to MA3 track.
        
        This is used after clearing the track - we just add all events fresh.
        
        Args:
            editor_events: List of events from Editor (Event objects or dicts)
            tc_no, tg_no, tr_no: MA3 track coordinates
            
        Returns:
            Dict with counts: {"added": N, "failed": M}
        """
        result = {"added": 0, "failed": 0}
        
        
        # Process each Editor event and add to MA3
        for i, event in enumerate(editor_events):
            # Extract event properties
            if isinstance(event, dict):
                ez_time_raw = event.get("time", 0.0)
                metadata = event.get("metadata", {})
                # Extract name - handle various types that might be returned
                name_val = event.get("name")
                classification_val = event.get("classification")
                ez_name = name_val if name_val else (classification_val if classification_val else "")
            else:
                ez_time_raw = getattr(event, 'time', 0.0)
                metadata = getattr(event, 'metadata', {}) or {}
                # Extract name - handle various types that might be returned
                classification_val = getattr(event, 'classification', None)
                name_val = getattr(event, 'name', None)
                ez_name = classification_val if classification_val else (name_val if name_val else "")
            
            # Validate and convert ez_time to float
            try:
                ez_time = float(ez_time_raw) if ez_time_raw is not None else 0.0
            except (ValueError, TypeError):
                Log.error(f"SyncSystemManager: Invalid ez_time value: {ez_time_raw} (type: {type(ez_time_raw)}) - skipping event")
                result["failed"] += 1
                continue
            
            # Ensure ez_name is a string
            try:
                if ez_name is None:
                    ez_name_str = ""
                elif isinstance(ez_name, str):
                    ez_name_str = ez_name
                elif isinstance(ez_name, bytes):
                    ez_name_str = ez_name.decode('utf-8', errors='replace')
                else:
                    ez_name_str = str(ez_name)
                
                if not isinstance(ez_name_str, str):
                    Log.error(f"SyncSystemManager: ez_name_str is not a string after conversion: {ez_name_str} (type: {type(ez_name_str)})")
                    ez_name_str = ""
                
                # Escape quotes in name for Lua string
                ez_name_escaped = ez_name_str.replace('"', '\\"')
                
                # Add event to MA3
                self._send_lua_command_with_target(
                    f"EZ.AddEvent({tc_no}, {tg_no}, {tr_no}, {ez_time}, \"{ez_name_escaped}\")"
                )
                result["added"] += 1
                if i < 3:  # Log first few for debugging
                    Log.debug(f"SyncSystemManager: Added event {i+1}/{len(editor_events)} at {ez_time:.3f}s with name '{ez_name_str}'")
            except Exception as e:
                Log.error(f"SyncSystemManager: Error adding event to MA3: {e}")
                Log.error(f"SyncSystemManager: Event details - ez_time={ez_time} (type: {type(ez_time)}), ez_name={ez_name} (type: {type(ez_name)})")
                import traceback
                Log.debug(traceback.format_exc())
                result["failed"] += 1
                continue
        
        Log.info(f"SyncSystemManager: Added {result['added']} events to MA3 track {tc_no}.{tg_no}.{tr_no} (failed: {result['failed']})")
        return result
    
    def _reconcile_events_to_ma3(
        self,
        editor_events: List[Any],
        ma3_events: List[Dict[str, Any]],
        tc_no: int,
        tg_no: int,
        tr_no: int,
    ) -> Dict[str, int]:
        """
        DEPRECATED: Use _add_all_editor_events_to_ma3 after clearing track instead.
        
        This method is kept for backward compatibility but should not be used.
        """
        """
        Reconcile Editor events to MA3 track.
        
        Computes the minimal set of changes needed and applies them:
        - MOVE: Events that exist in both but have different times
        - ADD: Events in Editor but not in MA3
        - DELETE: Events in MA3 but not in Editor (optional, disabled by default)
        
        Args:
            editor_events: List of events from Editor (Event objects or dicts)
            ma3_events: List of events from MA3 (dicts with idx, time, name)
            tc_no, tg_no, tr_no: MA3 track coordinates
            
        Returns:
            Dict with counts: {"moved": N, "added": N, "deleted": N}
        """
        
        result = {"moved": 0, "added": 0, "deleted": 0}
        
        # Build lookup of MA3 events by idx
        ma3_by_idx: Dict[int, Dict[str, Any]] = {}
        for e in ma3_events:
            idx = e.get("idx") or e.get("no")
            if idx:
                ma3_by_idx[idx] = e
        
        
        # Process each Editor event
        for i, event in enumerate(editor_events):
            # Extract event properties
            if isinstance(event, dict):
                ez_time_raw = event.get("time", 0.0)
                metadata = event.get("metadata", {})
                # Extract name - handle various types that might be returned
                name_val = event.get("name")
                classification_val = event.get("classification")
                ez_name = name_val if name_val else (classification_val if classification_val else "")
            else:
                ez_time_raw = getattr(event, 'time', 0.0)
                metadata = getattr(event, 'metadata', {}) or {}
                # Extract name - handle various types that might be returned
                classification_val = getattr(event, 'classification', None)
                name_val = getattr(event, 'name', None)
                ez_name = classification_val if classification_val else (name_val if name_val else "")
            
            # Validate and convert ez_time to float early (used in multiple places)
            try:
                ez_time = float(ez_time_raw) if ez_time_raw is not None else 0.0
            except (ValueError, TypeError):
                Log.error(f"SyncSystemManager: Invalid ez_time value: {ez_time_raw} (type: {type(ez_time_raw)}) - using 0.0")
                ez_time = 0.0
            
            
            ma3_idx = metadata.get("_ma3_idx")
            
            
            if ma3_idx and ma3_idx in ma3_by_idx:
                # Event exists in both - check if time changed
                ma3_event = ma3_by_idx[ma3_idx]
                ma3_time = ma3_event.get("time", 0.0)
                
                # Compare times (with small tolerance for floating point)
                # ez_time is already validated and converted to float above
                if abs(ez_time - ma3_time) > 0.001:
                    # Time changed - update in MA3
                    self._send_lua_command_with_target(
                        f"EZ.UpdateEvent({tc_no}, {tg_no}, {tr_no}, {ma3_idx}, {ez_time})"
                    )
                    result["moved"] += 1
                    Log.debug(f"Reconcile: Moved event {ma3_idx} from {ma3_time:.3f}s to {ez_time:.3f}s")
                
                # Remove from lookup (processed)
                del ma3_by_idx[ma3_idx]
            else:
                # Event only in Editor - add to MA3
                # Ensure ez_name is a string (could be None or other type)
                # Handle various edge cases: None, empty string, non-string types
                try:
                    if ez_name is None:
                        ez_name_str = ""
                    elif isinstance(ez_name, str):
                        ez_name_str = ez_name
                    elif isinstance(ez_name, bytes):
                        # Handle bytes objects
                        ez_name_str = ez_name.decode('utf-8', errors='replace')
                    else:
                        # Convert to string, handling any edge cases
                        ez_name_str = str(ez_name)
                    
                    # Ensure ez_name_str is actually a string before calling .replace()
                    if not isinstance(ez_name_str, str):
                        Log.error(f"SyncSystemManager: ez_name_str is not a string after conversion: {ez_name_str} (type: {type(ez_name_str)})")
                        ez_name_str = ""
                    
                    # Escape quotes in name for Lua string
                    ez_name_escaped = ez_name_str.replace('"', '\\"')
                    
                    # ez_time is already validated and converted to float above
                    # Use it directly here
                    self._send_lua_command_with_target(
                        f"EZ.AddEvent({tc_no}, {tg_no}, {tr_no}, {ez_time}, \"{ez_name_escaped}\")"
                    )
                    result["added"] += 1
                    Log.debug(f"Reconcile: Added new event at {ez_time:.3f}s with name '{ez_name_str}'")
                except Exception as e:
                    # Log the error with full context for debugging
                    Log.error(f"SyncSystemManager: Error adding event to MA3: {e}")
                    Log.error(f"SyncSystemManager: Event details - ez_time={ez_time} (type: {type(ez_time)}), ez_name={ez_name} (type: {type(ez_name)}), ez_name_str={ez_name_str if 'ez_name_str' in locals() else 'N/A'}, event={event}")
                    import traceback
                    Log.debug(traceback.format_exc())
                    # Don't fail the entire sync - continue with next event
                    continue
        
        # Events remaining in ma3_by_idx are only in MA3 (deleted from Editor)
        # CRITICAL: Delete from highest index to lowest to avoid index shifting issues
        # When you delete event at index 2, event at index 4 becomes index 3, etc.
        sorted_indices = sorted(ma3_by_idx.keys(), reverse=True)
        
        
        for idx in sorted_indices:
            ma3_event = ma3_by_idx[idx]
            
            # Ensure idx is an integer before using in f-string
            try:
                idx_int = int(idx) if idx is not None else 0
            except (ValueError, TypeError):
                Log.error(f"SyncSystemManager: Invalid idx for deletion: {idx} (type: {type(idx)})")
                continue
            
            try:
                # Build Lua command string - ensure all values are properly formatted
                lua_cmd = f"EZ.DeleteEvent({tc_no}, {tg_no}, {tr_no}, {idx_int})"
                self._send_lua_command_with_target(lua_cmd)
                result["deleted"] += 1
                Log.debug(f"Reconcile: Deleted MA3 event idx={idx_int}")
            except Exception as e:
                Log.error(f"SyncSystemManager: Error deleting MA3 event idx={idx_int}: {e}")
                import traceback
                Log.debug(traceback.format_exc())
                # Continue with next deletion - don't fail entire sync
                continue
        
        
        Log.info(f"SyncSystemManager: Reconciled to MA3 - moved={result['moved']}, added={result['added']}, deleted={result['deleted']}")
        return result
    
    def _merge_events(self, entity: SyncLayerEntity) -> None:
        """Merge events from both sides."""
        if not entity.editor_layer_id or not entity.ma3_coord:
            return
        
        # Get events from both sides
        editor_events = self._get_editor_events(entity.editor_layer_id, entity.editor_block_id)
        ma3_events = self._get_ma3_events(entity.ma3_coord)
        
        # Use SyncLayerManager to merge
        merged = SyncLayerManager.merge_events(editor_events, ma3_events)
        
        # Push merged to both sides
        Log.info(f"SyncSystemManager: Merged to {len(merged)} events")
    
    def _compare_entity(self, entity: SyncLayerEntity) -> Optional[SyncLayerComparison]:
        """Compare entity's events between MA3 and Editor."""
        if not entity.editor_layer_id or not entity.ma3_coord:
            return None
        
        editor_events = self._get_editor_events(entity.editor_layer_id, entity.editor_block_id)
        ma3_events = self._get_ma3_events(entity.ma3_coord)
        
        return SyncLayerManager.compare_events(editor_events, ma3_events)
    
    # =========================================================================
    # Private: MA3 Operations
    # =========================================================================
    
    def _hook_ma3_track(self, coord: str) -> None:
        """Hook MA3 track for change notifications."""
        if coord in self._hooked_tracks:
            return
        
        # Parse coord to get timecode, track group, and track numbers
        parts = self._parse_ma3_coord(coord)
        if not parts:
            Log.error(f"SyncSystemManager: Cannot hook - invalid coord: {coord}")
            return
        
        tc_no = parts["timecode_no"]
        tg_no = parts["track_group"]
        tr_no = parts["track"]
        
        # Send hook command to MA3
        if self.ma3_comm_service:
            success = self.ma3_comm_service.hook_cmdsubtrack(tc_no, tg_no, tr_no)
            if success:
                self._hooked_tracks[coord] = lambda changes: self._on_ma3_track_changed(coord, changes)
                Log.info(f"SyncSystemManager: Hooked MA3 track {coord}")
                self._hook_ma3_track_group_changes(tc_no, tg_no)
            else:
                Log.error(f"SyncSystemManager: Failed to hook MA3 track {coord}")
        else:
            Log.warning(f"SyncSystemManager: Cannot hook - no MA3 comm service")
    
    def _unhook_ma3_track(self, coord: str) -> None:
        """Unhook MA3 track."""
        if coord not in self._hooked_tracks:
            return
        
        # Parse coord to get timecode, track group, and track numbers
        parts = self._parse_ma3_coord(coord)
        if parts and self.ma3_comm_service:
            tc_no = parts["timecode_no"]
            tg_no = parts["track_group"]
            tr_no = parts["track"]
            self.ma3_comm_service.unhook_track(tc_no, tg_no, tr_no)
        
        del self._hooked_tracks[coord]
        Log.info(f"SyncSystemManager: Unhooked MA3 track {coord}")
    
    def unhook_all_ma3_tracks(self) -> None:
        """
        Unhook all MA3 tracks and track groups.
        
        Called on application shutdown, project change, or panel close
        to ensure all hooks are properly cleaned up.
        """
        if not self._hooked_tracks and not self._hooked_track_groups:
            return  # Nothing to unhook
        
        hook_count = len(self._hooked_tracks)
        group_count = len(self._hooked_track_groups)
        
        Log.info(f"SyncSystemManager: Unhooking all MA3 tracks ({hook_count} tracks, {group_count} groups)")
        
        # Use unhook_all command if available (more efficient than individual unhooks)
        if self.ma3_comm_service:
            try:
                self.ma3_comm_service.unhook_all()
                Log.info("SyncSystemManager: Sent unhook_all command to MA3")
            except Exception as e:
                Log.warning(f"SyncSystemManager: Failed to send unhook_all command: {e}")
                # Fall back to individual unhooks
                for coord in list(self._hooked_tracks.keys()):
                    try:
                        self._unhook_ma3_track(coord)
                    except Exception as e2:
                        Log.warning(f"SyncSystemManager: Failed to unhook {coord}: {e2}")
        else:
            # No comm service, just clear local tracking
            Log.warning("SyncSystemManager: No MA3 comm service available, clearing local hook tracking only")
        
        # Unhook all track groups individually
        if self.ma3_comm_service:
            for key in list(self._hooked_track_groups.keys()):
                try:
                    # Parse key format: "tc.tg"
                    parts = key.split(".")
                    if len(parts) == 2:
                        tc_no = int(parts[0])
                        tg_no = int(parts[1])
                        self.ma3_comm_service.unhook_track_group_changes(tc_no, tg_no)
                        Log.debug(f"SyncSystemManager: Unhooked track group TC{tc_no}.TG{tg_no}")
                except Exception as e:
                    Log.warning(f"SyncSystemManager: Failed to unhook track group {key}: {e}")
        
        # Clear local tracking
        self._hooked_tracks.clear()
        self._hooked_track_groups.clear()
        
        Log.info(f"SyncSystemManager: Completed unhooking all MA3 tracks")

    def _hook_ma3_track_group_changes(self, tc_no: int, tg_no: int) -> None:
        """Hook MA3 track group changes once per group."""
        if not self.ma3_comm_service:
            return
        key = f"{tc_no}.{tg_no}"
        if self._hooked_track_groups.get(key):
            return
        if self.ma3_comm_service.hook_track_group_changes(tc_no, tg_no):
            self._hooked_track_groups[key] = True
            Log.info(f"SyncSystemManager: Hooked MA3 track group TC{tc_no}.TG{tg_no}")
        else:
            Log.warning(f"SyncSystemManager: Failed to hook track group TC{tc_no}.TG{tg_no}")

    def _on_trackgroup_changed(self, message) -> None:
        """Handle MA3 track group change notifications."""
        try:
            tc_no = int(message.data.get("tc", 0))
            tg_no = int(message.data.get("tg", 0))
        except Exception:
            return
        if tc_no <= 0 or tg_no <= 0:
            return
        Log.info(f"SyncSystemManager: Track group changed TC{tc_no}.TG{tg_no}, refreshing tracks")
        self._refresh_ma3_tracks_sync(tc_no, tg_no)
        
        # Detect deleted or shifted tracks for synced layers in this group
        updated_any = False
        for entity in self._synced_layers.values():
            if entity.ma3_timecode_no != tc_no or entity.ma3_track_group != tg_no:
                continue
            if not entity.ma3_coord:
                continue
            if entity.ma3_coord in self._ma3_tracks:
                continue
            
            # Try to remap by name in case the track index shifted
            target_name = normalize_track_name(entity.name)
            if target_name:
                for track in self._ma3_tracks.values():
                    if track.timecode_no != tc_no or track.track_group_no != tg_no:
                        continue
                    if normalize_track_name(track.name) == target_name:
                        self._remap_entity_ma3_coord(entity, track.track_no)
                        updated_any = True
                        break
                if entity.ma3_coord in self._ma3_tracks:
                    continue
            
            # Track truly missing: notify and mark error
            entity.mark_error("MA3 track deleted")
            self.error_occurred.emit(entity.id, "MA3 track deleted")
            updated_any = True
        
        if updated_any:
            self._save_to_settings()
            self.entities_changed.emit()
    
    def is_track_hooked(self, coord: str) -> bool:
        """Check if a track is currently hooked."""
        return coord in self._hooked_tracks
    
    def _rehook_ma3_track(self, coord: str) -> bool:
        """
        Rehook MA3 track by unhooking first, then hooking.
        
        This forces a fresh hook and causes MA3 to re-send current events.
        Used during resync to ensure hook is active and data is current.
        
        Args:
            coord: MA3 track coordinate (e.g., "tc101_tg1_tr2")
            
        Returns:
            True if rehook command was sent successfully
        """
        parts = self._parse_ma3_coord(coord)
        if not parts:
            Log.error(f"SyncSystemManager: Cannot rehook - invalid coord: {coord}")
            return False
        
        tc_no = parts["timecode_no"]
        tg_no = parts["track_group"]
        tr_no = parts["track"]
        
        if not self.ma3_comm_service:
            Log.warning(f"SyncSystemManager: Cannot rehook - no MA3 comm service")
            return False
        
        success = self.ma3_comm_service.rehook_cmdsubtrack(tc_no, tg_no, tr_no)
        if success:
            # Update local hook tracking (rehook re-establishes the hook)
            self._hooked_tracks[coord] = lambda changes: self._on_ma3_track_changed(coord, changes)
            Log.info(f"SyncSystemManager: Rehook command sent for MA3 track {coord}")
        else:
            Log.error(f"SyncSystemManager: Failed to send rehook for MA3 track {coord}")
        
        return success
    
    def _request_ma3_events(self, coord: str, request_id: Optional[int] = None) -> bool:
        """Request events from MA3 for a track."""
        if not self.ma3_comm_service:
            Log.warning("SyncSystemManager: Cannot request events - no MA3 comm service")
            return False
        
        parts = self._parse_ma3_coord(coord)
        if not parts:
            Log.error(f"SyncSystemManager: Invalid coord for event request: {coord}")
            return False
        
        try:
            result = self.ma3_comm_service.get_events(
                parts["timecode_no"],
                parts["track_group"],
                parts["track"],
                request_id=request_id
            )
            Log.info(f"SyncSystemManager: Requested events for {coord}")
            return bool(result)
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to request events: {e}")
            return False
    
    def _on_ma3_track_changed(self, coord: str, changes: Any) -> None:
        """Handle MA3 track change notification."""
        entity = self.get_synced_layer_by_ma3_coord(coord)
        if not entity:
            return
        
        # Schedule push with coalescing (waits for drag to complete)
        self._schedule_push_ma3_to_editor(entity)
    
    # =========================================================================
    # Editor->MA3 Sync Operations
    # =========================================================================
    
    def sync_editor_to_ma3(
        self,
        editor_layer_id: str,
        target_timecode: int,
        target_track_group: int,
        target_sequence: int,
        action: str = "create_new",
        existing_track_no: Optional[int] = None,
    ) -> Optional[str]:
        """
        Sync an Editor layer to MA3.
        
        This is an ATOMIC operation - all decisions should be made by the UI layer
        before calling this method. No dialogs or prompts are shown.
        
        Actions:
        - "create_new": Create new MA3 track, push Editor events
        - "use_existing_keep_editor": Use existing track, push Editor events (overwrites MA3)
        - "use_existing_keep_ma3": Use existing track, receive MA3 events (overwrites Editor)
        
        Args:
            editor_layer_id: The Editor layer ID to sync
            target_timecode: MA3 timecode number
            target_track_group: MA3 track group number
            target_sequence: MA3 sequence number to assign
            action: One of "create_new", "use_existing_keep_editor", "use_existing_keep_ma3"
            existing_track_no: Required for use_existing_* actions
            
        Returns:
            Entity ID if successful, None if failed
        """
        Log.info(f"SyncSystemManager: sync_editor_to_ma3 layer={editor_layer_id} "
                f"tc={target_timecode} tg={target_track_group} seq={target_sequence} action={action}")
        
        # =========================================================================
        # Validation
        # =========================================================================
        
        # Check if already synced
        existing_entity = self.get_synced_layer_by_editor_layer(editor_layer_id)
        if existing_entity:
            Log.info(f"SyncSystemManager: Editor layer {editor_layer_id} already synced")
            return existing_entity.id
        
        # Get Editor layer info
        layer_info = self._get_editor_layer_info(editor_layer_id)
        if not layer_info:
            Log.error(f"SyncSystemManager: Editor layer not found: {editor_layer_id}")
            self.error_occurred.emit(editor_layer_id, "Editor layer not found")
            return None
        
        # Validate action
        valid_actions = ("create_new", "use_existing_keep_editor", "use_existing_keep_ma3")
        if action not in valid_actions:
            Log.error(f"SyncSystemManager: Invalid action '{action}'. Must be one of {valid_actions}")
            return None
        
        # Validate existing_track_no for use_existing actions
        if action.startswith("use_existing") and existing_track_no is None:
            Log.error(f"SyncSystemManager: existing_track_no required for action '{action}'")
            return None
        
        # =========================================================================
        # Gather Data
        # =========================================================================
        
        raw_name = layer_info.get("name", editor_layer_id)
        base_track_name = raw_name if raw_name.startswith("ez_") else f"ez_{raw_name}"
        editor_events = self._get_editor_events(editor_layer_id, layer_info.get("block_id"))
        
        # =========================================================================
        # Execute Action
        # =========================================================================
        
        if action == "create_new":
            # Create new track in MA3
            self._refresh_ma3_tracks_sync(target_timecode, target_track_group)
            
            # Check if name already exists, generate unique if needed
            if self._check_ma3_track_name_exists(target_timecode, target_track_group, base_track_name):
                track_name = self._generate_unique_track_name(
                    target_timecode, target_track_group, base_track_name
                )
            else:
                track_name = base_track_name
            
            track_no = self._create_ma3_track(target_timecode, target_track_group, track_name)
            if track_no is None:
                Log.error(f"SyncSystemManager: Failed to create MA3 track '{track_name}'")
                self.error_occurred.emit(editor_layer_id, "Failed to create MA3 track")
                return None
            
            coord = f"tc{target_timecode}_tg{target_track_group}_tr{track_no}"
            Log.info(f"SyncSystemManager: Created MA3 track: {coord} ('{track_name}')")
            
            # Assign sequence
            self._ensure_ma3_sequence_exists(target_sequence)
            self._assign_ma3_track_sequence(target_timecode, target_track_group, track_no, target_sequence)
            Log.info(f"SyncSystemManager: Assigned sequence {target_sequence} to track")
            
            # Allow OSC handlers to run before creating CmdSubTrack
            self._process_ma3_events_for(duration_s=0.2)
            
            # Create CmdSubTrack (required after sequence assignment)
            # For newly created tracks, TimeRange is at index 1
            time_range_index = 1
            self._create_ma3_cmd_subtrack(target_timecode, target_track_group, track_no, time_range_index)
            Log.info(f"SyncSystemManager: Created CmdSubTrack for track")
            
            # Allow MA3 to finish CmdSubTrack creation before verifying
            self._process_ma3_events_for(duration_s=0.3)
            
            # Verify track structure before adding events
            if not self._verify_track_ready(target_timecode, target_track_group, track_no):
                Log.error(f"SyncSystemManager: Track {coord} not ready after sequence assignment")
                # Don't proceed with events if track isn't ready
                self.error_occurred.emit(editor_layer_id, "Track not ready for events")
                return None
            
            # Push Editor events to MA3
            
            if editor_events:
                self._add_events_to_ma3(target_timecode, target_track_group, track_no, editor_events)
                Log.info(f"SyncSystemManager: Added {len(editor_events)} events to track")
            else:
                Log.warning(f"SyncSystemManager: No events found for layer '{editor_layer_id}' - track created but empty")
            
            # Skip initial hook response (we just pushed)
            if not hasattr(self, "_skip_initial_hook"):
                self._skip_initial_hook: Dict[str, bool] = {}
            self._skip_initial_hook[coord] = True
            
        elif action == "use_existing_keep_editor":
            # Use existing track, push Editor events (overwrites MA3)
            track_no = existing_track_no
            coord = f"tc{target_timecode}_tg{target_track_group}_tr{track_no}"
            
            # Get track name from cache
            track_info = self._ma3_tracks.get(coord)
            track_name = track_info.name if track_info else base_track_name
            
            Log.info(f"SyncSystemManager: Using existing track {coord}, pushing Editor events")
            
            # Clear MA3 track and push Editor events
            if editor_events:
                self._clear_ma3_track(target_timecode, target_track_group, track_no)
                self._add_events_to_ma3(target_timecode, target_track_group, track_no, editor_events)
                Log.info(f"SyncSystemManager: Pushed {len(editor_events)} Editor events to MA3")
            
            # Skip initial hook response (we just pushed)
            if not hasattr(self, "_skip_initial_hook"):
                self._skip_initial_hook: Dict[str, bool] = {}
            self._skip_initial_hook[coord] = True
            
        elif action == "use_existing_keep_ma3":
            # Use existing track, receive MA3 events (MA3 is authoritative)
            track_no = existing_track_no
            coord = f"tc{target_timecode}_tg{target_track_group}_tr{track_no}"
            
            # Get track name from cache
            track_info = self._ma3_tracks.get(coord)
            track_name = track_info.name if track_info else base_track_name
            
            Log.info(f"SyncSystemManager: Using existing track {coord}, MA3 events will overwrite Editor")
            # Skip initial hook response - we'll push MA3 events to Editor immediately after entity creation
            if not hasattr(self, "_skip_initial_hook"):
                self._skip_initial_hook: Dict[str, bool] = {}
            self._skip_initial_hook[coord] = True
        
        # =========================================================================
        # Create and Store Entity BEFORE Hooking
        # =========================================================================
        
        entity_id = str(uuid.uuid4())
        entity = SyncLayerEntity.from_editor_layer(
            id=entity_id,
            layer_id=editor_layer_id,
            block_id=layer_info.get("block_id", self._get_editor_block_id()),
            name=track_name,
            group_name=layer_info.get("group_name"),
            event_count=len(editor_events) if editor_events else 0,
        )
        
        entity.link_to_ma3(
            coord=coord,
            timecode_no=target_timecode,
            track_group=target_track_group,
            track=track_no,
        )
        
        entity.settings.track_group_no = target_track_group
        entity.settings.sequence_no = target_sequence
        entity.mark_synced()
        
        # Stamp EZ track ID on the MA3 track .note property for persistent identity.
        if editor_layer_id and self._ma3_connected:
            ez_track_id = f"ez:{editor_layer_id}"
            entity.ez_track_id = ez_track_id
            self._send_lua_command_with_target(
                f'EZ.SetTrackNote({target_timecode}, {target_track_group}, {track_no}, "{ez_track_id}")'
            )
            Log.info(f"SyncSystemManager: Stamped EZ ID '{ez_track_id}' on MA3 track {coord}")
        
        # Store the ORIGINAL data item ID so we update the same layer, not create a new one
        original_data_item_id = layer_info.get("data_item_id")
        if original_data_item_id:
            entity.editor_data_item_id = original_data_item_id
            Log.info(f"SyncSystemManager: Using original data item: {original_data_item_id}")
        
        # CRITICAL: Store entity BEFORE hooking
        self._synced_layers[entity_id] = entity
        self._save_to_settings()
        
        Log.info(f"SyncSystemManager: Entity created - ma3_coord={entity.ma3_coord}, "
                f"editor_layer_id={entity.editor_layer_id}, editor_block_id={entity.editor_block_id}, "
                f"data_item_id={entity.editor_data_item_id}")
        
        # =========================================================================
        # Hook Track for Bidirectional Sync
        # =========================================================================
        
        if self._ma3_connected:
            self._hook_ma3_track(coord)
            Log.info(f"SyncSystemManager: Hooked track {coord}")
        else:
            Log.warning(f"SyncSystemManager: MA3 not connected, track not hooked")
        
        # =========================================================================
        # For use_existing_keep_ma3: Push MA3 events to Editor immediately
        # =========================================================================
        
        if action == "use_existing_keep_ma3":
            # We already have MA3 events in cache (fetched during dialog flow)
            # Push them to the Editor now
            ma3_events = self._get_ma3_events(coord)
            if ma3_events:
                Log.info(f"SyncSystemManager: Pushing {len(ma3_events)} MA3 events to Editor")
                self._push_ma3_to_editor(entity)
            else:
                Log.info(f"SyncSystemManager: No MA3 events to push (track may be empty)")
        
        self.entities_changed.emit()
        Log.info(f"SyncSystemManager: Editor->MA3 sync complete: {editor_layer_id} -> {coord}")
        return entity_id
    
    def _generate_unique_track_name(self, tc: int, tg: int, base_name: str) -> str:
        """Generate a unique track name by adding numeric suffix if needed.
        
        Args:
            tc: Timecode number
            tg: Track group number
            base_name: Base track name (e.g., "ez_onset")
            
        Returns:
            Unique name (e.g., "ez_onset" or "ez_onset_2" or "ez_onset_3")
        """
        # Refresh tracks from MA3 to ensure we have current state
        # (user may have deleted tracks directly in MA3)
        self._refresh_ma3_tracks_sync(tc, tg)
        
        # Check if base name is available
        if not self._check_ma3_track_name_exists(tc, tg, base_name):
            return base_name
        
        # Find next available suffix
        suffix = 2
        while suffix < 1000:  # Safety limit
            candidate = f"{base_name}_{suffix}"
            if not self._check_ma3_track_name_exists(tc, tg, candidate):
                return candidate
            suffix += 1
        
        # Fallback: use timestamp
        import time
        return f"{base_name}_{int(time.time())}"
    
    def _check_ma3_track_name_exists(self, tc: int, tg: int, name: str) -> bool:
        """Check if a track with the given name exists in MA3.
        
        IMPORTANT: This method checks the local cache. Call _refresh_ma3_tracks_sync()
        before using this method to ensure the cache is up-to-date with MA3.
        """
        # Check local state (cache must be refreshed before calling this)
        for coord, track in self._ma3_tracks.items():
            if (track.timecode_no == tc and 
                track.track_group_no == tg and 
                track.name == name):
                return True
        
        return False
    
    def _find_ma3_track_by_name(self, tc: int, tg: int, name: str) -> Optional[int]:
        """Find track number by name in local state."""
        for coord, track in self._ma3_tracks.items():
            if (track.timecode_no == tc and 
                track.track_group_no == tg and 
                track.name == name):
                return track.track_no
        return None
    
    def _create_ma3_track(self, tc: int, tg: int, name: str) -> Optional[int]:
        """Create a new track in MA3.
        
        Returns the track number of the newly created track.
        
        Flow:
        1. Send CreateTrack command to MA3
        2. Wait for MA3 to process
        3. Refresh tracks from MA3 to get ACTUAL track number
        4. Find the track by name to get its real index
        
        IMPORTANT: We MUST get the actual track number from MA3, not calculate it.
        MA3 assigns track numbers, and our calculation may be wrong.
        """
        if not self.ma3_comm_service:
            Log.error("SyncSystemManager: No MA3 comm service")
            return None
        
        # Send Lua command to create track
        lua_cmd = f'EZ.CreateTrack({tc}, {tg}, "{name}")'
        success = self._send_lua_command_with_target(lua_cmd)
        
        if not success:
            Log.error(f"SyncSystemManager: Failed to send CreateTrack command")
            return None
        
        # Wait for MA3 to report the new track (exit early if it arrives quickly)
        actual_track_no = self._wait_for_track_in_cache(tc, tg, name, timeout_s=0.7)
        
        
        if actual_track_no is not None:
            Log.info(f"SyncSystemManager: Found created track '{name}' at track #{actual_track_no}")
        
        if actual_track_no is None:
            Log.error(f"SyncSystemManager: Created track '{name}' but could not find it in MA3 response")
            return None
        
        Log.info(f"SyncSystemManager: Created track '{name}' with ACTUAL track number #{actual_track_no}")
        return actual_track_no
    
    def _process_ma3_events_for(self, duration_s: float, poll_interval_s: float = 0.02) -> None:
        """Process Qt events for a short duration to flush OSC handlers."""
        if duration_s <= 0:
            return
        
        import time
        from PyQt6.QtCore import QCoreApplication
        
        start_time = time.time()
        while time.time() - start_time < duration_s:
            QCoreApplication.processEvents()
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)
    
    def _wait_for_ma3_tracks_update(self, start_version: int, timeout_s: float = 0.3, poll_interval_s: float = 0.02) -> bool:
        """Wait for MA3 tracks cache to update after a GetTracks request."""
        import time
        from PyQt6.QtCore import QCoreApplication
        
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            QCoreApplication.processEvents()
            if self._ma3_tracks_version > start_version:
                return True
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)
        
        return self._ma3_tracks_version > start_version
    
    def _wait_for_track_in_cache(
        self,
        tc: int,
        tg: int,
        name: str,
        timeout_s: float = 0.7,
        refresh_timeout_s: float = 0.25,
    ) -> Optional[int]:
        """Wait for a newly created track to appear in the local cache."""
        import time
        
        normalized_name = name.lower()
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            self._refresh_ma3_tracks_sync(tc, tg, timeout_s=refresh_timeout_s)
            for track in self._ma3_tracks.values():
                if (
                    track.timecode_no == tc
                    and track.track_group_no == tg
                    and track.name.lower() == normalized_name
                ):
                    return track.track_no
        
        return None
    
    def _refresh_ma3_tracks_sync(self, tc: int, tg: int, timeout_s: float = 0.3, poll_interval_s: float = 0.02) -> None:
        """Synchronously refresh tracks for a specific timecode/trackgroup from MA3.
        
        Sends GetTracks command and waits for response to update cache.
        Processes Qt events to ensure OSC response handlers run.
        """
        if not self.ma3_comm_service:
            return
        
        start_version = self._ma3_tracks_version
        
        # Send request for tracks
        lua_cmd = f"EZ.GetTracks({tc}, {tg})"
        self._send_lua_command_with_target(lua_cmd)
        
        # Wait for OSC response to be processed
        # The actual update happens in the OSC message handler (on_tracks_received)
        self._wait_for_ma3_tracks_update(
            start_version,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        
        Log.debug(f"SyncSystemManager: Refreshed tracks for TC{tc}.TG{tg}")
    
    def _ensure_ma3_sequence_exists(self, seq_no: int) -> bool:
        """Ensure a sequence exists in MA3, create if needed.
        
        First checks if sequence exists using EZ.SequenceExists, then only
        creates it if it doesn't exist. This avoids unnecessary creation
        attempts for sequences that already exist.
        """
        if not self.ma3_comm_service:
            return False
        
        # Clear any pending check for this sequence
        self._pending_sequence_checks[seq_no] = None
        
        # Check if sequence exists in MA3
        check_cmd = f"EZ.SequenceExists({seq_no})"
        check_sent = self._send_lua_command_with_target(check_cmd)
        
        if not check_sent:
            Log.warning(f"SyncSystemManager: Failed to check if sequence {seq_no} exists, will attempt creation")
            # Fall back to CreateSequence (which also checks internally)
            create_cmd = f"EZ.CreateSequence({seq_no})"
            return self._send_lua_command_with_target(create_cmd)
        
        # Wait for OSC response (with timeout)
        import time
        from PyQt6.QtCore import QCoreApplication
        
        timeout = 0.5  # 500ms timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            QCoreApplication.processEvents()
            if seq_no in self._pending_sequence_checks and self._pending_sequence_checks[seq_no] is not None:
                # Got response
                exists = self._pending_sequence_checks[seq_no]
                del self._pending_sequence_checks[seq_no]
                
                if exists:
                    Log.info(f"SyncSystemManager: Sequence {seq_no} already exists, skipping creation")
                    return True
                else:
                    Log.info(f"SyncSystemManager: Sequence {seq_no} does not exist, creating it")
                    create_cmd = f"EZ.CreateSequence({seq_no})"
                    success = self._send_lua_command_with_target(create_cmd)
                    if success:
                        Log.info(f"SyncSystemManager: Created sequence {seq_no}")
                    return success
            
            time.sleep(0.05)  # 50ms between checks
        
        # Timeout - didn't get response, fall back to CreateSequence
        Log.warning(f"SyncSystemManager: Timeout waiting for SequenceExists response for {seq_no}, attempting creation")
        if seq_no in self._pending_sequence_checks:
            del self._pending_sequence_checks[seq_no]
        
        create_cmd = f"EZ.CreateSequence({seq_no})"
        success = self._send_lua_command_with_target(create_cmd)
        if success:
            Log.info(f"SyncSystemManager: Ensured sequence {seq_no} exists (timeout fallback)")
        return success
    
    def _assign_ma3_track_sequence(self, tc: int, tg: int, track_no: int, seq_no: int) -> bool:
        """Assign a sequence to an MA3 track."""
        if not self.ma3_comm_service:
            return False
        
        lua_cmd = f"EZ.AssignTrackSequence({tc}, {tg}, {track_no}, {seq_no})"
        success = self._send_lua_command_with_target(lua_cmd)
        
        if success:
            Log.info(f"SyncSystemManager: Assigned track {tc}.{tg}.{track_no} to sequence {seq_no}")
        return success
    
    def _create_ma3_cmd_subtrack(self, tc: int, tg: int, track_no: int, time_range_index: int = 1) -> bool:
        """Create a CmdSubTrack in the specified TimeRange.
        
        This must be called after sequence assignment, as MA3 requires
        a sequence to be assigned before CmdSubTrack can be created.
        
        Args:
            tc: Timecode number
            tg: Track group number
            track_no: Track number (user-visible, 1-based)
            time_range_index: Index of TimeRange (default: 1 for first TimeRange)
            
        Returns:
            True if command was sent successfully, False otherwise
        """
        if not self.ma3_comm_service:
            return False
        
        lua_cmd = f"EZ.CreateCmdSubTrack({tc}, {tg}, {track_no}, {time_range_index})"
        success = self._send_lua_command_with_target(lua_cmd)
        
        if success:
            Log.info(f"SyncSystemManager: Created CmdSubTrack for track {tc}.{tg}.{track_no} (TimeRange index {time_range_index})")
        else:
            Log.error(f"SyncSystemManager: Failed to create CmdSubTrack for track {tc}.{tg}.{track_no}")
        
        return success
    
    def _verify_track_ready(self, tc: int, tg: int, track_no: int) -> bool:
        """Verify track is ready for events (has sequence, TimeRange, CmdSubTrack).
        
        Calls EZ.VerifyTrackReady and logs the result.
        """
        if not self.ma3_comm_service:
            return False
        
        # Call Lua verification
        lua_cmd = f"EZ.VerifyTrackReady({tc}, {tg}, {track_no})"
        success = self._send_lua_command_with_target(lua_cmd)
        
        if success:
            Log.info(f"SyncSystemManager: Track {tc}.{tg}.{track_no} verification requested")
        else:
            Log.error(f"SyncSystemManager: Failed to request track verification")
        
        # Note: The actual verification result comes via Lua logs
        # For now, assume success if command was sent
        # TODO: Add proper response handling via OSC
        return success
    
    def _clear_ma3_track(self, tc: int, tg: int, track_no: int) -> bool:
        """Clear all events from an MA3 track."""
        if not self.ma3_comm_service:
            return False
        
        lua_cmd = f"EZ.ClearTrack({tc}, {tg}, {track_no})"
        return self._send_lua_command_with_target(lua_cmd)
    
    def _add_events_to_ma3(self, tc: int, tg: int, track_no: int, events: List[Any]) -> bool:
        """Add events to an MA3 track.
        
        Args:
            tc: Timecode number
            tg: Track group number
            track_no: Track number
            events: List of events (can be EventInfo objects or dicts)
        """
        if not self.ma3_comm_service or not events:
            return True  # Nothing to add
        
        for event in events:
            # Handle both EventInfo objects and dictionaries
            if hasattr(event, 'time'):
                # EventInfo object
                time_val = event.time
                # Build cmd from metadata or use empty string
                cmd_val = ""
                if hasattr(event, 'metadata') and event.metadata:
                    cmd_val = event.metadata.get("cmd", "")
            else:
                # Dictionary
                time_val = event.get("time", 0)
                cmd_val = event.get("cmd", "")
            
            # Escape double quotes in cmd (we use double quotes for string params)
            cmd_val = str(cmd_val).replace('"', '\\"')
            
            lua_cmd = f'EZ.AddEvent({tc}, {tg}, {track_no}, {time_val}, "{cmd_val}")'
            if not self._send_lua_command_with_target(lua_cmd):
                Log.warning(f"SyncSystemManager: Failed to add event at {time_val}")
        
        Log.info(f"SyncSystemManager: Added {len(events)} events to MA3 track")
        return True
    
    def _get_ma3_track_info(self, coord: str) -> Optional[Dict[str, Any]]:
        """
        Get MA3 track info by coordinate from local state.
        """
        track_info = self._ma3_tracks.get(coord)
        if track_info:
            return {
                "coord": track_info.coord,
                "timecode_no": track_info.timecode_no,
                "track_group": track_info.track_group_no,
                "track": track_info.track_no,
                "name": track_info.name,
                "event_count": track_info.event_count,
                "sequence_no": track_info.sequence_no,
                "note": track_info.note,
            }
        
        # Fallback: parse coord if not in cache
        parts = self._parse_ma3_coord(coord)
        if not parts:
            return None
        
        return {
            "coord": coord,
            "timecode_no": parts["timecode_no"],
            "track_group": parts["track_group"],
            "track": parts["track"],
            "name": coord,
            "event_count": 0,
            "sequence_no": None,
            "note": "",
        }
    
    def _get_ma3_events(self, coord: str) -> List[Dict[str, Any]]:
        """Get events from MA3 track from local state."""
        events = self._ma3_track_events.get(coord, [])
        return [
            {
                "time": e.time,
                "name": e.name,
                "cmd": e.cmd,
                "idx": e.idx,
            }
            for e in events
        ]
    
    def _get_all_ma3_tracks(self) -> List[Dict[str, Any]]:
        """Get all MA3 tracks from local state."""
        return [
            {
                "coord": track.coord,
                "timecode_no": track.timecode_no,
                "track_group": track.track_group_no,
                "track": track.track_no,
                "name": track.name,
                "event_count": track.event_count,
                "sequence_no": track.sequence_no,
            }
            for track in self._ma3_tracks.values()
        ]
    
    # =========================================================================
    # MA3 State Update Methods (called when OSC messages arrive)
    # =========================================================================
    
    def on_track_groups_received(self, timecode_no: int, groups: List[Dict[str, Any]]) -> None:
        """
        Called when track groups list is received from MA3.
        
        Args:
            timecode_no: The timecode number
            groups: List of track group dicts with 'no', 'name', 'track_count'
        """
        self._ma3_track_groups[timecode_no] = [
            MA3TrackGroupInfo(
                timecode_no=timecode_no,
                track_group_no=g.get("no", 0),
                name=g.get("name", ""),
                track_count=g.get("track_count", 0),
            )
            for g in groups
        ]
        Log.info(f"SyncSystemManager: Stored {len(groups)} track groups for TC{timecode_no}")
        self.entities_changed.emit()
    
    def on_tracks_received(self, timecode_no: int, track_group_no: int, tracks: List[Dict[str, Any]]) -> None:
        """
        Called when tracks list is received from MA3.
        
        Args:
            timecode_no: The timecode number
            track_group_no: The track group number
            tracks: List of track dicts with 'no', 'name'
        """
        self._ma3_tracks_version += 1
        
        # First, remove all existing tracks for this TC/TG (handles deleted tracks)
        prefix = f"tc{timecode_no}_tg{track_group_no}_tr"
        stale_coords = [coord for coord in self._ma3_tracks if coord.startswith(prefix)]
        for coord in stale_coords:
            del self._ma3_tracks[coord]
        
        # Then add the current tracks from MA3
        for t in tracks:
            track_no = t.get("no", 0)
            coord = f"tc{timecode_no}_tg{track_group_no}_tr{track_no}"
            event_count = t.get("event_count", 0)
            sequence_no = t.get("sequence_no")  # May be None if not assigned
            note = t.get("note", "")
            self._ma3_tracks[coord] = MA3TrackInfo(
                timecode_no=timecode_no,
                track_group_no=track_group_no,
                track_no=track_no,
                name=t.get("name", f"Track {track_no}"),
                coord=coord,
                event_count=event_count,
                sequence_no=sequence_no,
                note=note,
            )
            Log.debug(f"SyncSystemManager: Track {coord}: name='{t.get('name')}', event_count={event_count}, sequence={sequence_no}, note='{note}'")
        Log.info(f"SyncSystemManager: Refreshed {len(tracks)} tracks for TC{timecode_no}.TG{track_group_no} (cleared {len(stale_coords)} stale)")
        
        # Auto-reconnect disconnected/unmapped entities using the fresh track list
        try:
            reconnect_results = self.auto_reconnect_layers()
            if reconnect_results:
                reconnected = sum(1 for v in reconnect_results.values() if v == "reconnected")
                if reconnected > 0:
                    Log.info(f"SyncSystemManager: Auto-reconnected {reconnected} layer(s) after track refresh")
        except Exception as e:
            Log.warning(f"SyncSystemManager: auto_reconnect_layers failed: {e}")
        
        self.entities_changed.emit()
    
    def on_track_events_received(self, coord: str, events: List[Dict[str, Any]]) -> None:
        """
        Called when track events are received from MA3 (via hook callback or fetch).
        
        This is where all divergence detection now happens (moved from Lua).
        When events are received, we:
        1. Store them in local state
        2. If the track is synced, compare with Editor events
        3. Auto-sync if bidirectional is enabled, or flag divergence
        
        Args:
            coord: The track coordinate (tc{n}_tg{n}_tr{n})
            events: List of event dicts with 'time', 'name', 'cmd', 'idx'
        """
        import time as _time_mod
        _recv_ts = _time_mod.time()
        _track_changed_ts = self._diag_track_changed_ts.get(coord, 0.0)
        _round_trip_ms = ((_recv_ts - _track_changed_ts) * 1000) if _track_changed_ts else -1
        Log.info(f"SyncSystemManager: on_track_events_received coord={coord} events={len(events)}")
        Log.info(f"[MULTITRACK-DIAG] on_track_events_received: coord={coord}, event_count={len(events)}, round_trip_ms={_round_trip_ms:.1f}, timestamp={_recv_ts:.4f}")
        # Clear retry counter on successful receipt
        self._events_request_retries.pop(coord, None)
        if hasattr(self, "_pending_events_requests"):
            self._pending_events_requests.pop(coord, None)
        if hasattr(self, "_pending_events_request_ids"):
            self._pending_events_request_ids.pop(coord, None)
        self._ma3_events_in_flight.pop(coord, None)
        
        
        # Store the raw events from MA3
        new_events = [
            MA3EventInfo(
                time=e.get("time", 0.0),
                name=e.get("name", ""),
                cmd=e.get("cmd", ""),
                idx=e.get("idx", i),
                tc=e.get("tc", 0),
                tg=e.get("tg", 0),
                track=e.get("track", 0),
            )
            for i, e in enumerate(events)
        ]
        
        # Get previous events for comparison
        old_events = self._ma3_track_events.get(coord, [])
        
        
        # Update local state
        self._ma3_track_events[coord] = new_events
        
        # Update event count in track info
        if coord in self._ma3_tracks:
            self._ma3_tracks[coord].event_count = len(events)
        
        Log.debug(f"SyncSystemManager: Stored {len(events)} events for {coord}")
        
        
        # Check if this track is synced
        entity = self.get_synced_layer_by_ma3_coord(coord)
        if not entity:
            # Debug: log what we're looking for vs what we have
            Log.info(f"SyncSystemManager: No entity found for coord '{coord}'")
            Log.debug(f"SyncSystemManager: Available coords: {[e.ma3_coord for e in self._synced_layers.values()]}")
            return
        
        Log.info(f"SyncSystemManager: Found entity for {coord}: editor_layer={entity.editor_layer_id}, sync_status={entity.sync_status}")
        
        
        # Ignore empty callbacks shortly after an Editor→MA3 push
        import time
        last_editor_push = self._last_editor_push_time.get(coord)
        ignore_empty = bool(last_editor_push) and len(events) == 0 and (time.time() - last_editor_push) < self._ma3_empty_ignore_window_s
        if ignore_empty:
            Log.info(f"SyncSystemManager: Ignoring empty MA3 callback for {coord} (within {self._ma3_empty_ignore_window_s}s of Editor→MA3 push)")
            self.entity_updated.emit(entity.id)
            return
        
        # Check if we should skip MA3->Editor sync (set during Editor->MA3 push to prevent feedback loop)
        skip_initial = self._skip_initial_hook.get(coord, False)
        
        if skip_initial:
            Log.info(f"SyncSystemManager: Skipping MA3->Editor sync for {coord} (Editor was just pushed to MA3, event_count={len(events)})")
            # Clear the flag after handling - cache is already updated above
            if coord in self._skip_initial_hook:
                del self._skip_initial_hook[coord]
            # Emit entity_updated so UI knows cache was refreshed (but don't sync to Editor)
            self.entity_updated.emit(entity.id)
            return
        
        # ---- SYNCED path: live editing, auto-push allowed ----
        if entity.sync_status == SyncStatus.SYNCED:
            Log.info(f"SyncSystemManager: Entity is SYNCED (live editing), checking for updates...")
            
            comparison = self._compare_entity(entity)
            
            
            apply_updates = True
            if entity.settings and hasattr(entity.settings, 'apply_updates_enabled'):
                apply_updates = entity.settings.apply_updates_enabled
            force_apply = bool(self._force_apply_to_ez.get(coord))
            if force_apply:
                apply_updates = True
            
            needs_push = False
            if comparison and comparison.diverged:
                Log.info(f"SyncSystemManager: Live divergence for {coord}: "
                        f"MA3={comparison.ma3_count}, Editor={comparison.editor_count}")
                needs_push = True
            elif not entity.editor_data_item_id:
                Log.info(f"SyncSystemManager: First-time sync for {coord} (creating data item)")
                needs_push = True
            elif apply_updates and len(new_events) > 0:
                Log.info(f"SyncSystemManager: Live MA3 events for {coord} ({len(new_events)} events) - pushing to Editor")
                needs_push = True
            
            if force_apply:
                needs_push = True
            
            if needs_push and apply_updates:
                self._schedule_push_ma3_to_editor(entity)
                Log.info(f"SyncSystemManager: Scheduled live MA3->Editor push for {coord}")
                if force_apply:
                    self._force_apply_to_ez.pop(coord, None)
            elif needs_push and not apply_updates:
                Log.info(f"SyncSystemManager: Live updates disabled for {coord}, no push")
        
        # ---- PENDING path: reconnect/load, compare only, never auto-push ----
        elif entity.sync_status == SyncStatus.PENDING:
            Log.info(f"SyncSystemManager: Entity is PENDING (reconnect/load), comparing...")
            
            comparison = self._compare_entity(entity)
            
            if comparison and comparison.diverged:
                entity.mark_diverged()
                Log.info(
                    f"SyncSystemManager: PENDING entity '{entity.name}' diverged: "
                    f"MA3={comparison.ma3_count}, Editor={comparison.editor_count}, "
                    f"Matched={comparison.matched_count}"
                )
                self._save_to_settings()
                self.entities_changed.emit()
                self._push_sync_state_to_all_layers()
            else:
                entity.mark_synced()
                Log.info(f"SyncSystemManager: Promoted '{entity.name}' from PENDING to SYNCED (no divergence)")
                self._save_to_settings()
                self._push_sync_state_to_all_layers()
        
        self.entity_updated.emit(entity.id)
    
    def clear_ma3_state(self) -> None:
        """Clear all MA3 state (e.g., on disconnect)."""
        self._ma3_track_groups.clear()
        self._ma3_tracks.clear()
        self._ma3_track_events.clear()
        Log.info("SyncSystemManager: Cleared MA3 state")
    
    def _update_ma3_sequence(self, entity: SyncLayerEntity) -> None:
        """Update MA3 track.target (sequence assignment)."""
        if not entity.ma3_coord or not self.ma3_comm_service:
            return
        
        # TODO: Call EZ.AssignTrackSequence via OSC
        Log.debug(f"SyncSystemManager: Update MA3 sequence for {entity.ma3_coord}")
    
    def _parse_ma3_coord(self, coord: str) -> Optional[Dict[str, int]]:
        """Parse MA3 coordinate string to components."""
        import re
        match = re.match(r"tc(\d+)_tg(\d+)_tr(\d+)", coord)
        if match:
            return {
                "timecode_no": int(match.group(1)),
                "track_group": int(match.group(2)),
                "track": int(match.group(3)),
            }
        return None
    
    # =========================================================================
    # Private: Editor Operations
    # =========================================================================
    
    def _get_editor_block_id(self) -> Optional[str]:
        """
        Get the Editor block ID connected to this ShowManager.
        
        Uses a cache to avoid repeated connection traversal on every sync
        callback.  Call ``invalidate_editor_cache()`` when connections change.
        """
        if self._cached_editor_block_id is not None:
            return self._cached_editor_block_id
        
        try:
            connections_result = self._facade.list_connections()
            if not connections_result.success or not connections_result.data:
                return None
            
            for conn in connections_result.data:
                if conn.source_block_id == self._show_manager_block_id:
                    target_result = self._facade.describe_block(conn.target_block_id)
                    if target_result.success and target_result.data:
                        if target_result.data.type == "Editor":
                            self._cached_editor_block_id = target_result.data.id
                            return self._cached_editor_block_id
        except Exception as e:
            Log.warning(f"SyncSystemManager: Error finding connected Editor: {e}")
        
        return None
    
    def invalidate_editor_cache(self) -> None:
        """Clear cached editor block ID and API.
        
        Call when block connections change (e.g., Editor reconnected).
        """
        self._cached_editor_block_id = None
        self._cached_editor_api = None
        self._cached_editor_api_block_id = None
    
    def _get_editor_api(self, block_id: Optional[str]) -> Optional["EditorAPI"]:
        """Get EditorAPI for a block (cached to avoid repeated creation)."""
        if not block_id:
            block_id = self._get_editor_block_id()
        if not block_id:
            return None
        
        # Return cached API if it matches the requested block
        if (self._cached_editor_api is not None
                and self._cached_editor_api_block_id == block_id):
            return self._cached_editor_api
        
        try:
            from src.features.blocks.application.editor_api import create_editor_api
            api = create_editor_api(self._facade, block_id)
            self._cached_editor_api = api
            self._cached_editor_api_block_id = block_id
            return api
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to create EditorAPI: {e}")
            return None
    
    def _get_editor_layer_info(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Get Editor layer info, including the data item ID."""
        
        editor_api = self._get_editor_api(None)
        if not editor_api:
            return None
        
        try:
            layer = editor_api.get_layer(layer_id)
            
            if layer:
                # Find the data item that contains this layer's events
                data_item_id = self._find_data_item_for_layer(layer.group_name)
                result = {
                    "layer_id": layer.name,
                    "name": layer.name,
                    "block_id": self._get_editor_block_id(),
                    "group_id": layer.group_id,
                    "group_name": layer.group_name,
                    "event_count": layer.event_count,
                    "data_item_id": data_item_id,
                }
                return result
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to get layer info: {e}")
        
        return None
    
    def _find_data_item_for_layer(self, group_name: str) -> Optional[str]:
        """Find the data item ID that matches a layer's group_name."""
        from src.shared.domain.entities import EventDataItem
        
        editor_block_id = self._get_editor_block_id()
        if not editor_block_id or not self._facade.data_item_repo:
            return None
        
        items = self._facade.data_item_repo.list_by_block(editor_block_id)
        for item in items:
            if isinstance(item, EventDataItem):
                # Match by name (group_name corresponds to data item name)
                if item.name == group_name:
                    return item.id
        
        return None
    
    def _get_editor_events(self, layer_id: str, block_id: Optional[str]) -> List[Any]:
        """Get events from Editor layer."""
        
        editor_api = self._get_editor_api(block_id)
        if not editor_api:
            return []
        
        try:
            events = editor_api.get_events_in_layer(layer_id)
            return events or []
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to get editor events: {e}")
            return []
    
    def _get_all_editor_layers(self) -> List[Dict[str, Any]]:
        """Get all Editor layers."""
        editor_api = self._get_editor_api(None)
        if not editor_api:
            return []
        
        try:
            layers = editor_api.get_layers()
            editor_block_id = self._get_editor_block_id()
            return [
                {
                    "layer_id": layer.name,
                    "name": layer.name,
                    "block_id": editor_block_id,
                    "group_name": layer.group_name,
                    "event_count": layer.event_count,
                    "is_synced": layer.is_synced,
                }
                for layer in layers
            ]
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to get all layers: {e}")
            return []
    
    def _create_editor_layer(
        self,
        name: str,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        is_synced: bool = False,
        ma3_track_coord: Optional[str] = None,
    ) -> Optional[str]:
        """Create a new Editor layer with proper group information."""
        editor_api = self._get_editor_api(None)
        if not editor_api:
            Log.warning("SyncSystemManager: Cannot create layer - no EditorAPI")
            return name  # Return the name as a fallback
        
        # Ensure we have valid group info (required for layer validation)
        if not group_id:
            group_id = "ma3_synced"
        if not group_name:
            group_name = "MA3 Synced"
        
        try:
            layer = editor_api.create_layer(
                name=name,
                group_id=group_id,
                group_name=group_name,
                is_synced=is_synced,
                show_manager_block_id=self._show_manager_block_id if is_synced else None,
                ma3_track_coord=ma3_track_coord,
            )
            if layer:
                Log.info(f"SyncSystemManager: Created Editor layer: {name} (group: {group_name})")
                return layer.name
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to create layer: {e}")
        
        return name  # Return the name as a fallback
    
    def _schedule_sync_state_push(self) -> None:
        """Schedule a deferred push of sync states to all layers.
        
        Uses QTimer.singleShot(0) to run after all pending Qt events
        (e.g., BlockUpdated -> Editor reload creating fresh TimelineLayer objects).
        """
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._push_sync_state_to_all_layers)
    
    def _push_sync_state_to_all_layers(self) -> None:
        """Push sync_connection_state onto every Editor TimelineLayer.
        
        This is the ONLY place sync icon state is determined.
        Source of truth: self._synced_layers (in-memory SSM entities).
        
        Called after any mutation (sync, unsync, MA3 connect/disconnect).
        
        Uses a cached reference to the LayerLabels widget(s) to avoid
        scanning ``app.allWidgets()`` (which can contain thousands of
        widgets) on every call.
        """
        from src.features.show_manager.domain.sync_layer_entity import SyncStatus
        
        # Build map: editor_layer_id -> desired state
        state_map: Dict[str, str] = {}
        for entity in self._synced_layers.values():
            if entity.editor_layer_id:
                # Entities for a different timecode are never "active"
                if (entity.ma3_timecode_no is not None
                        and entity.ma3_timecode_no != self._configured_timecode):
                    state_map[entity.editor_layer_id] = "disconnected"
                elif entity.sync_status == SyncStatus.SYNCED:
                    state_map[entity.editor_layer_id] = "active"
                elif entity.sync_status == SyncStatus.DIVERGED:
                    state_map[entity.editor_layer_id] = "diverged"
                elif entity.sync_status == SyncStatus.AWAITING_CONNECTION:
                    state_map[entity.editor_layer_id] = "awaiting_connection"
                else:
                    state_map[entity.editor_layer_id] = "disconnected"
        
        # Use cached widget references; fall back to full scan when stale.
        layer_labels_widgets = self._get_layer_labels_widgets()
        
        for widget in layer_labels_widgets:
            layer_manager = getattr(widget, '_layer_manager', None)
            if not layer_manager:
                continue
            changed = False
            for layer in layer_manager.get_all_layers():
                desired = state_map.get(layer.name)
                if desired is None:
                    # Not in any SSM entity
                    if getattr(layer, 'derived_from_ma3', False):
                        desired = "derived"
                    elif getattr(layer, 'is_synced', False):
                        # Layer claims synced but SSM has no matching entity
                        desired = "disconnected"
                    else:
                        desired = "none"
                if getattr(layer, 'sync_connection_state', 'none') != desired:
                    layer.sync_connection_state = desired
                    changed = True
            if changed:
                widget.update()
    
    def _get_layer_labels_widgets(self) -> list:
        """Return cached LayerLabels widget references.
        
        Caches the result of the expensive ``app.allWidgets()`` scan.
        Automatically re-scans when the cached references become stale
        (widget deleted by Qt).
        """
        from PyQt6.QtWidgets import QApplication
        from PyQt6 import sip  # PyQt6 bundles sip as PyQt6.sip
        from ui.qt_gui.widgets.timeline.core.widget import LayerLabels
        
        # Check if cached references are still alive
        if hasattr(self, "_cached_layer_labels"):
            alive = []
            for w in self._cached_layer_labels:
                try:
                    if not sip.isdeleted(w):
                        alive.append(w)
                except Exception:
                    pass
            if alive:
                return alive
        
        # Full scan (only on first call or when cache becomes stale)
        app = QApplication.instance()
        if not app:
            self._cached_layer_labels = []
            return []
        
        self._cached_layer_labels = [
            w for w in app.allWidgets() if isinstance(w, LayerLabels)
        ]
        return self._cached_layer_labels
        
    
    def _update_existing_editor_layer_sync(
        self,
        editor_block_id: str,
        editor_layer_id: str,
        ma3_track_coord: str,
    ) -> None:
        """Mark an existing Editor layer as synced. Fail loud.
        
        Single path, no fallbacks:
        1. Find the in-memory TimelineLayer by name (raise if not found)
        2. Set is_synced, show_manager_block_id, ma3_track_coord on it
        3. Persist the full layer set to UI state (raise if it fails)
        4. Invalidate sync cache so the icon repaints green
        
        Args:
            editor_block_id: The Editor block ID
            editor_layer_id: The Editor layer name/ID to update
            ma3_track_coord: The MA3 track coordinate being synced
            
        Raises:
            RuntimeError: If the layer cannot be found or state cannot be persisted
        """
        from PyQt6.QtWidgets import QApplication
        from ui.qt_gui.widgets.timeline.core.widget import LayerLabels
        
        # --- Step 1: Find the in-memory TimelineLayer ---
        app = QApplication.instance()
        if not app:
            raise RuntimeError(
                f"_update_existing_editor_layer_sync: No QApplication instance"
            )
        
        target_layer = None
        target_labels_widget = None
        for widget in app.allWidgets():
            if not isinstance(widget, LayerLabels):
                continue
            layer_manager = getattr(widget, '_layer_manager', None)
            if not layer_manager:
                continue
            for layer in layer_manager.get_all_layers():
                if layer.name == editor_layer_id:
                    target_layer = layer
                    target_labels_widget = widget
                    break
            if target_layer:
                break
        
        if target_layer is None:
            raise RuntimeError(
                f"_update_existing_editor_layer_sync: TimelineLayer '{editor_layer_id}' "
                f"not found in any LayerLabels widget"
            )
        
        # --- Step 2: Set sync properties on the in-memory layer ---
        target_layer.is_synced = True
        target_layer.show_manager_block_id = self._show_manager_block_id
        target_layer.ma3_track_coord = ma3_track_coord
        target_layer.derived_from_ma3 = False
        target_layer.sync_connection_state = "active"
        
        # --- Step 3: Persist the full layer set to UI state ---
        # Read current persisted layers, update the matching entry (or create
        # one), and write back.
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=editor_block_id,
        )
        layers = result.data.get('layers', []) if result.success and result.data else []
        
        # Build the updated entry from the live layer object
        updated_entry = {
            'name': target_layer.name,
            'height': target_layer.height,
            'color': target_layer.color,
            'visible': target_layer.visible,
            'locked': target_layer.locked,
            'group_id': target_layer.group_id,
            'group_name': target_layer.group_name,
            'group_index': getattr(target_layer, 'group_index', None),
            'is_synced': True,
            'show_manager_block_id': self._show_manager_block_id,
            'ma3_track_coord': ma3_track_coord,
        }
        
        # Replace existing entry or append new one
        found = False
        for i, entry in enumerate(layers):
            if entry.get('name') == editor_layer_id:
                layers[i] = updated_entry
                found = True
                break
        if not found:
            layers.append(updated_entry)
        
        save_result = self._facade.set_ui_state(
            state_type='editor_layers',
            entity_id=editor_block_id,
            data={'layers': layers},
        )
        if not save_result.success:
            raise RuntimeError(
                f"_update_existing_editor_layer_sync: Failed to persist UI state "
                f"for layer '{editor_layer_id}': {save_result.message}"
            )
        
        # --- Step 4: Invalidate sync cache and repaint ---
        target_labels_widget.invalidate_sync_cache()
        
        Log.info(
            f"SyncSystemManager: Marked Editor layer '{editor_layer_id}' as synced "
            f"(coord={ma3_track_coord}, block={editor_block_id})"
        )
    
    def _delete_editor_layer(self, block_id: str, layer_id: str) -> bool:
        """Delete an Editor layer."""
        editor_api = self._get_editor_api(block_id)
        if not editor_api:
            Log.warning("SyncSystemManager: Cannot delete layer - no EditorAPI")
            return False
        
        try:
            success = editor_api.delete_layer(layer_id)
            if success:
                Log.info(f"SyncSystemManager: Deleted Editor layer: {layer_id}")
            return success
        except Exception as e:
            Log.warning(f"SyncSystemManager: Failed to delete layer: {e}")
            return False
    
    # =========================================================================
    # Private: Persistence
    # =========================================================================
    
    def _load_from_settings(self) -> None:
        """Load synced layers and configuration from settings."""
        if not self._settings_manager:
            return
        
        # Load configured timecode from settings
        self._configured_timecode = self._settings_manager.target_timecode
        Log.info(f"SyncSystemManager: Loaded target_timecode={self._configured_timecode} from settings")
        
        synced_data = self._settings_manager.synced_layers
        
        for data in synced_data:
            if not isinstance(data, dict):
                continue
            
            try:
                # Try new format first
                if "id" in data and "source" in data:
                    entity = SyncLayerEntity.from_dict(data)
                    # If entity already exists in memory, preserve its runtime
                    # sync_status. The in-memory status reflects real-time state
                    # (e.g. PENDING after downgrade) while persisted status may
                    # be stale. Only structural fields are updated from DB.
                    existing = self._synced_layers.get(entity.id)
                    if existing:
                        existing.name = entity.name
                        existing.ma3_coord = entity.ma3_coord
                        existing.ma3_timecode_no = entity.ma3_timecode_no
                        existing.ma3_track_group = entity.ma3_track_group
                        existing.ma3_track = entity.ma3_track
                        existing.editor_layer_id = entity.editor_layer_id
                        existing.editor_block_id = entity.editor_block_id
                        existing.editor_data_item_id = entity.editor_data_item_id
                        existing.settings = entity.settings
                        existing.ez_track_id = entity.ez_track_id
                        existing.group_name = entity.group_name
                        # Preserve: sync_status, error_message, event_count,
                        # last_sync_time (runtime state)
                    else:
                        self._synced_layers[entity.id] = entity
                else:
                    # Handle legacy format
                    entity = self._migrate_legacy_entity(data)
                    if entity:
                        existing = self._synced_layers.get(entity.id)
                        if not existing:
                            self._synced_layers[entity.id] = entity
            except Exception as e:
                Log.warning(f"SyncSystemManager: Failed to load entity: {e}")
        
        Log.info(f"SyncSystemManager: Loaded {len(self._synced_layers)} synced layers")
        
        # Validate loaded entities have required fields - FAIL LOUD
        for entity_id, entity in list(self._synced_layers.items()):
            if entity.ma3_coord:
                # Entity has MA3 coord - timecode MUST be present
                if entity.ma3_timecode_no is None:
                    error_msg = (
                        f"SyncSystemManager: CRITICAL - Entity {entity_id} has ma3_coord={entity.ma3_coord} "
                        f"but ma3_timecode_no is None. This indicates a data corruption bug. "
                        f"Entity: {entity.name}, source: {entity.source.value}"
                    )
                    Log.error(error_msg)
                    # Remove invalid entity to prevent silent failures
                    del self._synced_layers[entity_id]
                    # Try to extract from coord as diagnostic, but log as error
                    parts = self._parse_ma3_coord(entity.ma3_coord)
                    if parts:
                        Log.error(
                            f"SyncSystemManager: Extracted timecode {parts['timecode_no']} from coord, "
                            f"but entity should have been saved with timecode. This is a bug."
                        )
                    continue
            
            # Validate entity has either MA3 side OR Editor side
            if not entity.ma3_coord and not entity.editor_layer_id:
                error_msg = (
                    f"SyncSystemManager: CRITICAL - Entity {entity_id} has neither ma3_coord nor editor_layer_id. "
                    f"This is invalid. Entity: {entity.name}, source: {entity.source.value}"
                )
                Log.error(error_msg)
                del self._synced_layers[entity_id]
                continue
        
        # Re-initialize layers for current timecode
        self._reinitialize_for_current_timecode()
        
        # If no actual MA3 track data has been received, downgrade SYNCED
        # entities to AWAITING_CONNECTION. We check _ma3_tracks (populated
        # only when MA3 actually sends track data via OSC) rather than
        # _ma3_connected (which only means the OSC listener socket is open,
        # not that MA3 is actually connected and communicating).
        # When MA3 sends data, auto_reconnect_layers() will upgrade them
        # back to SYNCED (or DIVERGED if events changed).
        if not self._ma3_tracks:
            downgraded = 0
            for entity in self._synced_layers.values():
                if entity.sync_status == SyncStatus.SYNCED:
                    entity.sync_status = SyncStatus.AWAITING_CONNECTION
                    downgraded += 1
            if downgraded:
                Log.info(
                    f"SyncSystemManager: Downgraded {downgraded} SYNCED entities to AWAITING_CONNECTION "
                    f"(no MA3 track data received yet)"
                )
        
        # Push sync states to editor after loading (ensures icons reflect current state)
        self._push_sync_state_to_all_layers()
    
    def _reinitialize_for_current_timecode(self) -> None:
        """
        Re-initialize layers for the current configured timecode.
        
        - Unhooks all layers NOT matching current timecode
        - Re-hooks and re-syncs all layers matching current timecode
        - Preserves sync relationships (entities remain in _synced_layers)
        - Layers without timecode info are treated as not matching (unhooked)
        """
        current_tc = self._configured_timecode
        
        # Unhook layers from other timecodes (or without timecode info)
        for entity in self._synced_layers.values():
            if entity.ma3_coord:
                # Unhook if timecode doesn't match or is None
                if entity.ma3_timecode_no != current_tc:
                    if entity.ma3_coord in self._hooked_tracks:
                        self._unhook_ma3_track(entity.ma3_coord)
        
        # Re-hook and re-sync layers matching current timecode
        if self.ma3_comm_service:
            for entity in self._synced_layers.values():
                if entity.ma3_coord and entity.ma3_timecode_no == current_tc:
                    if entity.ma3_coord not in self._hooked_tracks:
                        self._hook_ma3_track(entity.ma3_coord)
                        self._request_ma3_events(entity.ma3_coord)
    
    def _on_project_changed(self, event) -> None:
        """
        Handle project change events.
        
        When a project is loaded or created, reload synced layers from settings
        and unhook all MA3 tracks to ensure clean state for the new project.
        """
        Log.info("SyncSystemManager: Project changed, reloading synced layers and unhooking MA3 tracks")
        # Reload synced layers from settings (project file contains block metadata with settings)
        self._load_from_settings()
        # Unhook all MA3 tracks to ensure clean state
        self.unhook_all_ma3_tracks()
    
    def cleanup(self) -> None:
        """
        Cleanup resources on shutdown.
        
        Unhooks all MA3 tracks and clears local state.
        Should be called when the panel closes or application shuts down.
        """
        Log.info("SyncSystemManager: Cleaning up resources")
        
        # Save synced layers before cleanup
        self._save_to_settings()
        
        self.unhook_all_ma3_tracks()
        
        # Clear other state
        self._synced_layers.clear()
        self._ma3_tracks.clear()
        self._ma3_track_groups.clear()
        self._ma3_track_events.clear()
        self._syncing_from_ma3.clear()
        self._last_ma3_push_time.clear()
        self._last_editor_push_time.clear()
        self._force_apply_to_ez.clear()
        self._ma3_apply_cooldown_until.clear()
        self._ma3_events_in_flight.clear()
        self._skip_initial_hook.clear()
        self._pending_sequence_checks.clear()
        
        Log.info("SyncSystemManager: Cleanup complete")
    
    def _save_to_settings(self) -> None:
        """Save synced layers to settings."""
        if not self._settings_manager:
            return
        
        synced_data = [entity.to_dict() for entity in self._synced_layers.values()]
        self._settings_manager.synced_layers = synced_data
        # Force immediate save to ensure persistence
        self._settings_manager.force_save()
    
    def _migrate_legacy_entity(self, data: Dict[str, Any]) -> Optional[SyncLayerEntity]:
        """Migrate legacy entity format to new SyncLayerEntity."""
        entity_id = str(uuid.uuid4())
        
        # Determine source from available fields
        if "coord" in data:
            # MA3TrackEntity format
            return SyncLayerEntity(
                id=entity_id,
                source=SyncSource.MA3,
                name=data.get("name", ""),
                ma3_coord=data.get("coord"),
                ma3_timecode_no=data.get("timecode_no"),
                ma3_track_group=data.get("track_group"),
                ma3_track=data.get("track"),
                editor_layer_id=data.get("mapped_editor_layer_id"),
                editor_block_id=None,  # Will need to look up
                sync_status=SyncStatus.SYNCED if data.get("mapped_editor_layer_id") else SyncStatus.UNMAPPED,
                event_count=data.get("event_count", 0),
                settings=SyncLayerSettings.from_dict(data.get("settings", {})),
                group_name=data.get("group_name"),
            )
        elif "layer_id" in data:
            # EditorLayerEntity format
            return SyncLayerEntity(
                id=entity_id,
                source=SyncSource.EDITOR,
                name=data.get("name", data.get("layer_id", "")),
                editor_layer_id=data.get("layer_id"),
                editor_block_id=data.get("block_id"),
                ma3_coord=data.get("mapped_ma3_track_id"),
                sync_status=SyncStatus.SYNCED if data.get("mapped_ma3_track_id") else SyncStatus.UNMAPPED,
                event_count=data.get("event_count", 0),
                settings=SyncLayerSettings.from_dict(data.get("settings", {})),
                group_name=data.get("group_name"),
            )
        
        return None

__all__ = ["SyncSystemManager"]
