"""
OSC Bridge Service

Handles OSC-based bidirectional communication with grandMA3 lighting console.
This is the "brain" side - EchoZero prepares commands, MA3 executes them.

Architecture:
- EchoZero sends OSC commands to MA3 (e.g., /echozero/create_event)
- MA3 sends OSC responses/status to EchoZero (e.g., /ma3/ack, /ma3/status)
- All heavy lifting (data processing, event detection) happens in EchoZero
- MA3 is a thin command executor

OSC Address Scheme:
- /echozero/*  : Commands from EchoZero to MA3
- /ma3/*       : Responses/events from MA3 to EchoZero
"""
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.message import Log

try:
    from pythonosc import udp_client, dispatcher, osc_server
    from pythonosc.osc_message_builder import OscMessageBuilder
    HAS_OSC = True
except ImportError:
    HAS_OSC = False
    Log.warning("python-osc not installed. OSC features disabled. Install with: pip install python-osc")


@dataclass
class OSCConfig:
    """OSC connection configuration."""
    # EchoZero listens on this port for MA3 responses
    listen_port: int = 9000
    listen_address: str = "0.0.0.0"
    
    # MA3's built-in OSC port (default 8000)
    # Configure in MA3: Menu > In & Out > OSC
    ma3_port: int = 8000
    ma3_address: str = "127.0.0.1"


@dataclass
class OSCMessage:
    """Parsed OSC message."""
    address: str
    args: List[Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self):
        return f"{self.address} {self.args}"


class OSCBridgeService:
    """
    Service for OSC-based communication with grandMA3.
    
    Responsibilities:
    - Send OSC commands to MA3 (create events, tracks, etc.)
    - Receive OSC responses from MA3 (acknowledgments, status)
    - Provide high-level API for event export
    
    Thread Safety:
    - Server runs in background thread
    - Callbacks are invoked in server thread (use thread-safe patterns)
    """
    
    def __init__(self, config: Optional[OSCConfig] = None):
        """
        Initialize OSC bridge.
        
        Args:
            config: OSC configuration (uses defaults if None)
        """
        if not HAS_OSC:
            Log.error("OSCBridgeService: python-osc not available")
            self._available = False
            return
        
        self._available = True
        self.config = config or OSCConfig()
        
        # Client for sending to MA3
        self._client: Optional[udp_client.SimpleUDPClient] = None
        
        # Server for receiving from MA3
        self._server: Optional[osc_server.ThreadingOSCUDPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._listening = False
        
        # Message handlers
        self._handlers: Dict[str, Callable[[OSCMessage], None]] = {}
        self._default_handler: Optional[Callable[[OSCMessage], None]] = None
        
        # Message log for debugging
        self._message_log: List[OSCMessage] = []
        self._max_log_size = 100
        
        Log.info(f"OSCBridgeService: Initialized (listen: {self.config.listen_address}:{self.config.listen_port}, "
                 f"send: {self.config.ma3_address}:{self.config.ma3_port})")
    
    @property
    def is_available(self) -> bool:
        """Check if OSC is available (python-osc installed)."""
        return self._available
    
    @property
    def is_listening(self) -> bool:
        """Check if server is listening for MA3 messages."""
        return self._listening
    
    # =========================================================================
    # Sending Commands to MA3
    # =========================================================================
    
    def _ensure_client(self) -> bool:
        """Ensure client is connected."""
        if not self._available:
            return False
        
        if self._client is None:
            try:
                self._client = udp_client.SimpleUDPClient(
                    self.config.ma3_address,
                    self.config.ma3_port
                )
                Log.debug(f"OSCBridgeService: Client created for {self.config.ma3_address}:{self.config.ma3_port}")
            except Exception as e:
                Log.error(f"OSCBridgeService: Failed to create client: {e}")
                return False
        return True
    
    def send(self, address: str, *args) -> bool:
        """
        Send OSC message to MA3.
        
        Args:
            address: OSC address (e.g., "/echozero/ping")
            *args: Message arguments (strings, ints, floats)
            
        Returns:
            True if sent successfully
        """
        if not self._ensure_client():
            return False
        
        try:
            self._client.send_message(address, args if args else [])
            Log.debug(f"OSCBridgeService: Sent {address} {args}")
            return True
        except Exception as e:
            Log.error(f"OSCBridgeService: Send failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # MA3 Built-in OSC Support
    # -------------------------------------------------------------------------
    # MA3 has built-in OSC that uses /cmd to execute commands.
    # This is the recommended approach vs custom Lua socket handling.
    
    def send_lua(self, lua_code: str) -> bool:
        """
        Send Lua code to MA3 via its built-in OSC /cmd address.
        
        MA3's OSC receiver executes the argument as a console command.
        We wrap Lua code in 'Lua "..."' format.
        
        Args:
            lua_code: Lua code to execute (without Lua"" wrapper)
            
        Returns:
            True if sent successfully
            
        Example:
            send_lua("EZ.Ping()")
            -> Sends: /cmd 'Lua "EZ.Ping()"'
        """
        # Escape any quotes in the Lua code
        escaped_code = lua_code.replace('"', '\\"')
        cmd = f'Lua "{escaped_code}"'
        return self.send("/cmd", cmd)
    
    def send_ma3_command(self, command: str) -> bool:
        """
        Send a console command to MA3 via OSC.
        
        Args:
            command: MA3 console command (e.g., "Go Executor 1")
            
        Returns:
            True if sent successfully
        """
        return self.send("/cmd", command)
    
    # -------------------------------------------------------------------------
    # High-Level Commands (EchoZero -> MA3) - Uses EZ Spine
    # -------------------------------------------------------------------------
    
    def ping(self) -> bool:
        """Send ping to MA3. Expects /ma3/pong response."""
        return self.send_lua("EZ.Ping()")
    
    def echo(self, message: str) -> bool:
        """Send echo request. MA3 should echo back the message."""
        escaped = message.replace("'", "\\'")
        return self.send_lua(f"EZ.Echo('{escaped}')")
    
    def status_request(self) -> bool:
        """Request status from MA3."""
        return self.send_lua("EZ.Status()")
    
    def create_track_group(self, timecode_no: int, name: str) -> bool:
        """
        Create a track group in MA3 timecode.
        
        Args:
            timecode_no: Timecode pool number
            name: Name for the track group
        """
        escaped_name = name.replace("'", "\\'")
        return self.send_lua(f"EZ.AddTrackGroup({timecode_no}, '{escaped_name}')")
    
    def create_track(self, timecode_no: int, track_group_idx: int, name: str) -> bool:
        """
        Create a track in MA3.
        
        Args:
            timecode_no: Timecode pool number
            track_group_idx: Track group index
            name: Name for the track
        """
        escaped_name = name.replace("'", "\\'")
        return self.send_lua(f"EZ.AddTrack({timecode_no}, {track_group_idx}, '{escaped_name}')")
    
    def create_event(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int,
        time_seconds: float,
        event_type: str = "cmd",
        classification: str = "",
        properties: Optional[str] = None
    ) -> bool:
        """
        Create an event in MA3 timecode track.
        
        Args:
            timecode_no: Timecode pool number
            track_group_idx: Track group index
            track_idx: Track index within group
            time_seconds: Event time in seconds
            event_type: Event type ("cmd" or "fader")
            classification: Event classification/label
            properties: JSON string of additional properties (not yet used)
        """
        escaped_class = classification.replace("'", "\\'")
        lua_code = f"EZ.AddEvent({timecode_no}, {track_group_idx}, {track_idx}, {time_seconds}, '{escaped_class}', '{event_type}')"
        return self.send_lua(lua_code)
    
    def delete_event_by_index(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int,
        event_idx: int
    ) -> bool:
        """Delete an event by index."""
        return self.send_lua(f"EZ.DeleteEvent({timecode_no}, {track_group_idx}, {track_idx}, {event_idx})")
    
    def get_timecodes(self) -> bool:
        """Request list of all timecodes from MA3."""
        return self.send_lua("EZ.GetTimecodes()")
    
    def get_track_groups(self, timecode_no: int) -> bool:
        """Request track groups for a timecode."""
        return self.send_lua(f"EZ.GetTrackGroups({timecode_no})")
    
    def get_tracks(self, timecode_no: int, track_group_idx: int) -> bool:
        """Request tracks in a track group."""
        return self.send_lua(f"EZ.GetTracks({timecode_no}, {track_group_idx})")
    
    def get_events(self, timecode_no: int, track_group_idx: int, track_idx: int) -> bool:
        """Request events in a specific track."""
        return self.send_lua(f"EZ.GetEvents({timecode_no}, {track_group_idx}, {track_idx})")
    
    def get_all_events(self, timecode_no: int) -> bool:
        """Request all events in a timecode."""
        return self.send_lua(f"EZ.GetAllEvents({timecode_no})")
    
    def batch_start(self, batch_id: str, total_count: int) -> bool:
        """Signal start of batch operation (for progress tracking)."""
        Log.info(f"OSCBridgeService: Starting batch '{batch_id}' with {total_count} items")
        # Could send a Lua call here if needed
        return True
    
    def batch_end(self, batch_id: str) -> bool:
        """Signal end of batch operation."""
        Log.info(f"OSCBridgeService: Batch '{batch_id}' complete")
        return True
    
    # -------------------------------------------------------------------------
    # Extended Commands (Phase 1 MA3 Bridge)
    # -------------------------------------------------------------------------
    
    def add_event(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int,
        time: float,
        event_type: str = "cmd",
        properties_json: str = "{}"
    ) -> bool:
        """
        Add event to MA3 track.
        
        Args:
            timecode_no: Timecode pool number
            track_group_idx: Track group index
            track_idx: Track index
            time: Event time in seconds
            event_type: "cmd" or "fader"
            properties_json: JSON string of properties
        """
        return self.send(
            "/echozero/timecode/add_event",
            timecode_no, track_group_idx, track_idx, time, event_type, properties_json
        )
    
    def move_event(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int,
        event_idx: int,
        new_time: float
    ) -> bool:
        """Move event to new time."""
        return self.send(
            "/echozero/timecode/move_event",
            timecode_no, track_group_idx, track_idx, event_idx, new_time
        )
    
    def delete_event(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int,
        event_idx: int
    ) -> bool:
        """Delete event from track."""
        return self.send(
            "/echozero/timecode/delete_event",
            timecode_no, track_group_idx, track_idx, event_idx
        )
    
    def update_event(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int,
        event_idx: int,
        updates_json: str
    ) -> bool:
        """Update event properties."""
        return self.send(
            "/echozero/timecode/update_event",
            timecode_no, track_group_idx, track_idx, event_idx, updates_json
        )
    
    def add_track(
        self,
        timecode_no: int,
        track_group_idx: int,
        name: str
    ) -> bool:
        """Add track to track group."""
        return self.send(
            "/echozero/timecode/add_track",
            timecode_no, track_group_idx, name
        )
    
    def delete_track(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int
    ) -> bool:
        """Delete track from track group."""
        return self.send(
            "/echozero/timecode/delete_track",
            timecode_no, track_group_idx, track_idx
        )
    
    def rename_track(
        self,
        timecode_no: int,
        track_group_idx: int,
        track_idx: int,
        new_name: str
    ) -> bool:
        """Rename track."""
        return self.send(
            "/echozero/timecode/rename_track",
            timecode_no, track_group_idx, track_idx, new_name
        )
    
    def add_track_group(
        self,
        timecode_no: int,
        name: str
    ) -> bool:
        """Add track group to timecode."""
        return self.send(
            "/echozero/timecode/add_trackgroup",
            timecode_no, name
        )
    
    def query_timecode(self, timecode_no: int) -> bool:
        """Query timecode structure."""
        return self.send("/echozero/query/timecode", timecode_no)
    
    def query_tracks(self, timecode_no: int, track_group_idx: int) -> bool:
        """Query tracks in track group."""
        return self.send("/echozero/query/tracks", timecode_no, track_group_idx)
    
    def query_events(self, timecode_no: int, track_group_idx: int, track_idx: int) -> bool:
        """Query events in track."""
        return self.send("/echozero/query/events", timecode_no, track_group_idx, track_idx)
    
    # -------------------------------------------------------------------------
    # Change Notification Handlers
    # -------------------------------------------------------------------------
    
    def register_change_handlers(self, event_bus=None):
        """
        Register handlers for MA3 change notifications.
        
        When MA3 notifies of changes, these handlers emit domain events
        through the event bus for UI/block updates.
        
        Args:
            event_bus: Optional EventBus for emitting domain events
        """
        self._event_bus = event_bus
        
        # Event changes
        self.register_handler("/ma3/change/event_added", self._handle_event_added)
        self.register_handler("/ma3/change/event_moved", self._handle_event_moved)
        self.register_handler("/ma3/change/event_deleted", self._handle_event_deleted)
        self.register_handler("/ma3/change/event_updated", self._handle_event_updated)
        
        # Track changes
        self.register_handler("/ma3/change/track_added", self._handle_track_added)
        self.register_handler("/ma3/change/track_deleted", self._handle_track_deleted)
        self.register_handler("/ma3/change/track_renamed", self._handle_track_renamed)
        
        # Acknowledgments
        self.register_handler("/ma3/ack", self._handle_ack)
        self.register_handler("/ma3/error", self._handle_error)
        
        # Query responses
        self.register_handler("/ma3/query/timecode", self._handle_query_timecode)
        self.register_handler("/ma3/query/tracks", self._handle_query_tracks)
        self.register_handler("/ma3/query/events", self._handle_query_events)
        
        Log.info("OSCBridgeService: Registered MA3 change notification handlers")
    
    def _emit_event(self, event_name: str, data: Dict[str, Any]):
        """Emit domain event through event bus."""
        if hasattr(self, '_event_bus') and self._event_bus:
            try:
                self._event_bus.publish(event_name, data)
            except Exception as e:
                Log.warning(f"OSCBridgeService: Failed to emit event: {e}")
    
    def _handle_event_added(self, msg: OSCMessage):
        """Handle event added notification from MA3."""
        Log.debug(f"OSCBridgeService: MA3 event added: {msg.args}")
        self._emit_event("MA3EventAdded", {"args": msg.args})
    
    def _handle_event_moved(self, msg: OSCMessage):
        """Handle event moved notification from MA3."""
        Log.debug(f"OSCBridgeService: MA3 event moved: {msg.args}")
        self._emit_event("MA3EventMoved", {"args": msg.args})
    
    def _handle_event_deleted(self, msg: OSCMessage):
        """Handle event deleted notification from MA3."""
        Log.debug(f"OSCBridgeService: MA3 event deleted: {msg.args}")
        self._emit_event("MA3EventDeleted", {"args": msg.args})
    
    def _handle_event_updated(self, msg: OSCMessage):
        """Handle event updated notification from MA3."""
        Log.debug(f"OSCBridgeService: MA3 event updated: {msg.args}")
        self._emit_event("MA3EventUpdated", {"args": msg.args})
    
    def _handle_track_added(self, msg: OSCMessage):
        """Handle track added notification from MA3."""
        Log.debug(f"OSCBridgeService: MA3 track added: {msg.args}")
        self._emit_event("MA3TrackAdded", {"args": msg.args})
    
    def _handle_track_deleted(self, msg: OSCMessage):
        """Handle track deleted notification from MA3."""
        Log.debug(f"OSCBridgeService: MA3 track deleted: {msg.args}")
        self._emit_event("MA3TrackDeleted", {"args": msg.args})
    
    def _handle_track_renamed(self, msg: OSCMessage):
        """Handle track renamed notification from MA3."""
        Log.debug(f"OSCBridgeService: MA3 track renamed: {msg.args}")
        self._emit_event("MA3TrackRenamed", {"args": msg.args})
    
    def _handle_ack(self, msg: OSCMessage):
        """Handle acknowledgment from MA3."""
        Log.debug(f"OSCBridgeService: MA3 ack: {msg.args}")
        if len(msg.args) >= 2:
            address = msg.args[0]
            status = msg.args[1]
            message = msg.args[2] if len(msg.args) > 2 else ""
            self._emit_event("MA3Ack", {
                "address": address,
                "status": status,
                "message": message
            })
    
    def _handle_error(self, msg: OSCMessage):
        """Handle error from MA3."""
        Log.warning(f"OSCBridgeService: MA3 error: {msg.args}")
        error_message = msg.args[0] if msg.args else "Unknown error"
        self._emit_event("MA3Error", {"message": error_message})
    
    def _handle_query_timecode(self, msg: OSCMessage):
        """Handle timecode query response from MA3."""
        Log.debug(f"OSCBridgeService: MA3 timecode query result: {msg.args}")
        self._emit_event("MA3TimecodeQueryResult", {"args": msg.args})
    
    def _handle_query_tracks(self, msg: OSCMessage):
        """Handle tracks query response from MA3."""
        Log.debug(f"OSCBridgeService: MA3 tracks query result: {msg.args}")
        self._emit_event("MA3TracksQueryResult", {"args": msg.args})
    
    def _handle_query_events(self, msg: OSCMessage):
        """Handle events query response from MA3."""
        Log.debug(f"OSCBridgeService: MA3 events query result: {msg.args}")
        self._emit_event("MA3EventsQueryResult", {"args": msg.args})
    
    # =========================================================================
    # Receiving Messages from MA3
    # =========================================================================
    
    def start_listening(self) -> bool:
        """
        Start listening for OSC messages from MA3.
        
        Returns:
            True if server started successfully
        """
        if not self._available:
            Log.error("OSCBridgeService: Cannot start - python-osc not available")
            return False
        
        if self._listening:
            Log.warning("OSCBridgeService: Already listening")
            return True
        
        try:
            # Create dispatcher
            disp = dispatcher.Dispatcher()
            
            # Register specific handlers
            for address, handler in self._handlers.items():
                disp.map(address, self._make_osc_callback(handler))
            
            # Default handler for unregistered addresses
            disp.set_default_handler(self._handle_default_message)
            
            # Create server
            self._server = osc_server.ThreadingOSCUDPServer(
                (self.config.listen_address, self.config.listen_port),
                disp
            )
            
            # Start in background thread
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="OSCBridgeServer"
            )
            self._server_thread.start()
            self._listening = True
            
            Log.info(f"OSCBridgeService: Listening on {self.config.listen_address}:{self.config.listen_port}")
            return True
            
        except Exception as e:
            Log.error(f"OSCBridgeService: Failed to start server: {e}")
            return False
    
    def stop_listening(self) -> None:
        """Stop listening for OSC messages."""
        if not self._listening:
            return
        
        if self._server:
            self._server.shutdown()
            self._server = None
        
        self._server_thread = None
        self._listening = False
        Log.info("OSCBridgeService: Stopped listening")
    
    def register_handler(self, address: str, handler: Callable[[OSCMessage], None]) -> None:
        """
        Register handler for specific OSC address.
        
        Args:
            address: OSC address pattern (e.g., "/ma3/pong")
            handler: Callback function(msg: OSCMessage)
        """
        self._handlers[address] = handler
        Log.debug(f"OSCBridgeService: Registered handler for {address}")
    
    def set_default_handler(self, handler: Callable[[OSCMessage], None]) -> None:
        """Set handler for messages without specific handler."""
        self._default_handler = handler
    
    def _make_osc_callback(self, handler: Callable[[OSCMessage], None]):
        """Create pythonosc callback wrapper."""
        def callback(address: str, *args):
            msg = OSCMessage(address=address, args=list(args))
            self._log_message(msg)
            try:
                handler(msg)
            except Exception as e:
                Log.error(f"OSCBridgeService: Handler error for {address}: {e}")
        return callback
    
    def _handle_default_message(self, address: str, *args):
        """Handle messages without specific handler."""
        msg = OSCMessage(address=address, args=list(args))
        self._log_message(msg)
        
        if self._default_handler:
            try:
                self._default_handler(msg)
            except Exception as e:
                Log.error(f"OSCBridgeService: Default handler error: {e}")
        else:
            Log.debug(f"OSCBridgeService: Unhandled message: {msg}")
    
    def _log_message(self, msg: OSCMessage) -> None:
        """Log message for debugging."""
        self._message_log.append(msg)
        if len(self._message_log) > self._max_log_size:
            self._message_log.pop(0)
    
    def get_message_log(self) -> List[OSCMessage]:
        """Get recent message log."""
        return self._message_log.copy()
    
    def clear_message_log(self) -> None:
        """Clear message log."""
        self._message_log.clear()
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_listening()
        self._client = None
        Log.info("OSCBridgeService: Cleaned up")


# =============================================================================
# Event Export Helper
# =============================================================================

class EventExporter:
    """
    Exports EchoZero events to MA3 timecode.
    
    This helper converts EventDataItem events to MA3 OSC commands.
    It handles batching, progress reporting, and track organization.
    """
    
    def __init__(self, bridge: OSCBridgeService):
        """
        Initialize exporter.
        
        Args:
            bridge: OSC bridge service for sending commands
        """
        self.bridge = bridge
    
    def export_events(
        self,
        events: List[Any],  # List of Event objects
        timecode_no: int,
        track_group_name: str = "EchoZero",
        group_by_classification: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Export events to MA3 timecode.
        
        Args:
            events: List of Event objects from EventDataItem
            timecode_no: Target MA3 timecode number
            track_group_name: Name for the track group
            group_by_classification: If True, create separate track per classification
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Export result dict with success, counts, errors
        """
        if not self.bridge.is_available:
            return {"success": False, "error": "OSC not available"}
        
        result = {
            "success": True,
            "track_group_name": track_group_name,
            "timecode_no": timecode_no,
            "total_events": len(events),
            "events_sent": 0,
            "tracks_created": set(),
            "errors": []
        }
        
        # Group events by classification if requested
        if group_by_classification:
            grouped: Dict[str, List] = {}
            for event in events:
                classification = getattr(event, 'classification', '') or 'default'
                if classification not in grouped:
                    grouped[classification] = []
                grouped[classification].append(event)
        else:
            grouped = {"events": events}
        
        # Start batch
        import uuid
        batch_id = str(uuid.uuid4())[:8]
        self.bridge.batch_start(batch_id, len(events))
        
        # Create track group
        self.bridge.create_track_group(timecode_no, track_group_name)
        
        # Create tracks and events
        track_idx = 1  # Start at 1 (0 is Marker in MA3)
        event_count = 0
        
        for classification, event_list in grouped.items():
            track_name = classification if classification else "default"
            result["tracks_created"].add(track_name)
            
            # Create track
            self.bridge.create_track(timecode_no, 1, track_name)
            
            # Create events
            for event in event_list:
                time = getattr(event, 'time', 0)
                success = self.bridge.create_event(
                    timecode_no=timecode_no,
                    track_group_idx=1,
                    track_idx=track_idx,
                    time_seconds=time,
                    event_type="CmdEvent"
                )
                
                if success:
                    result["events_sent"] += 1
                else:
                    result["errors"].append(f"Failed to send event at {time}s")
                
                event_count += 1
                if progress_callback:
                    progress_callback(event_count, len(events))
            
            track_idx += 1
        
        # End batch
        self.bridge.batch_end(batch_id)
        
        result["tracks_created"] = list(result["tracks_created"])
        result["success"] = len(result["errors"]) == 0
        
        return result
