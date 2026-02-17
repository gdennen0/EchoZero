"""
MA3 Communication Service

Handles bidirectional communication with grandMA3 lighting console.
Listens for UDP messages from MA3 plugins and can send responses.

Protocol:
- MA3 plugin sends OSC messages to EchoZero (default port 9000)
- Messages use address /ez/message with a string argument
- String argument is pipe-delimited: type=X|change=Y|timestamp=Z|...
- JSON data (arrays, objects) is encoded in field values
"""
import socket
import threading
import struct
import json
import time
from typing import Optional, Callable, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.message import Log

from src.application.events.event_bus import EventBus

@dataclass
class MA3Message:
    """Parsed message from grandMA3"""
    object_type: str
    change_type: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Convenience accessors for common fields
    @property
    def tc(self) -> Optional[int]:
        """Timecode number"""
        val = self.data.get('tc')
        return int(val) if val is not None else None
    
    @property
    def tg(self) -> Optional[int]:
        """Track group number"""
        val = self.data.get('tg')
        return int(val) if val is not None else None
    
    @property
    def track(self) -> Optional[int]:
        """Track number"""
        val = self.data.get('track')
        return int(val) if val is not None else None
    
    @property
    def events(self) -> List[Dict[str, Any]]:
        """Event list (if present)"""
        return self.data.get('events', [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'object_type': self.object_type,
            'change_type': self.change_type,
            'timestamp': self.timestamp,
            **self.data
        }

class MA3CommunicationService:
    """
    Service for communicating with grandMA3 console.
    
    Listens for UDP messages from MA3 plugins and publishes events.
    Can send messages back to MA3 (future: bidirectional communication).
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        listen_port: int = 9000,
        listen_address: str = "127.0.0.1",
        send_port: int = 9001,
        send_address: str = "127.0.0.1"
    ):
        """
        Initialize MA3 communication service.
        
        Args:
            event_bus: Event bus for publishing MA3 events
            listen_port: UDP port to listen on (default: 9000)
            listen_address: IP address to bind to (default: 127.0.0.1)
            send_port: UDP port to send to MA3 (default: 9001)
            send_address: IP address to send to (default: 127.0.0.1)
        """
        self.event_bus = event_bus
        self.listen_port = listen_port
        self.listen_address = listen_address
        self.send_port = send_port
        self.send_address = send_address
        
        self._socket: Optional[socket.socket] = None
        self._listening = False
        self._listener_thread: Optional[threading.Thread] = None
        self._message_handlers: Dict[str, Callable[[MA3Message], None]] = {}
        self._last_message_time: Optional[float] = None
        self._last_message_addr: Optional[Tuple[str, int]] = None
        
        Log.info(f"MA3CommunicationService: Created (not started - will start when block connects to EchoZero)")
    
    def start_listening(self) -> bool:
        """
        Start listening for UDP messages from MA3.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._listening:
            Log.warning("MA3CommunicationService: Already listening")
            return False
        
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Increase receive buffer to 1MB to handle burst UDP traffic
            # from multi-track sync (N tracks * chunked events can produce
            # 25+ messages in rapid succession)
            try:
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
                actual_buf = self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                Log.info(f"MA3CommunicationService: SO_RCVBUF set to {actual_buf} bytes")
            except Exception as buf_err:
                Log.warning(f"MA3CommunicationService: Could not set SO_RCVBUF: {buf_err}")
            self._socket.bind((self.listen_address, self.listen_port))
            self._socket.settimeout(1.0)  # Allow periodic checks for shutdown
            
            self._listening = True
            self._listener_thread = threading.Thread(
                target=self._listen_loop,
                daemon=True,
                name="MA3Listener"
            )
            self._listener_thread.start()
            
            Log.info(f"MA3CommunicationService: Started listening on {self.listen_address}:{self.listen_port}")
            return True
            
        except Exception as e:
            Log.error(f"MA3CommunicationService: Failed to start listening: {e}")
            self._listening = False
            return False
    
    def stop_listening(self) -> None:
        """Stop listening for UDP messages."""
        if not self._listening:
            return
        
        self._listening = False
        
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                Log.error(f"MA3CommunicationService: Error closing socket: {e}")
            self._socket = None
        
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        
        Log.info("MA3CommunicationService: Stopped listening")

    def set_listen_port(self, listen_port: int) -> None:
        """Update listen port for inbound OSC."""
        if listen_port:
            self.listen_port = listen_port

    def set_listen_address(self, listen_address: str) -> None:
        """Update listen address/interface for inbound OSC."""
        if listen_address:
            self.listen_address = listen_address
    
    def _listen_loop(self) -> None:
        """Main listening loop (runs in background thread)."""
        while self._listening:
            try:
                if not self._socket:
                    break
                
                data, addr = self._socket.recvfrom(65536)  # Larger buffer for JSON data
                
                if data:
                    self._last_message_time = time.time()
                    self._last_message_addr = (addr[0], addr[1])
                    # Try to parse as OSC first
                    address, args = self._parse_osc_message(data)
                    
                    if address:
                        # It's an OSC message
                        self._handle_osc_message(address, args, addr, data)
                    else:
                        # Fallback: try as plain text (backwards compatibility)
                        message_str = data.decode('utf-8', errors='ignore').strip()
                        if message_str:
                            self._handle_message(message_str, addr)
                    
            except socket.timeout:
                # Expected - allows periodic checks for shutdown
                continue
            except OSError as e:
                # Socket closed or error
                if self._listening:
                    Log.error("MA3CommunicationService: Socket error in listen loop")
                break
            except Exception as e:
                Log.error(f"MA3CommunicationService: Error in listen loop: {e}")
    
    def _parse_osc_string(self, data: bytes, offset: int) -> Tuple[str, int]:
        """Parse a null-terminated, 4-byte aligned OSC string."""
        end = offset
        while end < len(data) and data[end] != 0:
            end += 1
        s = data[offset:end].decode('utf-8', errors='ignore')
        # Align to 4 bytes
        offset = end + 1
        offset += (4 - offset % 4) % 4
        return s, offset
    
    def _parse_osc_message(self, data: bytes) -> Tuple[str, List[Any]]:
        """Parse OSC message, return (address, args)."""
        try:
            if len(data) < 4:
                return "", []
            
            # Parse address
            address, offset = self._parse_osc_string(data, 0)
            if offset >= len(data):
                return address, []
            
            # Parse type tag
            type_tag, offset = self._parse_osc_string(data, offset)
            if not type_tag.startswith(','):
                return address, []
            
            args = []
            for t in type_tag[1:]:
                if offset >= len(data):
                    break
                if t == 's':
                    arg, offset = self._parse_osc_string(data, offset)
                    args.append(arg)
                elif t == 'i':
                    if offset + 4 > len(data):
                        break
                    args.append(struct.unpack('>i', data[offset:offset+4])[0])
                    offset += 4
                elif t == 'f':
                    if offset + 4 > len(data):
                        break
                    args.append(struct.unpack('>f', data[offset:offset+4])[0])
                    offset += 4
            
            return address, args
        except Exception as e:
            Log.warning(f"MA3CommunicationService: OSC parse error: {e}")
            return "", []
    
    def _handle_osc_message(self, address: str, args: List[Any], addr: tuple, raw_data: bytes) -> None:
        """Handle parsed OSC message from MA3."""
        Log.debug(f"MA3CommunicationService: OSC from {addr[0]}:{addr[1]}: {address} {args}")
        
        # Publish raw OSC for UI consumption
        try:
            from src.application.events.events import MA3OscInbound
            self.event_bus.publish(MA3OscInbound(data={
                "address": address,
                "args": args,
                "addr": addr,
                "raw_data": raw_data
            }))
        except Exception:
            pass
        
        if address == '/ez/message' and args:
            # Main message format: pipe-delimited string
            self._handle_message(str(args[0]), addr)
        elif address == '/ez/ping':
            Log.info("MA3CommunicationService: Received ping from MA3")
            # Could send pong back
        else:
            Log.debug(f"MA3CommunicationService: Unhandled OSC address: {address}")
    
    def _handle_message(self, message_str: str, addr: tuple) -> None:
        """
        Parse and handle incoming message from MA3.
        
        Args:
            message_str: Raw message string (pipe-delimited format)
            addr: Source address (host, port)
        
        Message format:
            type=events|change=list|timestamp=123|tc=101|count=5|events=[{...}]
        """
        try:
            # Parse pipe-delimited format from MA3 plugin
            parts = message_str.split('|')
            data = {}
            
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to parse JSON values (arrays, objects)
                    if value.startswith('[') or value.startswith('{'):
                        try:
                            data[key] = json.loads(value)
                        except json.JSONDecodeError:
                            data[key] = value
                    # Try to parse numbers
                    elif value.isdigit():
                        data[key] = int(value)
                    elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = value
                    # Boolean
                    elif value.lower() in ('true', 'false'):
                        data[key] = value.lower() == 'true'
                    else:
                        data[key] = value
            
            # Extract main fields
            object_type = str(data.pop('type', 'unknown'))
            change_type = str(data.pop('change', 'unknown'))
            timestamp = float(data.pop('timestamp', datetime.now().timestamp()))
            
            
            message = MA3Message(
                object_type=object_type,
                change_type=change_type,
                timestamp=timestamp,
                data=data
            )
            
            Log.debug(f"MA3CommunicationService: Message from {addr[0]}:{addr[1]}: {object_type}.{change_type}")
            
            # Call registered handler if exists
            handler_key = f"{object_type}.{change_type}"
            if handler_key in self._message_handlers:
                try:
                    self._message_handlers[handler_key](message)
                except Exception as e:
                    Log.error(f"MA3CommunicationService: Error in handler for {handler_key}: {e}")
            
            # Also try wildcard handlers (e.g., "event.*" or "*.changed")
            for pattern, handler in self._message_handlers.items():
                if '*' in pattern:
                    pattern_type, pattern_change = pattern.split('.', 1)
                    if (pattern_type == '*' or pattern_type == object_type) and \
                       (pattern_change == '*' or pattern_change == change_type):
                        if pattern != handler_key:  # Don't call twice
                            try:
                                handler(message)
                            except Exception as e:
                                Log.error(f"MA3CommunicationService: Error in wildcard handler {pattern}: {e}")
            
            # Always publish event for general consumption
            self._publish_ma3_event(message, raw_message=message_str)
            
        except Exception as e:
            Log.error(f"MA3CommunicationService: Error parsing message '{message_str[:100]}...': {e}")
    
    def _publish_ma3_event(self, message: MA3Message, raw_message: Optional[str] = None) -> None:
        """
        Publish MA3 message as domain event.
        
        Args:
            message: Parsed MA3 message
        """
        from src.application.events import MA3MessageReceived
        
        event = MA3MessageReceived(
            data={
                'object_type': message.object_type,
                'change_type': message.change_type,
                'timestamp': message.timestamp,
                'ma3_data': message.data,
                'raw_message': raw_message
            }
        )
        self.event_bus.publish(event)
    
    def is_connected(self) -> bool:
        """Check if service is listening and presumably connected."""
        return self._listening

    def last_message_time(self) -> Optional[float]:
        """Return last inbound message time (epoch seconds)."""
        return self._last_message_time

    def last_message_addr(self) -> Optional[Tuple[str, int]]:
        """Return last inbound message sender address."""
        return self._last_message_addr
    
    def register_handler(
        self,
        object_type: str,
        change_type: str,
        handler: Callable[[MA3Message], None]
    ) -> None:
        """
        Register a handler for specific message types.
        
        Args:
            object_type: Type of MA3 object (e.g., "sequence", "cue")
            change_type: Type of change (e.g., "changed", "created", "deleted")
            handler: Callback function that receives MA3Message
        """
        handler_key = f"{object_type}.{change_type}"
        self._message_handlers[handler_key] = handler
        Log.debug(f"MA3CommunicationService: Registered handler for {handler_key}")
    
    def send_message(self, message: str, target_ip: Optional[str] = None, target_port: Optional[int] = None) -> bool:
        """
        Send UDP message to MA3.
        
        Args:
            message: Message string to send
            target_ip: Target IP address (defaults to configured send_address)
            target_port: Target port number (defaults to configured send_port)
            
        Returns:
            True if sent successfully, False otherwise
        """
        ip = target_ip or self.send_address
        port = target_port or self.send_port
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(message.encode('utf-8'), (ip, port))
            sock.close()
            Log.info(f"MA3CommunicationService: Sent message to {ip}:{port}: {message[:50]}")
            return True
        except Exception as e:
            Log.error(f"MA3CommunicationService: Error sending message: {e}")
            return False
    
    def send_ping(self) -> bool:
        """Send a ping message to MA3 to test connection."""
        return self.send_lua_command('EZ.Ping()')
    
    def send_lua_command(self, lua_code: str, target_ip: Optional[str] = None, target_port: Optional[int] = None) -> bool:
        """
        Send a Lua command to MA3 via OSC.
        
        Args:
            lua_code: Lua code to execute (e.g., 'EZ.GetEvents(101, 1, 1)')
            target_ip: Target IP address (defaults to configured send_address)
            target_port: Target port number (defaults to configured send_port)
            
        Returns:
            True if sent successfully, False otherwise
        """
        ip = target_ip or self.send_address
        port = target_port or self.send_port
        
        # Ensure lua_code is a string (could be None or other type in edge cases)
        if not isinstance(lua_code, str):
            Log.error(f"MA3CommunicationService: Invalid lua_code type: {type(lua_code)}, value: {lua_code}")
            try:
                lua_code = str(lua_code) if lua_code is not None else ""
            except Exception as e:
                Log.error(f"MA3CommunicationService: Failed to convert lua_code to string: {e}")
                return False
        
        # Build OSC message with /cmd address and Lua command
        # Use single quotes so we can pass double quotes untouched.
        # Escape single quotes and backslashes inside the Lua code payload.
        try:
            lua_code_escaped = lua_code.replace("\\", "\\\\").replace("'", "\\'")
        except (AttributeError, TypeError) as e:
            Log.error(f"MA3CommunicationService: Error escaping lua_code: {e}, type: {type(lua_code)}, value: {lua_code}")
            return False
        
        cmd = f"Lua '{lua_code_escaped}'"
        osc_data = self._build_osc_message('/cmd', 's', cmd)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(osc_data, (ip, port))
            sock.close()
            # Publish outbound OSC to Monitoring tab
            if self.event_bus:
                try:
                    from src.application.events.events import MA3OscOutbound
                    self.event_bus.publish(MA3OscOutbound(data={
                        "ip": ip,
                        "port": port,
                        "lua_code": lua_code,
                        "osc_len": len(osc_data),
                        "success": True
                    }))
                except Exception:
                    pass
            Log.debug(f"MA3CommunicationService: Sent Lua command to {ip}:{port}: {lua_code}")
            return True
        except Exception as e:
            if self.event_bus:
                try:
                    from src.application.events.events import MA3OscOutbound
                    self.event_bus.publish(MA3OscOutbound(data={
                        "ip": ip,
                        "port": port,
                        "lua_code": lua_code,
                        "osc_len": len(osc_data) if 'osc_data' in locals() else None,
                        "success": False,
                        "error": str(e)
                    }))
                except Exception:
                    pass
            Log.error(f"MA3CommunicationService: Error sending Lua command: {e}")
            return False
    
    def _build_osc_message(self, address: str, types: str, *args) -> bytes:
        """Build an OSC message."""
        def osc_string(s: str) -> bytes:
            encoded = s.encode('utf-8') + b'\x00'
            padding = (4 - len(encoded) % 4) % 4
            return encoded + b'\x00' * padding
        
        data = osc_string(address) + osc_string(',' + types)
        
        for i, t in enumerate(types):
            if i >= len(args):
                break
            v = args[i]
            if t == 's':
                data += osc_string(str(v))
            elif t == 'i':
                data += struct.pack('>i', int(v))
            elif t == 'f':
                data += struct.pack('>f', float(v))
        
        return data

    def set_target(self, ip: str, port: int) -> None:
        """Update default send target for outbound OSC."""
        if ip:
            self.send_address = ip
        if port:
            self.send_port = port
    
    # Convenience methods for EchoZero Lua commands
    def get_timecodes(self) -> bool:
        """Request list of timecodes from MA3."""
        return self.send_lua_command('EZ.GetTimecodes()')
    
    def get_track_groups(self, timecode_no: int) -> bool:
        """Request track groups for a timecode."""
        return self.send_lua_command(f'EZ.GetTrackGroups({timecode_no})')
    
    def get_tracks(self, timecode_no: int, track_group: int) -> bool:
        """Request tracks in a track group."""
        return self.send_lua_command(f'EZ.GetTracks({timecode_no}, {track_group})')
    
    def get_events(self, timecode_no: int, track_group: int, track: int, request_id: Optional[int] = None) -> bool:
        """Request events in a track."""
        if request_id is None:
            lua = f'EZ.GetEvents({timecode_no}, {track_group}, {track})'
        else:
            lua = f'EZ.GetEvents({timecode_no}, {track_group}, {track}, {request_id})'
        result = self.send_lua_command(lua)
        return result
    
    def get_all_events(self, timecode_no: int) -> bool:
        """Request all events in a timecode."""
        return self.send_lua_command(f'EZ.GetAllEvents({timecode_no})')
    
    def hook_track(self, timecode_no: int, track_group: int, track: int) -> bool:
        """Subscribe to changes on a track."""
        return self.send_lua_command(f'EZ.HookTrack({timecode_no}, {track_group}, {track})')
    
    def hook_cmdsubtrack(self, timecode_no: int, track_group: int, track: int, time_range_idx: int = 1) -> bool:
        """
        Subscribe to changes on a track's CmdSubTrack (recommended).
        
        Hooks the CmdSubTrack object instead of individual events, which is
        more reliable and avoids MA3 hook limits.
        
        When events change (add/delete/update), the Lua plugin detects changes
        using fingerprint-based comparison and sends change notifications via OSC.
        
        Args:
            timecode_no: Timecode number (e.g., 101)
            track_group: Track group number (e.g., 1)
            track: Track number (user-visible, 1-based, Lua adds offset for Marker track)
            time_range_idx: TimeRange index (default: 1, which is standard for new tracks)
            
        Returns:
            True if command was sent successfully
        """
        return self.send_lua_command(f'EZ.HookCmdSubTrack({timecode_no}, {track_group}, {track}, {time_range_idx})')

    def hook_track_group_changes(self, timecode_no: int, track_group: int) -> bool:
        """Subscribe to changes on a track group (track add/delete/reorder)."""
        return self.send_lua_command(f'EZ.HookTrackGroupChanges({timecode_no}, {track_group})')
    
    def unhook_track(self, timecode_no: int, track_group: int, track: int) -> bool:
        """Unsubscribe from track changes."""
        return self.send_lua_command(f'EZ.UnhookTrack({timecode_no}, {track_group}, {track})')

    def unhook_track_group_changes(self, timecode_no: int, track_group: int) -> bool:
        """Unsubscribe from track group change notifications."""
        return self.send_lua_command(f'EZ.UnhookTrackGroupChanges({timecode_no}, {track_group})')
    
    def rehook_cmdsubtrack(self, timecode_no: int, track_group: int, track: int, time_range_idx: int = 1) -> bool:
        """
        Re-hook a track by unhooking first, then hooking.
        
        This is useful for resync operations where we want to ensure a fresh
        hook is established and current events are re-sent from MA3.
        
        Args:
            timecode_no: Timecode number (e.g., 101)
            track_group: Track group number (e.g., 1)
            track: Track number (user-visible, 1-based)
            time_range_idx: TimeRange index (default: 1, which is standard for new tracks)
            
        Returns:
            True if command was sent successfully
        """
        return self.send_lua_command(f'EZ.RehookCmdSubTrack({timecode_no}, {track_group}, {track}, {time_range_idx})')
    
    def unhook_all(self) -> bool:
        """Unsubscribe from all track changes."""
        return self.send_lua_command('EZ.UnhookAll()')
    
    def add_event(self, timecode_no: int, track_group: int, track: int, time: float, cmd: str) -> bool:
        """Add an event in MA3."""
        # Escape quotes in command
        cmd_escaped = cmd.replace('"', '\\"')
        return self.send_lua_command(f'EZ.AddEvent({timecode_no}, {track_group}, {track}, {time}, "{cmd_escaped}")')
    
    def delete_event(self, timecode_no: int, track_group: int, track: int, event_idx: int) -> bool:
        """Delete an event in MA3."""
        return self.send_lua_command(f'EZ.DeleteEvent({timecode_no}, {track_group}, {track}, {event_idx})')
    
    def update_event(self, timecode_no: int, track_group: int, track: int, 
                     event_idx: int, time: float = None, cmd: str = None) -> bool:
        """
        Update an event's time and/or command in MA3.
        
        Args:
            timecode_no: Timecode number
            track_group: Track group number
            track: Track number
            event_idx: Event index within track
            time: New time (optional, None to keep current)
            cmd: New command (optional, None to keep current)
            
        Returns:
            True if command was sent successfully
        """
        # Build Lua call - UpdateEvent(tc, tg, track, idx, time, cmd)
        # If time or cmd is None, pass nil to Lua
        time_arg = str(time) if time is not None else "nil"
        if cmd is not None:
            cmd_escaped = cmd.replace('"', '\\"')
            cmd_arg = f'"{cmd_escaped}"'
        else:
            cmd_arg = "nil"
        return self.send_lua_command(
            f'EZ.UpdateEvent({timecode_no}, {track_group}, {track}, {event_idx}, {time_arg}, {cmd_arg})'
        )
    
    def clear_track(self, timecode_no: int, track_group: int, track: int) -> bool:
        """Clear all events in a track."""
        return self.send_lua_command(f'EZ.ClearTrack({timecode_no}, {track_group}, {track})')

    def create_track(self, timecode_no: int, track_group: int, name: str) -> bool:
        """Create a new track in MA3."""
        name_escaped = (name or "").replace('"', '\\"')
        return self.send_lua_command(f'EZ.CreateTrack({timecode_no}, {track_group}, "{name_escaped}")')

    def assign_track_sequence(self, timecode_no: int, track_group: int, track: int, sequence_no: int) -> bool:
        """Assign a sequence to a track in MA3."""
        seq = int(sequence_no or 1)
        return self.send_lua_command(f'EZ.AssignTrackSequence({timecode_no}, {track_group}, {track}, {seq})')
    
    def set_cmd_mode(self, mode: str = "feedback") -> bool:
        """
        Set the default command execution mode on the MA3 Lua plugin.
        
        Controls whether MA3 commands executed by EZ.RunCommand() use
        Cmd() (shows in console, returns result) or CmdIndirect() (clean console).
        
        Args:
            mode: "feedback" for Cmd() or "silent" for CmdIndirect()
            
        Returns:
            True if command was sent successfully
        """
        if mode not in ("feedback", "silent"):
            Log.error(f"MA3CommunicationService: Invalid cmd mode '{mode}' (use 'feedback' or 'silent')")
            return False
        return self.send_lua_command(f'EZ.SetCmdMode("{mode}")')

    def run_ma3_command(self, cmd: str, silent: bool = False) -> bool:
        """
        Execute an MA3 command via the Lua plugin's EZ.RunCommand() wrapper.
        
        This allows per-call control over whether the command shows in the
        MA3 console (Cmd) or runs silently (CmdIndirect).
        
        Args:
            cmd: The MA3 command string (e.g., 'Store Sequence 1')
            silent: If True, uses CmdIndirect (clean console). 
                    If False, uses Cmd (shows feedback).
                    
        Returns:
            True if the Lua command was sent successfully
        """
        cmd_escaped = cmd.replace('"', '\\"')
        mode = '"silent"' if silent else '"feedback"'
        return self.send_lua_command(f'EZ.RunCommand("{cmd_escaped}", {mode})')

    def is_listening(self) -> bool:
        """Check if service is currently listening."""
        return self._listening

