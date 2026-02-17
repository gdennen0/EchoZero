"""
OSC Parser

Parses raw OSC data into structured messages.
Handles both raw UDP bytes and pipe-delimited string format.
"""

import json
import struct
from typing import Any, Dict, List, Optional, Tuple

from src.features.ma3.domain.osc_message import (
    OSCMessage, 
    MessageType, 
    ChangeType,
    TrackGroupData,
    TrackData,
    EventData,
)
from src.utils.message import Log
from src.utils.paths import get_debug_log_path


class OSCParser:
    """
    Parser for OSC messages from MA3.
    
    Handles:
    - Raw UDP OSC packet parsing (address + type tags + arguments)
    - Pipe-delimited message format (type=X|change=Y|...)
    - JSON embedded in message fields
    
    Usage:
        parser = OSCParser()
        
        # Parse raw UDP packet
        address, args = parser.parse_raw_packet(data)
        
        # Parse pipe-delimited message string
        message = parser.parse_message(args[0])
    """
    
    def parse_raw_packet(self, data: bytes) -> Tuple[str, List[Any]]:
        """
        Parse raw OSC UDP packet into address and arguments.
        
        Args:
            data: Raw UDP packet bytes
            
        Returns:
            Tuple of (address, arguments list)
        """
        try:
            # Extract null-terminated address
            addr_end = data.find(b'\x00')
            if addr_end == -1:
                return "", []
            
            address = data[:addr_end].decode('utf-8')
            
            # Find type tag string (starts with comma)
            # Align to 4-byte boundary after address
            pos = self._align_to_4(addr_end + 1)
            
            if pos >= len(data) or data[pos:pos+1] != b',':
                return address, []
            
            # Extract type tags
            type_end = data.find(b'\x00', pos)
            if type_end == -1:
                return address, []
            
            type_tags = data[pos+1:type_end].decode('utf-8')
            pos = self._align_to_4(type_end + 1)
            
            # Parse arguments based on type tags
            args = []
            for tag in type_tags:
                if pos >= len(data):
                    break
                    
                if tag == 'i':
                    # 32-bit big-endian integer
                    if pos + 4 <= len(data):
                        args.append(struct.unpack('>i', data[pos:pos+4])[0])
                        pos += 4
                elif tag == 'f':
                    # 32-bit big-endian float
                    if pos + 4 <= len(data):
                        args.append(struct.unpack('>f', data[pos:pos+4])[0])
                        pos += 4
                elif tag == 's':
                    # Null-terminated string
                    str_end = data.find(b'\x00', pos)
                    if str_end != -1:
                        args.append(data[pos:str_end].decode('utf-8', errors='replace'))
                        pos = self._align_to_4(str_end + 1)
                elif tag == 'b':
                    # Blob (length-prefixed binary)
                    if pos + 4 <= len(data):
                        blob_len = struct.unpack('>i', data[pos:pos+4])[0]
                        pos += 4
                        if pos + blob_len <= len(data):
                            args.append(data[pos:pos+blob_len])
                            pos = self._align_to_4(pos + blob_len)
            
            return address, args
            
        except Exception as e:
            Log.error(f"OSCParser: Error parsing raw packet: {e}")
            return "", []
    
    def parse_message(self, message_str: str, address: str = "/ez/message") -> Optional[OSCMessage]:
        """
        Parse pipe-delimited message string into structured OSCMessage.
        
        Format: type=X|change=Y|timestamp=Z|field1=value1|field2=value2...
        
        Args:
            message_str: Pipe-delimited message string
            address: OSC address (default: /ez/message)
            
        Returns:
            OSCMessage or None if parsing fails
        """
        if not message_str:
            return None
            
        try:
            # Use smart split that respects JSON structure (doesn't split inside [...] or {...} or "...")
            parts = self._smart_split_pipe(message_str)
            raw_data: Dict[str, Any] = {}
            
            for part in parts:
                if '=' not in part:
                    continue
                    
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse value types
                parsed_value = self._parse_value(value)
                raw_data[key] = parsed_value
            
            # Extract standard fields
            msg_type_str = str(raw_data.pop('type', 'unknown'))
            change_str = str(raw_data.pop('change', 'unknown'))
            timestamp = float(raw_data.pop('timestamp', 0))
            
            # Convert to enums
            try:
                message_type = MessageType(msg_type_str)
            except ValueError:
                message_type = MessageType.UNKNOWN
                
            try:
                change_type = ChangeType(change_str)
            except ValueError:
                change_type = ChangeType.UNKNOWN
                # Log when we can't parse change type (for debugging track.changed)
                if msg_type_str == "track" and change_str == "changed":
                    import json as _json
                    try:
                        open(str(get_debug_log_path()),'a').write(_json.dumps({"hypothesisId":"K","location":"osc_parser.py:parse_message","message":"Failed to parse track.changed","data":{"msg_type":msg_type_str,"change_str":change_str,"available_changes":[ct.value for ct in ChangeType]},"timestamp":__import__('time').time()})+'\n')
                    except:
                        pass
            
            return OSCMessage(
                address=address,
                message_type=message_type,
                change_type=change_type,
                timestamp=timestamp,
                data=raw_data,
                raw=message_str
            )
            
        except Exception as e:
            Log.error(f"OSCParser: Error parsing message: {e}")
            return None
    
    def parse_trackgroups(self, message: OSCMessage) -> List[TrackGroupData]:
        """Parse track groups from message data."""
        result = []
        trackgroups = message.get('trackgroups', [])
        
        if isinstance(trackgroups, str):
            try:
                trackgroups = json.loads(trackgroups)
            except json.JSONDecodeError:
                return result
        
        for tg in trackgroups:
            if isinstance(tg, dict):
                result.append(TrackGroupData(
                    no=tg.get('no', 0),
                    name=tg.get('name', ''),
                    track_count=tg.get('track_count', 0)
                ))
        
        return result
    
    def parse_tracks(self, message: OSCMessage) -> List[TrackData]:
        """Parse tracks from message data."""
        result = []
        tracks = message.get('tracks', [])
        
        if isinstance(tracks, str):
            try:
                tracks = json.loads(tracks)
            except json.JSONDecodeError:
                return result
        
        for track in tracks:
            if isinstance(track, dict):
                result.append(TrackData(
                    no=track.get('no', 0),
                    name=track.get('name', '')
                ))
        
        return result
    
    def parse_events(self, message: OSCMessage) -> List[EventData]:
        """Parse events from message data."""
        result = []
        events = message.get('events', [])
        
        if isinstance(events, str):
            try:
                events = json.loads(events)
            except json.JSONDecodeError:
                return result
        
        for evt in events:
            if isinstance(evt, dict):
                result.append(EventData(
                    no=evt.get('no', 0),
                    time=evt.get('time', 0.0),
                    duration=evt.get('duration', 0.0),
                    cmd=evt.get('cmd', ''),
                    name=evt.get('name', ''),
                    cue=evt.get('cue'),
                    subtrack_type=evt.get('subtrack_type', '')
                ))
        
        return result
    
    def _parse_value(self, value: str) -> Any:
        """Parse a string value into appropriate Python type."""
        # Try JSON (arrays, objects)
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        
        # Try integer
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        
        # Try float
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        
        # Try boolean
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        
        # Default to string
        return value
    
    def _smart_split_pipe(self, message_str: str) -> List[str]:
        """
        Split message string on pipe characters, but only when outside JSON structures.
        
        This handles cases where JSON values contain pipe characters (e.g., fingerprints
        like "0.409233||Go+" which have || inside the JSON string).
        
        Args:
            message_str: The pipe-delimited message string
            
        Returns:
            List of key=value parts, properly split
        """
        parts = []
        current_part = []
        bracket_depth = 0  # Track [...] and {...} nesting
        in_string = False  # Track if we're inside a JSON string
        escape_next = False  # Track escape characters
        
        for char in message_str:
            if escape_next:
                # Previous char was backslash, include this char literally
                current_part.append(char)
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                # Escape character inside string
                current_part.append(char)
                escape_next = True
                continue
            
            if char == '"' and bracket_depth > 0:
                # Toggle string state when inside JSON structure
                in_string = not in_string
                current_part.append(char)
                continue
            
            if not in_string:
                if char in '[{':
                    bracket_depth += 1
                    current_part.append(char)
                    continue
                elif char in ']}':
                    bracket_depth -= 1
                    current_part.append(char)
                    continue
                elif char == '|' and bracket_depth == 0:
                    # Split point - pipe outside JSON
                    parts.append(''.join(current_part))
                    current_part = []
                    continue
            
            current_part.append(char)
        
        # Don't forget the last part
        if current_part:
            parts.append(''.join(current_part))
        
        return parts
    
    def _align_to_4(self, pos: int) -> int:
        """Align position to 4-byte boundary."""
        return (pos + 3) & ~3


# Singleton instance for convenience
_parser_instance: Optional[OSCParser] = None


def get_osc_parser() -> OSCParser:
    """Get the singleton OSC parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = OSCParser()
    return _parser_instance
