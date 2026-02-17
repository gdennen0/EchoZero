"""
MA3 Command Registry

Central registry for all MA3 commands.
Provides lookup, validation, and command creation utilities.
"""

from typing import Any, Dict, List, Optional, Type

from .base import BaseMA3Command, CommandContext
from .event_commands import (
    AddEventCommand,
    MoveEventCommand,
    DeleteEventCommand,
    UpdateEventCommand,
)
from .layer_commands import (
    AddLayerCommand,
    DeleteLayerCommand,
    RenameLayerCommand,
)
from .track_commands import (
    AddTrackCommand,
    DeleteTrackCommand,
    AddTrackGroupCommand,
)


class MA3CommandRegistry:
    """
    Registry of all MA3 commands.
    
    Provides:
    - Command lookup by type
    - Command creation from OSC messages
    - Command validation
    - Command documentation
    """
    
    # Map of command type to command class
    _commands: Dict[str, Type[BaseMA3Command]] = {
        # Event commands
        "add_event": AddEventCommand,
        "move_event": MoveEventCommand,
        "delete_event": DeleteEventCommand,
        "update_event": UpdateEventCommand,
        # Layer commands
        "add_layer": AddLayerCommand,
        "delete_layer": DeleteLayerCommand,
        "rename_layer": RenameLayerCommand,
        # Track commands
        "add_track": AddTrackCommand,
        "delete_track": DeleteTrackCommand,
        "add_track_group": AddTrackGroupCommand,
    }
    
    # Map of OSC address to command type
    _osc_addresses: Dict[str, str] = {
        "/echozero/timecode/add_event": "add_event",
        "/echozero/timecode/move_event": "move_event",
        "/echozero/timecode/delete_event": "delete_event",
        "/echozero/timecode/update_event": "update_event",
        "/echozero/timecode/add_track": "add_track",
        "/echozero/timecode/delete_track": "delete_track",
        "/echozero/timecode/rename_track": "rename_layer",
        "/echozero/timecode/add_trackgroup": "add_track_group",
    }
    
    @classmethod
    def get_command_class(cls, command_type: str) -> Optional[Type[BaseMA3Command]]:
        """
        Get command class by type.
        
        Args:
            command_type: The command type identifier (e.g., "add_event")
            
        Returns:
            The command class, or None if not found
        """
        return cls._commands.get(command_type)
    
    @classmethod
    def get_command_type_from_osc(cls, address: str) -> Optional[str]:
        """
        Get command type from OSC address.
        
        Args:
            address: The OSC address (e.g., "/echozero/timecode/add_event")
            
        Returns:
            The command type, or None if not found
        """
        return cls._osc_addresses.get(address)
    
    @classmethod
    def list_commands(cls) -> Dict[str, Type[BaseMA3Command]]:
        """List all registered command types with their classes."""
        return cls._commands.copy()
    
    @classmethod
    def list_command_types(cls) -> List[str]:
        """List all registered command type names."""
        return list(cls._commands.keys())
    
    @classmethod
    def list_osc_addresses(cls) -> List[str]:
        """List all registered OSC addresses."""
        return list(cls._osc_addresses.keys())
    
    @classmethod
    def create_command_from_osc(
        cls,
        context: CommandContext,
        address: str,
        args: List[Any],
    ) -> Optional[BaseMA3Command]:
        """
        Create a command from an OSC message.
        
        Args:
            context: Command context
            address: OSC address
            args: OSC arguments
            
        Returns:
            The created command, or None if address not recognized
        """
        import json
        
        command_type = cls.get_command_type_from_osc(address)
        if not command_type:
            return None
        
        command_class = cls.get_command_class(command_type)
        if not command_class:
            return None
        
        # Parse arguments based on command type
        try:
            if command_type == "add_event":
                # Args: [timecode_no, track_group_idx, track_idx, time, event_type, props_json]
                if len(args) < 6:
                    return None
                context.timecode_no = int(args[0])
                context.track_group_idx = int(args[1])
                props = json.loads(args[5]) if args[5] else {}
                return AddEventCommand(
                    context=context,
                    time=float(args[3]),
                    event_type=str(args[4]),
                    track_idx=int(args[2]),
                    properties=props,
                )
            
            elif command_type == "move_event":
                # Args: [timecode_no, track_group_idx, track_idx, event_idx, new_time]
                if len(args) < 5:
                    return None
                context.timecode_no = int(args[0])
                context.track_group_idx = int(args[1])
                return MoveEventCommand(
                    context=context,
                    event_idx=int(args[3]),
                    new_time=float(args[4]),
                    track_idx=int(args[2]),
                )
            
            elif command_type == "delete_event":
                # Args: [timecode_no, track_group_idx, track_idx, event_idx]
                if len(args) < 4:
                    return None
                context.timecode_no = int(args[0])
                context.track_group_idx = int(args[1])
                return DeleteEventCommand(
                    context=context,
                    event_idx=int(args[3]),
                    track_idx=int(args[2]),
                )
            
            elif command_type == "update_event":
                # Args: [timecode_no, track_group_idx, track_idx, event_idx, updates_json]
                if len(args) < 5:
                    return None
                context.timecode_no = int(args[0])
                context.track_group_idx = int(args[1])
                updates = json.loads(args[4]) if args[4] else {}
                return UpdateEventCommand(
                    context=context,
                    event_idx=int(args[3]),
                    track_idx=int(args[2]),
                    classification=updates.get("classification"),
                    duration=updates.get("duration"),
                    properties=updates.get("properties"),
                )
            
            elif command_type == "add_track":
                # Args: [timecode_no, track_group_idx, name]
                if len(args) < 3:
                    return None
                context.timecode_no = int(args[0])
                return AddTrackCommand(
                    context=context,
                    name=str(args[2]),
                    track_group_idx=int(args[1]),
                )
            
            elif command_type == "delete_track":
                # Args: [timecode_no, track_group_idx, track_idx]
                if len(args) < 3:
                    return None
                context.timecode_no = int(args[0])
                return DeleteTrackCommand(
                    context=context,
                    track_idx=int(args[2]),
                    track_group_idx=int(args[1]),
                )
            
            elif command_type == "rename_layer":
                # Args: [timecode_no, track_group_idx, track_idx, new_name]
                if len(args) < 4:
                    return None
                context.timecode_no = int(args[0])
                context.track_group_idx = int(args[1])
                return RenameLayerCommand(
                    context=context,
                    layer_idx=int(args[2]),
                    new_name=str(args[3]),
                )
            
            elif command_type == "add_track_group":
                # Args: [timecode_no, name]
                if len(args) < 2:
                    return None
                context.timecode_no = int(args[0])
                return AddTrackGroupCommand(
                    context=context,
                    name=str(args[1]),
                )
            
        except (ValueError, KeyError, IndexError) as e:
            from src.utils.message import Log
            Log.warning(f"Failed to parse OSC command {address}: {e}")
            return None
        
        return None
    
    @classmethod
    def get_command_documentation(cls, command_type: str) -> Optional[str]:
        """Get documentation for a command type."""
        command_class = cls.get_command_class(command_type)
        if command_class:
            return command_class.__doc__
        return None
