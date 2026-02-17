"""
MA3 Commands Package

Commands for bidirectional communication between EchoZero and grandMA3.
These commands can execute on either the Editor block (local) or MA3 (via OSC).
"""

from .base import BaseMA3Command, CommandContext, CommandResult, ValidationResult
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
from .registry import MA3CommandRegistry

__all__ = [
    # Base
    "BaseMA3Command",
    "CommandContext",
    "CommandResult",
    "ValidationResult",
    # Event commands
    "AddEventCommand",
    "MoveEventCommand",
    "DeleteEventCommand",
    "UpdateEventCommand",
    # Layer commands
    "AddLayerCommand",
    "DeleteLayerCommand",
    "RenameLayerCommand",
    # Track commands
    "AddTrackCommand",
    "DeleteTrackCommand",
    "AddTrackGroupCommand",
    # Registry
    "MA3CommandRegistry",
]
