"""
EchoZero Command System - Standardized Undoable Operations

Firmware-style architecture for bulletproof undo/redo:
- ALL undoable operations go through facade.command_bus.execute()
- Commands are automatically logged and tracked
- Commands follow a strict standard (see base_command.py)

USAGE
=====
    from src.application.commands import AddBlockCommand

    # Execute a command (this pushes to undo stack AND executes)
    cmd = AddBlockCommand(facade, "LoadAudio", "MyBlock")
    facade.command_bus.execute(cmd)

    # Undo/Redo
    facade.command_bus.undo()
    facade.command_bus.redo()

    # Macro (multiple commands as one undo step)
    facade.command_bus.begin_macro("Delete Selection")
    facade.command_bus.execute(DeleteBlockCommand(facade, block1_id))
    facade.command_bus.execute(DeleteBlockCommand(facade, block2_id))
    facade.command_bus.end_macro()

AVAILABLE COMMANDS
==================
Block Commands:
    - AddBlockCommand(facade, block_type, name)
    - DeleteBlockCommand(facade, block_id)
    - DuplicateBlockCommand(facade, block_id)
    - RenameBlockCommand(facade, block_id, new_name)
    - MoveBlockCommand(facade, block_id, new_x, new_y)
    - UpdateBlockMetadataCommand(facade, block_id, key, new_value)
    - SetBlockInputCommand(facade, block_id, input_name, new_value)

Connection Commands:
    - CreateConnectionCommand(facade, src_block, src_port, tgt_block, tgt_port)
    - DeleteConnectionCommand(facade, connection_id)

CREATING NEW COMMANDS
=====================
1. Inherit from EchoZeroCommand
2. Implement redo() - execute the operation
3. Implement undo() - reverse the operation
4. Store state needed for undo in first redo() call

See base_command.py for the full standard and template.
"""

from .base_command import EchoZeroCommand
from .block_commands import (
    AddBlockCommand,
    DeleteBlockCommand,
    DuplicateBlockCommand,
    RenameBlockCommand,
    MoveBlockCommand,
    UpdateBlockMetadataCommand,
    SetBlockInputCommand,
    ConfigureBlockCommand,
    BatchUpdateMetadataCommand,
    ResetBlockStateCommand,
)
# Connection commands moved to src.features.connections.application
# Import from there: from src.features.connections.application import CreateConnectionCommand
# Re-export for backwards compatibility (lazy import to avoid circular dependency - see __getattr__ at end)
from .timeline_commands import (
    MoveEventCommand,
    ResizeEventCommand,
    CreateEventCommand,
    DeleteEventCommand,
    BatchMoveEventsCommand,
    # Layer commands
    SetLayerVisibilityCommand,
    RenameLayerCommand,
    SetLayerColorCommand,
    SetLayerLockCommand,
    MoveLayerCommand,
    DeleteLayerCommand,
)

from .data_item_commands import (
    CreateEventDataItemCommand,
    UpdateEventDataItemCommand,
    DeleteEventDataItemCommand,
    BatchDeleteEventsFromDataItemCommand,
    AddEventToDataItemCommand,
    UpdateEventInDataItemCommand,
    BatchUpdateEventsCommand as BatchUpdateDataItemEventsCommand,
)
from .action_item_commands import (
    AddActionItemCommand,
    UpdateActionItemCommand,
    DeleteActionItemCommand,
    ReorderActionItemsCommand,
)
from .editor_commands import (
    EditorCreateLayerCommand,
    EditorAddEventsCommand,
    ApplyLayerSnapshotCommand,
    EditorUpdateLayerCommand,
    EditorGetLayersCommand,
    EditorGetEventsCommand,
    EditorDeleteLayerCommand,
)
# Layer sync commands moved to src.features.show_manager.application.commands
# Import from there or use: from src.application.commands.layer_sync import SyncLayerCommand
# Re-exported via lazy import to avoid circular dependency (see __getattr__ below)
from .command_bus import CommandBus

# Lazy imports for layer_sync commands
_LAYER_SYNC_COMMANDS = {
    'SyncLayerCommand',
    'CreateEditorLayerFromMA3Command',
    'CreateMA3TrackFromEditorCommand',
    'CreateEditorLayerCommand',  # Alias
    'CreateMA3TrackCommand',  # Alias
    'MapLayersCommand',
    'UnmapLayersCommand',
    'BatchSyncCommand',
    'UpdateEntitySettingsCommand',
    'AddSyncedEditorLayerCommand',
    'AddSyncedMA3TrackCommand',
    'RemoveSyncedEntityCommand',
    'SetTargetTimecodeCommand',
}

_original_getattr = __getattr__ if '__getattr__' in dir() else None

def __getattr__(name):
    if name in ('CreateConnectionCommand', 'DeleteConnectionCommand'):
        from src.features.connections.application.connection_commands import CreateConnectionCommand, DeleteConnectionCommand
        if name == 'CreateConnectionCommand':
            return CreateConnectionCommand
        return DeleteConnectionCommand
    if name in _LAYER_SYNC_COMMANDS:
        from src.features.show_manager.application import commands as layer_sync_cmds
        return getattr(layer_sync_cmds, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Command Bus (singleton - the ONLY way to execute commands)
    "CommandBus",
    
    # Base class for custom commands
    "EchoZeroCommand",
    
    # Block commands
    "AddBlockCommand",
    "DeleteBlockCommand",
    "DuplicateBlockCommand",
    "RenameBlockCommand",
    "MoveBlockCommand",
    "UpdateBlockMetadataCommand",
    "SetBlockInputCommand",
    "ConfigureBlockCommand",
    "BatchUpdateMetadataCommand",
    "ResetBlockStateCommand",
    
    # Connection commands
    "CreateConnectionCommand",
    "DeleteConnectionCommand",
    
    # Timeline commands
    "MoveEventCommand",
    "ResizeEventCommand",
    "CreateEventCommand",
    "DeleteEventCommand",
    "BatchMoveEventsCommand",
    # Layer commands
    "SetLayerVisibilityCommand",
    "RenameLayerCommand",
    "SetLayerColorCommand",
    "SetLayerLockCommand",
    "MoveLayerCommand",
    "DeleteLayerCommand",
    
    # Data item commands
    "CreateEventDataItemCommand",
    "UpdateEventDataItemCommand",
    "DeleteEventDataItemCommand",
    "BatchDeleteEventsFromDataItemCommand",
    "AddEventToDataItemCommand",
    "UpdateEventInDataItemCommand",
    "BatchUpdateDataItemEventsCommand",
    
    # Action item commands
    "AddActionItemCommand",
    "UpdateActionItemCommand",
    "DeleteActionItemCommand",
    "ReorderActionItemsCommand",
    
    # Editor API commands (for external block manipulation)
    "EditorCreateLayerCommand",
    "EditorAddEventsCommand",
    "ApplyLayerSnapshotCommand",
    "EditorUpdateLayerCommand",
    "EditorGetLayersCommand",
    "EditorGetEventsCommand",
    "EditorDeleteLayerCommand",
    
    # Layer sync commands
    "SyncLayerCommand",
    "CreateEditorLayerFromMA3Command",
    "CreateMA3TrackFromEditorCommand",
    "MapLayersCommand",
    "UnmapLayersCommand",
    "BatchSyncCommand",
    "UpdateEntitySettingsCommand",
    "AddSyncedEditorLayerCommand",
    "AddSyncedMA3TrackCommand",
    "RemoveSyncedEntityCommand",
    "SetTargetTimecodeCommand",
]

