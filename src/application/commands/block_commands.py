"""
Block Commands - Standardized Undoable Block Operations

All block-related commands that flow through facade.command_bus.

STANDARD BLOCK COMMANDS
=======================

| Command                    | Redo Action           | Undo Action              |
|----------------------------|----------------------|--------------------------|
| AddBlockCommand            | Creates block        | Deletes block            |
| DeleteBlockCommand         | Deletes block        | Recreates block + conns  |
| RenameBlockCommand         | Sets new name        | Restores old name        |
| MoveBlockCommand           | Sets new position    | Restores old position    |
| UpdateBlockMetadataCommand | Sets metadata value  | Restores old value       |

USAGE
=====
```python
from src.application.commands import AddBlockCommand

cmd = AddBlockCommand(facade, "LoadAudio", "MyBlock")
facade.command_bus.execute(cmd)
# Commands should be executed via facade.command_bus.execute(cmd)
```
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
from copy import deepcopy
from .base_command import EchoZeroCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade

class AddBlockCommand(EchoZeroCommand):
    """
    Create a new block.
    
    Redo: Creates a new block with the specified type and name
    Undo: Deletes the created block
    
    Args:
        facade: ApplicationFacade instance
        block_type: Block type ID (e.g., "LoadAudio", "Separator")
        name: Optional block name (auto-generated if None)
    """
    
    COMMAND_TYPE = "block.add"
    
    def __init__(
        self, 
        facade: "ApplicationFacade", 
        block_type: str, 
        name: Optional[str] = None
    ):
        description = f"Add {name or block_type}"
        super().__init__(facade, description)
        
        self._block_type = block_type
        self._name = name
        self._created_block_id: Optional[str] = None
    
    def redo(self):
        """Create the block."""
        
        result = self._facade.add_block(self._block_type, self._name)
        
        
        if result.success and result.data:
            self._created_block_id = result.data.id
            # Update description with actual name
            self.setText(f"Add {result.data.name}")
            
        else:
            self._log_error(f"Failed to add block: {result.message if hasattr(result, 'message') else 'Unknown error'}")
            
    
    def undo(self):
        """Delete the created block."""
        if self._created_block_id:
            self._facade.delete_block(self._created_block_id)

class DeleteBlockCommand(EchoZeroCommand):
    """
    Delete a block and all its connections.
    
    Redo: Deletes the block (connections are automatically removed)
    Undo: Recreates the block with original properties and restores connections
    
    State Preserved:
        - Block type and name
        - Block metadata (settings)
        - All connections (incoming and outgoing)
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block to delete
    """
    
    COMMAND_TYPE = "block.delete"
    
    def __init__(self, facade: "ApplicationFacade", block_id: str):
        # Get block name for description
        result = facade.describe_block(block_id)
        name = result.data.name if result.success and result.data else block_id
        
        super().__init__(facade, f"Delete {name}")
        
        self._block_id = block_id
        self._deleted_block_data: Optional[Dict[str, Any]] = None
        self._deleted_connections: List[Dict[str, Any]] = []
        self._deleted_position: Optional[Dict[str, float]] = None
    
    def redo(self):
        """Delete the block, storing state for undo."""
        # Store block data before deletion
        result = self._facade.describe_block(self._block_id)
        if result.success and result.data:
            block = result.data
            # Use block.to_dict() to properly serialize PortType objects
            block_dict = block.to_dict()
            self._deleted_block_data = {
                "type": block_dict["type"],
                "name": block_dict["name"],
                "metadata": block_dict.get("metadata", {}).copy() if block_dict.get("metadata") else {},
                "inputs": block_dict.get("inputs", {}),
                "outputs": block_dict.get("outputs", {}),
                "bidirectional": block_dict.get("bidirectional", {}),
            }
        
        # Store position if available
        pos_result = self._facade.get_ui_state("block_position", self._block_id)
        if pos_result.success and pos_result.data:
            self._deleted_position = pos_result.data
        
        # Store connections before deletion
        conn_result = self._facade.list_connections()
        if conn_result.success and conn_result.data:
            self._deleted_connections = [
                {
                    "source_block_id": c.source_block_id,
                    "source_output_name": c.source_output_name,
                    "target_block_id": c.target_block_id,
                    "target_input_name": c.target_input_name,
                }
                for c in conn_result.data
                if c.source_block_id == self._block_id 
                or c.target_block_id == self._block_id
            ]
        
        # Delete the block
        self._facade.delete_block(self._block_id)
    
    def undo(self):
        """Recreate the deleted block and connections."""
        if not self._deleted_block_data:
            self._log_warning("No block data stored, cannot undo")
            return
        
        # Recreate block
        result = self._facade.add_block(
            self._deleted_block_data["type"],
            self._deleted_block_data["name"]
        )
        
        if not result.success or not result.data:
            self._log_error("Failed to recreate block")
            return
        
        new_block = result.data
        new_block_id = new_block.id
        
        # Restore metadata
        if self._deleted_block_data.get("metadata"):
            block_result = self._facade.describe_block(new_block_id)
            if block_result.success and block_result.data:
                block_result.data.metadata = self._deleted_block_data["metadata"]
                self._facade.block_service.update_block(
                    self._facade.current_project_id,
                    new_block_id,
                    block_result.data
                )
        
        # Restore position
        if self._deleted_position:
            self._facade.set_ui_state("block_position", new_block_id, self._deleted_position)
        
        # Restore connections
        for conn in self._deleted_connections:
            src_id = new_block_id if conn["source_block_id"] == self._block_id else conn["source_block_id"]
            tgt_id = new_block_id if conn["target_block_id"] == self._block_id else conn["target_block_id"]
            
            self._facade.connect_blocks(
                src_id, conn["source_output_name"],
                tgt_id, conn["target_input_name"]
            )
        
        # Update reference for subsequent redo
        self._block_id = new_block_id

class RenameBlockCommand(EchoZeroCommand):
    """
    Rename a block.
    
    Redo: Sets the block's name to the new value
    Undo: Restores the original name
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block to rename
        new_name: New name for the block
    """
    
    COMMAND_TYPE = "block.rename"
    
    def __init__(self, facade: "ApplicationFacade", block_id: str, new_name: str):
        super().__init__(facade, f"Rename to {new_name}")
        
        self._block_id = block_id
        self._new_name = new_name
        self._old_name: Optional[str] = None
    
    def redo(self):
        """Rename to new name."""
        # Store old name (first time only)
        if self._old_name is None:
            result = self._facade.describe_block(self._block_id)
            if result.success and result.data:
                self._old_name = result.data.name
        
        self._facade.rename_block(self._block_id, self._new_name)
    
    def undo(self):
        """Rename back to old name."""
        if self._old_name:
            self._facade.rename_block(self._block_id, self._old_name)

class DuplicateBlockCommand(EchoZeroCommand):
    """
    Duplicate a block with its settings and filters (but not connections).
    
    Redo: Creates a new block with copied settings and filters
    Undo: Deletes the duplicated block
    
    State Preserved:
        - Block type
        - Block metadata (settings)
        - Filter selections (from block.metadata["filter_selections"])
        - Block position (offset to avoid overlap)
    
    State NOT Preserved:
        - Connections (duplicated block has no connections)
        - Data items (duplicated block starts fresh)
        - Block local state (duplicated block starts fresh)
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block to duplicate
    """
    
    COMMAND_TYPE = "block.duplicate"
    
    def __init__(self, facade: "ApplicationFacade", block_id: str):
        # Get block name for description
        result = facade.describe_block(block_id)
        name = result.data.name if result.success and result.data else block_id
        
        super().__init__(facade, f"Duplicate {name}")
        
        self._source_block_id = block_id
        self._duplicated_block_id: Optional[str] = None
        self._source_block_data: Optional[Dict[str, Any]] = None
        self._source_position: Optional[Dict[str, float]] = None
    
    def redo(self):
        """Create duplicate block with copied settings and filters."""
        # Get source block data (first time only)
        if self._source_block_data is None:
            result = self._facade.describe_block(self._source_block_id)
            if not result.success or not result.data:
                self._log_error("Failed to get source block data")
                return
            
            block = result.data
            self._source_block_data = {
                "type": block.type,
                "name": block.name,
                "metadata": deepcopy(block.metadata) if block.metadata else {},
            }
            
            # Get source position
            pos_result = self._facade.get_ui_state("block_position", self._source_block_id)
            if pos_result.success and pos_result.data:
                self._source_position = pos_result.data.copy()
        
        # Create new block with same type and name + " Copy"
        new_name = f"{self._source_block_data['name']} Copy"
        result = self._facade.add_block(
            self._source_block_data["type"],
            new_name
        )
        
        if not result.success or not result.data:
            self._log_error("Failed to create duplicated block")
            return
        
        new_block = result.data
        self._duplicated_block_id = new_block.id
        
        # Copy metadata (includes settings and filter_selections)
        # Use block_service.update_block to ensure metadata is properly saved
        if self._source_block_data.get("metadata"):
            result = self._facade.describe_block(self._duplicated_block_id)
            if result.success and result.data:
                block = result.data
                # Deep copy metadata to ensure nested structures are properly copied
                block.metadata = deepcopy(self._source_block_data["metadata"])
                self._facade.block_service.update_block(
                    self._facade.current_project_id,
                    self._duplicated_block_id,
                    block
                )
        
        # Set position offset from source (to avoid overlap)
        if self._source_position:
            offset_x = 250  # Offset to the right
            offset_y = 0
            new_x = self._source_position.get("x", 0) + offset_x
            new_y = self._source_position.get("y", 0) + offset_y
            
            self._facade.set_ui_state("block_position", self._duplicated_block_id, {
                "x": new_x,
                "y": new_y,
                "block_name": new_name
            })
        
        # Update description with actual name
        self.setText(f"Duplicate {new_name}")
    
    def undo(self):
        """Delete the duplicated block."""
        if self._duplicated_block_id:
            self._facade.delete_block(self._duplicated_block_id)

class MoveBlockCommand(EchoZeroCommand):
    """
    Move a block to a new position.
    
    Redo: Sets the block's position to the new coordinates
    Undo: Restores the original position
    
    Supports command merging for drag operations (multiple small moves
    become one undo step).
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block to move
        new_x: New X coordinate
        new_y: New Y coordinate
        old_x: Optional original X (if known, avoids DB lookup)
        old_y: Optional original Y (if known, avoids DB lookup)
    """
    
    COMMAND_TYPE = "block.move"
    MERGE_ID = 1000  # All move commands can merge
    
    def __init__(
        self, 
        facade: "ApplicationFacade", 
        block_id: str, 
        new_x: float, 
        new_y: float,
        old_x: Optional[float] = None,
        old_y: Optional[float] = None
    ):
        super().__init__(facade, "Move Block")
        
        self._block_id = block_id
        self._new_x = new_x
        self._new_y = new_y
        self._old_x = old_x
        self._old_y = old_y
    
    def redo(self):
        """Move to new position."""
        # Store old position (first time only if not provided)
        is_first_run = self._old_x is None
        
        if is_first_run:
            result = self._facade.get_ui_state("block_position", self._block_id)
            if result.success and result.data:
                self._old_x = result.data.get("x", 0)
                self._old_y = result.data.get("y", 0)
            else:
                self._old_x = 0
                self._old_y = 0
        
        self._facade.set_ui_state("block_position", self._block_id, {
            "x": self._new_x,
            "y": self._new_y
        })
    
    def undo(self):
        """Move back to old position."""
        if self._old_x is not None:
            self._facade.set_ui_state("block_position", self._block_id, {
                "x": self._old_x,
                "y": self._old_y
            })
    
    def id(self) -> int:
        """Enable merging of move commands for same block."""
        return self.MERGE_ID
    
    def mergeWith(self, other) -> bool:
        """
        Merge consecutive moves of the same block.
        
        This combines multiple small moves (e.g., during dragging) into
        one undo step that returns to the original position.
        """
        if not isinstance(other, MoveBlockCommand):
            return False
        if other._block_id != self._block_id:
            return False
        
        # Take the new command's position, keep our old position
        self._new_x = other._new_x
        self._new_y = other._new_y
        self.setText(f"Move Block to ({self._new_x:.0f}, {self._new_y:.0f})")
        return True

class UpdateBlockMetadataCommand(EchoZeroCommand):
    """
    Update a metadata value on a block.
    
    Redo: Sets the metadata key to the new value
    Undo: Restores the previous value (or removes key if it didn't exist)
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block to update
        key: Metadata key to update
        new_value: New value for the key
        description: Optional custom description for the command
    """
    
    COMMAND_TYPE = "block.metadata"
    
    def __init__(
        self, 
        facade: "ApplicationFacade", 
        block_id: str, 
        key: str,
        new_value: Any,
        description: Optional[str] = None
    ):
        desc = description or f"Update {key}"
        super().__init__(facade, desc)
        
        self._block_id = block_id
        self._key = key
        self._new_value = new_value
        self._old_value: Any = None
        self._key_existed: bool = True
    
    def redo(self):
        """Apply new value."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            self._log_error(f"Block not found: {self._block_id}")
            return
        
        block = result.data
        
        # Store old value (first time only)
        if not self._executed:
            self._key_existed = self._key in block.metadata
            self._old_value = block.metadata.get(self._key)
            self._executed = True
        
        # Set new value
        block.metadata[self._key] = self._new_value
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block,
            changed_keys=[self._key],
            update_source="settings",
        )
    
    def undo(self):
        """Restore old value."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            self._log_error(f"Block not found: {self._block_id}")
            return
        
        block = result.data
        
        if not self._key_existed:
            # Key didn't exist before, remove it
            block.metadata.pop(self._key, None)
        else:
            block.metadata[self._key] = self._old_value
        
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block,
            changed_keys=[self._key],
            update_source="settings",
        )

class SetBlockInputCommand(EchoZeroCommand):
    """
    Set the value of a block's input port.
    
    Redo: Sets the input to the new value
    Undo: Restores the previous value
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block
        input_name: Name of the input port
        new_value: New value for the input
    """
    
    COMMAND_TYPE = "block.input"
    
    def __init__(
        self, 
        facade: "ApplicationFacade", 
        block_id: str, 
        input_name: str,
        new_value: Any
    ):
        super().__init__(facade, f"Set {input_name}")
        
        self._block_id = block_id
        self._input_name = input_name
        self._new_value = new_value
        self._old_value: Any = None
    
    def redo(self):
        """Set new input value."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        
        # Store old value (first time only)
        from src.features.blocks.domain import PortDirection
        from src.shared.domain.value_objects.port_type import PortType
        from src.features.blocks.domain import Port
        
        if not self._executed:
            input_port = block.get_port(self._input_name, PortDirection.INPUT)
            self._old_value = input_port.port_type if input_port else None
            self._executed = True
        
        # Set new value - update port type
        if isinstance(self._new_value, PortType):
            port = block.get_port(self._input_name, PortDirection.INPUT)
            if port:
                # Update existing port
                block.ports[block._make_port_key(self._input_name, PortDirection.INPUT)] = Port(
                    name=self._input_name,
                    port_type=self._new_value,
                    direction=PortDirection.INPUT,
                    metadata=port.metadata
                )
            else:
                # Create new port
                block.add_port(self._input_name, self._new_value, PortDirection.INPUT)
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )
    
    def undo(self):
        """Restore old input value."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        from src.features.blocks.domain import PortDirection
        from src.shared.domain.value_objects.port_type import PortType
        from src.features.blocks.domain import Port
        
        if isinstance(self._old_value, PortType):
            port = block.get_port(self._input_name, PortDirection.INPUT)
            if port:
                # Restore old port type
                block.ports[block._make_port_key(self._input_name, PortDirection.INPUT)] = Port(
                    name=self._input_name,
                    port_type=self._old_value,
                    direction=PortDirection.INPUT,
                    metadata=port.metadata
                )
            elif self._old_value:
                # Restore port if it existed
                block.add_port(self._input_name, self._old_value, PortDirection.INPUT)
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block
        )

class ConfigureBlockCommand(EchoZeroCommand):
    """
    Configure a block by executing a block command (e.g., set_model, set_device).
    
    This wraps facade.execute_block_command for undo support.
    
    Redo: Executes the command with new value
    Undo: Executes the command with old value
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block to configure
        command_name: Command to execute (e.g., "set_model")
        new_value: New value to set
        old_value: Optional previous value (for undo)
        description: Optional custom description
    """
    
    COMMAND_TYPE = "block.configure"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        command_name: str,
        new_value: Any,
        old_value: Optional[Any] = None,
        description: Optional[str] = None
    ):
        # Create user-friendly description
        desc = description or f"Set {command_name.replace('set_', '').replace('_', ' ').title()}"
        super().__init__(facade, desc)
        
        self._block_id = block_id
        self._command_name = command_name
        self._new_value = new_value
        self._old_value = old_value
        self._metadata_key: Optional[str] = None
    
    def _get_metadata_key_for_command(self, command_name: str) -> str:
        """Map command name to metadata key."""
        # Common mappings
        mappings = {
            "set_model": "model",
            "set_device": "device",
            "set_audio_format": "audio_format",
            "set_onset_threshold": "onset_threshold",
            "set_min_duration": "min_duration",
            "set_export_dir": "export_dir",
            "set_audio_path": "file_path",
            "set_path": "file_path",
            "set_plot_style": "plot_style",
            "set_figure_size": "figure_size",
            "set_dpi": "dpi",
            "set_show_labels": "show_labels",
            "set_show_grid": "show_grid",
            "set_two_stems": "two_stems",
        }
        return mappings.get(command_name, command_name.replace("set_", ""))
    
    def redo(self):
        """Execute the configuration command."""
        # Get old value from metadata if not provided
        if self._old_value is None and not self._executed:
            result = self._facade.describe_block(self._block_id)
            if result.success and result.data:
                self._metadata_key = self._get_metadata_key_for_command(self._command_name)
                self._old_value = result.data.metadata.get(self._metadata_key)
            self._executed = True
        
        # Execute the command
        args = [self._new_value] if self._new_value is not None else []
        self._facade.execute_block_command(
            identifier=self._block_id,
            command_name=self._command_name,
            args=[str(a) for a in args],
            kwargs={}
        )
    
    def undo(self):
        """Restore previous configuration."""
        if self._old_value is not None:
            args = [self._old_value]
            self._facade.execute_block_command(
                identifier=self._block_id,
                command_name=self._command_name,
                args=[str(a) for a in args],
                kwargs={}
            )
        elif self._metadata_key:
            # If old value was None, remove the key
            result = self._facade.describe_block(self._block_id)
            if result.success and result.data:
                block = result.data
                block.metadata.pop(self._metadata_key, None)
                self._facade.block_service.update_block(
                    self._facade.current_project_id,
                    self._block_id,
                    block
                )

class BatchUpdateMetadataCommand(EchoZeroCommand):
    """
    Update multiple metadata keys at once.
    
    Redo: Sets all new values
    Undo: Restores all old values
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block
        updates: Dict of key -> new_value
        description: Optional custom description
    """
    
    COMMAND_TYPE = "block.batch_metadata"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        updates: Dict[str, Any],
        description: Optional[str] = None
    ):
        desc = description or f"Update {len(updates)} settings"
        super().__init__(facade, desc)
        
        self._block_id = block_id
        self._updates = updates.copy()
        self._old_values: Dict[str, Any] = {}
    
    def redo(self):
        """Apply all updates."""
        from src.utils.message import Log
        
        
        
        result = self._facade.describe_block(self._block_id)
        if not result.success:
            
            Log.error(
                f"BatchUpdateMetadataCommand: Failed to load block {self._block_id}: "
                f"{result.message if hasattr(result, 'message') else 'Unknown error'}"
            )
            return
        
        if not result.data:
            
            Log.error(f"BatchUpdateMetadataCommand: Block {self._block_id} not found")
            return
        
        block = result.data
        
        
        
        # Store old values (first time only)
        if not self._executed:
            for key in self._updates:
                self._old_values[key] = block.metadata.get(key)
            self._executed = True
        
        # Apply updates
        Log.debug(
            f"BatchUpdateMetadataCommand: Updating {len(self._updates)} metadata keys for block '{block.name}': "
            f"{', '.join(self._updates.keys())}"
        )
        for key, value in self._updates.items():
            old_value = block.metadata.get(key)
            block.metadata[key] = value
            
            Log.debug(
                f"BatchUpdateMetadataCommand: Updated '{key}': {old_value} -> {value}"
            )
        
        try:
            
            self._facade.block_service.update_block(
                self._facade.current_project_id,
                self._block_id,
                block,
                changed_keys=list(self._updates.keys()),
                update_source="settings",
            )
            
            Log.debug(
                f"BatchUpdateMetadataCommand: Successfully updated block '{block.name}' metadata"
            )
        except Exception as e:
            
            Log.error(
                f"BatchUpdateMetadataCommand: Failed to update block '{block.name}': {e}"
            )
            raise
    
    def undo(self):
        """Restore all old values."""
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        
        for key, old_value in self._old_values.items():
            if old_value is None:
                block.metadata.pop(key, None)
            else:
                block.metadata[key] = old_value
        
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            self._block_id,
            block,
            changed_keys=list(self._old_values.keys()),
            update_source="settings",
        )


class ResetBlockStateCommand(EchoZeroCommand):
    """
    Reset a block's state to its initial state.
    
    Clears:
    - All metadata (configuration and settings)
    - All owned data items (EventDataItems, AudioDataItems, etc.)
    - All local state (input/output references)
    
    Redo: Clears all state (resets to initial state)
    Undo: Restores the original metadata (data items and local state are not restored)
    
    Note: Data items and local state are permanently deleted and cannot be restored via undo.
    Only metadata is restored on undo. This is intentional as reset is a destructive operation.
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of block to reset
    """
    
    COMMAND_TYPE = "block.reset_state"
    
    def __init__(self, facade: "ApplicationFacade", block_id: str):
        # Get block name for description
        result = facade.describe_block(block_id)
        name = result.data.name if result.success and result.data else block_id
        
        super().__init__(facade, f"Reset {name} state")
        
        self._block_id = block_id
        self._original_metadata: Optional[Dict[str, Any]] = None
    
    def redo(self):
        """Reset block state (metadata, data items, local state)."""
        from src.utils.message import Log
        
        # Store original metadata (first time only) for undo
        if self._original_metadata is None:
            result = self._facade.describe_block(self._block_id)
            if result.success and result.data:
                # Deep copy metadata to preserve it
                import copy
                self._original_metadata = copy.deepcopy(result.data.metadata or {})
            else:
                self._original_metadata = {}
        
        # Reset block state (clears metadata, data items, local state)
        # IMPORTANT: Call block_service directly to avoid infinite recursion
        # (facade.reset_block_state creates another command, causing recursion)
        try:
            self._facade.block_service.reset_block_state(
                self._facade.current_project_id,
                self._block_id,
                data_item_repo=getattr(self._facade, 'data_item_repo', None),
                block_local_state_repo=getattr(self._facade, 'block_local_state_repo', None)
            )
        except Exception as e:
            self._log_error(f"Failed to reset block state: {e}")
    
    def undo(self):
        """Restore original metadata (data items and local state are not restored)."""
        if self._original_metadata is not None:
            # Restore all original metadata
            # Note: Data items and local state are not restored - reset is destructive
            self._facade.update_block_metadata(
                self._block_id,
                self._original_metadata
            )
