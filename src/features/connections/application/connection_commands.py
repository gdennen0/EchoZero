"""
Connection Commands - Standardized Undoable Connection Operations

All connection-related commands that flow through facade.command_bus.

STANDARD CONNECTION COMMANDS
============================

| Command                   | Redo Action           | Undo Action              |
|---------------------------|----------------------|--------------------------|
| CreateConnectionCommand   | Creates connection   | Deletes connection       |
| DeleteConnectionCommand   | Deletes connection   | Recreates connection     |

USAGE
=====
    from src.application.commands import CreateConnectionCommand

    cmd = CreateConnectionCommand(
        facade,
        source_block_id="abc123",
        source_output_name="audio_out",
        target_block_id="def456",
        target_input_name="audio_in"
    )
    facade.command_bus.execute(cmd)
"""

from typing import TYPE_CHECKING, Optional
from src.application.commands.base_command import EchoZeroCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class CreateConnectionCommand(EchoZeroCommand):
    """
    Create a connection between two blocks.
    
    Redo: Creates a connection from source output to target input
    Undo: Deletes the created connection
    
    Args:
        facade: ApplicationFacade instance
        source_block_id: ID of the source block
        source_output_name: Name of the output port on source block
        target_block_id: ID of the target block
        target_input_name: Name of the input port on target block
    """
    
    COMMAND_TYPE = "connection.create"
    
    def __init__(
        self, 
        facade: "ApplicationFacade",
        source_block_id: str,
        source_output_name: str,
        target_block_id: str,
        target_input_name: str
    ):
        description = f"Connect {source_output_name} -> {target_input_name}"
        super().__init__(facade, description)
        
        self._source_block_id = source_block_id
        self._source_output_name = source_output_name
        self._target_block_id = target_block_id
        self._target_input_name = target_input_name
        self._created_connection_id: Optional[str] = None
    
    def redo(self):
        """Create the connection."""
        result = self._facade.connect_blocks(
            self._source_block_id,
            self._source_output_name,
            self._target_block_id,
            self._target_input_name
        )
        if result.success and result.data:
            self._created_connection_id = result.data.id
        else:
            self._log_error(f"Failed to create connection: {getattr(result, 'message', 'Unknown error')}")
    
    def undo(self):
        """Delete the created connection."""
        if self._created_connection_id:
            self._facade.disconnect_blocks(self._created_connection_id)


class DeleteConnectionCommand(EchoZeroCommand):
    """
    Delete an existing connection.
    
    Redo: Deletes the connection
    Undo: Recreates the connection with original endpoints
    
    State Preserved:
        - Source block and port
        - Target block and port
    
    Args:
        facade: ApplicationFacade instance
        connection_id: ID of connection to delete
    """
    
    COMMAND_TYPE = "connection.delete"
    
    def __init__(self, facade: "ApplicationFacade", connection_id: str):
        super().__init__(facade, "Delete Connection")
        
        self._connection_id = connection_id
        self._source_block_id: Optional[str] = None
        self._source_output_name: Optional[str] = None
        self._target_block_id: Optional[str] = None
        self._target_input_name: Optional[str] = None
    
    def _find_connection(self, connection_id: str):
        """Find connection data from list_connections."""
        result = self._facade.list_connections()
        if result.success and result.data:
            for conn in result.data:
                if conn.id == connection_id:
                    return conn
        return None
    
    def redo(self):
        """Delete the connection, storing state for undo."""
        # Store connection data before deletion (first time only)
        if not self._executed:
            conn = self._find_connection(self._connection_id)
            if conn:
                self._source_block_id = conn.source_block_id
                self._source_output_name = conn.source_output_name
                self._target_block_id = conn.target_block_id
                self._target_input_name = conn.target_input_name
                
                # Update description to be more informative
                self.setText(f"Delete {self._source_output_name} -> {self._target_input_name}")
            self._executed = True
        
        # Delete the connection
        self._facade.disconnect_blocks(self._connection_id)
    
    def undo(self):
        """Recreate the deleted connection."""
        if not all([
            self._source_block_id,
            self._source_output_name,
            self._target_block_id,
            self._target_input_name
        ]):
            self._log_warning("No connection data stored, cannot undo")
            return
        
        result = self._facade.connect_blocks(
            self._source_block_id,
            self._source_output_name,
            self._target_block_id,
            self._target_input_name
        )
        
        if result.success and result.data:
            # Update reference for subsequent redo
            self._connection_id = result.data.id
        else:
            self._log_error("Failed to recreate connection")
