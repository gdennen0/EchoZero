"""
Connections API - Feature-specific facade for connection operations.

Provides a focused API for managing block connections.
"""
from typing import TYPE_CHECKING

from src.application.api.result_types import CommandResult

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class ConnectionsAPI:
    """
    Connections feature API.
    
    Provides connection management operations:
    - Connect/disconnect blocks
    - List connections
    - Validate connections
    
    Usage:
        connections = ConnectionsAPI(facade)
        result = connections.connect(source_id, "audio_out", target_id, "audio_in")
    """
    
    def __init__(self, facade: "ApplicationFacade"):
        """Initialize with reference to main facade."""
        self._facade = facade
    
    def connect(
        self,
        source_block_id: str,
        source_output_name: str,
        target_block_id: str,
        target_input_name: str
    ) -> CommandResult:
        """Connect two blocks."""
        return self._facade.connect_blocks(
            source_block_id,
            source_output_name,
            target_block_id,
            target_input_name
        )
    
    def disconnect(self, connection_id: str) -> CommandResult:
        """Disconnect blocks by connection ID."""
        return self._facade.disconnect_blocks(connection_id)
    
    def list_connections(self) -> CommandResult:
        """List all connections in current project."""
        return self._facade.list_connections()
    
    def list_connections_for_block(self, block_id: str) -> CommandResult:
        """List connections involving a specific block."""
        return self._facade.list_connections_for_block(block_id)
