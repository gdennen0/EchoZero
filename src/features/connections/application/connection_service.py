"""
Connection Service

Orchestrates connection-related use cases.
Validates port compatibility and handles block connections.
"""
from typing import List, Optional

from src.features.connections.domain.connection import Connection
from src.features.connections.domain.connection_summary import ConnectionSummary
from src.features.connections.domain.connection_repository import ConnectionRepository
from src.features.blocks.domain import Block, BlockRepository, PortDirection
from src.application.events.event_bus import EventBus
from src.application.events import ConnectionCreated, ConnectionRemoved, ConnectionsChanged, BlockChanged
from src.utils.message import Log


class ConnectionService:
    """
    Service for managing connections between blocks.
    
    Orchestrates connection operations:
    - Creating connections with validation
    - Removing connections
    - Validating port compatibility
    - Listing connections
    
    Emits domain events for UI synchronization.
    """
    
    def __init__(
        self,
        connection_repo: ConnectionRepository,
        block_repo: BlockRepository,
        event_bus: EventBus
    ):
        """
        Initialize connection service.
        
        Args:
            connection_repo: Repository for connection persistence
            block_repo: Repository for block validation
            event_bus: Event bus for publishing domain events
        """
        self._connection_repo = connection_repo
        self._block_repo = block_repo
        self._event_bus = event_bus
        Log.info("ConnectionService: Initialized")
    
    def connect_blocks(
        self,
        source_block_id: str,
        source_output_name: str,
        target_block_id: str,
        target_input_name: str
    ) -> Connection:
        """
        Connect two blocks together.
        
        Args:
            source_block_id: Source block identifier
            source_output_name: Output port name on source block
            target_block_id: Target block identifier
            target_input_name: Input port name on target block
            
        Returns:
            Created Connection entity
            
        Raises:
            ValueError: If validation fails (blocks don't exist, ports don't exist,
                       ports incompatible, or target already connected)
        """
        # Get source block
        source_block = self._block_repo.get_by_id(source_block_id)
        if not source_block:
            raise ValueError(f"Source block '{source_block_id}' not found")
        
        # Get target block
        target_block = self._block_repo.get_by_id(target_block_id)
        if not target_block:
            raise ValueError(f"Target block '{target_block_id}' not found")
        
        # Check port directions FIRST (before validating as output/input)
        source_is_bidirectional = source_block.has_port(source_output_name, PortDirection.BIDIRECTIONAL)
        target_is_bidirectional = target_block.has_port(target_input_name, PortDirection.BIDIRECTIONAL)
        
        # Handle bidirectional connections
        if source_is_bidirectional and target_is_bidirectional:
            # Bidirectional to bidirectional connection
            # Validate both ports exist as bidirectional
            source_port_type = source_block.get_port_type(source_output_name, PortDirection.BIDIRECTIONAL)
            target_port_type = target_block.get_port_type(target_input_name, PortDirection.BIDIRECTIONAL)
            
            if not source_port_type:
                raise ValueError(
                    f"Bidirectional port '{source_output_name}' not found on source block '{source_block.name}'"
                )
            if not target_port_type:
                raise ValueError(
                    f"Bidirectional port '{target_input_name}' not found on target block '{target_block.name}'"
                )
        elif source_is_bidirectional or target_is_bidirectional:
            # Mixed connection - not allowed
            raise ValueError(
                "Cannot connect bidirectional ports to regular input/output ports"
            )
        else:
            # Regular input/output connection
            # Validate source output exists
            if not source_block.has_port(source_output_name, PortDirection.OUTPUT):
                raise ValueError(
                    f"Output port '{source_output_name}' not found on source block '{source_block.name}'"
                )
            
            # Validate target input exists
            if not target_block.has_port(target_input_name, PortDirection.INPUT):
                raise ValueError(
                    f"Input port '{target_input_name}' not found on target block '{target_block.name}'"
                )
            
            # Get port types
            source_port_type = source_block.get_port_type(source_output_name, PortDirection.OUTPUT)
            target_port_type = target_block.get_port_type(target_input_name, PortDirection.INPUT)
        
        if not source_port_type.is_compatible_with(target_port_type):
            raise ValueError(
                f"Port types incompatible: '{source_port_type.name}' (output) cannot connect "
                f"to '{target_port_type.name}' (input)"
            )
        
        # Note: Multiple connections to the same input port are now allowed
        # This enables connecting multiple EventDataItems to Event input ports
        
        # Create connection
        connection = Connection(
            id="",  # Will be generated
            source_block_id=source_block_id,
            source_output_name=source_output_name,
            target_block_id=target_block_id,
            target_input_name=target_input_name
        )
        
        # Save to repository
        created_connection = self._connection_repo.create(connection)
        
        # Emit event
        self._event_bus.publish(ConnectionCreated(
            project_id=target_block.project_id,  # Use target block's project
            data={
                "id": created_connection.id,
                "source_block_id": source_block_id,
                "source_output_name": source_output_name,
                "target_block_id": target_block_id,
                "target_input_name": target_input_name
            }
        ))
        
        # Emit status changed for both blocks (connection affects status of both)
        self._event_bus.publish(BlockChanged(
            project_id=target_block.project_id,
            data={
                "block_id": target_block_id,
                "change_type": "connection"
            }
        ))
        # Also emit for source block (ShowManager status depends on connections)
        self._event_bus.publish(BlockChanged(
            project_id=source_block.project_id,
            data={
                "block_id": source_block_id,
                "change_type": "connection"
            }
        ))
        
        Log.info(
            f"ConnectionService: Connected {source_block.name}.{source_output_name} -> "
            f"{target_block.name}.{target_input_name}"
        )
        
        # Note: Expected outputs recalculation is handled by the facade after connection is created
        # This keeps ConnectionService focused on connection management only
        
        return created_connection
    
    def disconnect_blocks(self, connection_id: str) -> None:
        """
        Disconnect blocks by connection ID.
        
        Args:
            connection_id: Connection identifier
            
        Raises:
            ValueError: If connection not found
        """
        connection = self._connection_repo.get(connection_id)
        if not connection:
            raise ValueError(f"Connection with id '{connection_id}' not found")
        
        # Get project_id from target block
        target_block = self._block_repo.get_by_id(connection.target_block_id)
        project_id = target_block.project_id if target_block else None
        
        # Delete connection
        self._connection_repo.delete(connection_id)
        
        # Emit event
        self._event_bus.publish(ConnectionRemoved(
            project_id=project_id,
            data={
                "id": connection_id,
                "source_block_id": connection.source_block_id,
                "target_block_id": connection.target_block_id
            }
        ))
        
        # Emit status changed for both blocks (connection removal affects status of both)
        if project_id:
            self._event_bus.publish(BlockChanged(
                project_id=project_id,
                data={
                    "block_id": connection.target_block_id,
                    "change_type": "connection"
                }
            ))
            # Also emit for source block (ShowManager status depends on connections)
            source_block = self._block_repo.get_by_id(connection.source_block_id)
            if source_block:
                self._event_bus.publish(BlockChanged(
                    project_id=source_block.project_id,
                    data={
                        "block_id": connection.source_block_id,
                        "change_type": "connection"
                    }
                ))
        
        Log.info(f"ConnectionService: Disconnected connection {connection_id}")
    
    def disconnect_by_target(self, target_block_id: str, target_input_name: str) -> None:
        """
        Disconnect all connections by target block and input port name.
        
        Deletes all connections to the specified input port.
        
        Args:
            target_block_id: Target block identifier
            target_input_name: Target input port name
            
        Raises:
            ValueError: If no connections found
        """
        connections = self._connection_repo.list_by_target(target_block_id, target_input_name)
        if not connections:
            raise ValueError(
                f"No connections found for target '{target_block_id}.{target_input_name}'"
            )
        
        # Disconnect all connections to this input port
        for connection in connections:
            self.disconnect_blocks(connection.id)
    
    def validate_connection(
        self,
        source_block_id: str,
        source_output_name: str,
        target_block_id: str,
        target_input_name: str
    ) -> bool:
        """
        Validate if a connection would be valid.
        
        Does not create the connection, just validates it would work.
        
        Args:
            source_block_id: Source block identifier
            source_output_name: Output port name on source block
            target_block_id: Target block identifier
            target_input_name: Input port name on target block
            
        Returns:
            True if connection would be valid, False otherwise
        """
        try:
            # Get source block
            source_block = self._block_repo.get_by_id(source_block_id)
            if not source_block:
                return False
            
            # Get target block
            target_block = self._block_repo.get_by_id(target_block_id)
            if not target_block:
                return False
            
            # Validate ports exist
            if not source_block.has_port(source_output_name, PortDirection.OUTPUT):
                return False
            if not target_block.has_port(target_input_name, PortDirection.INPUT):
                return False
            
            # Validate port compatibility
            source_port_type = source_block.get_port_type(source_output_name, PortDirection.OUTPUT)
            target_port_type = target_block.get_port_type(target_input_name, PortDirection.INPUT)
            
            if not source_port_type.is_compatible_with(target_port_type):
                return False
            
            # Note: Multiple connections to the same input port are now allowed
            
            return True
            
        except Exception as e:
            Log.debug(f"ConnectionService: Validation error: {e}")
            return False
    
    def list_connections_by_block(self, block_id: str) -> List[ConnectionSummary]:
        """
        List all connections involving a block (summary view).
        
        Args:
            block_id: Block identifier
            
        Returns:
            List of ConnectionSummary entries (incoming + outgoing)
        """
        return self._connection_repo.list_connection_summaries_by_block(block_id)
    
    def list_connections_by_project(self, project_id: str) -> List[ConnectionSummary]:
        """
        List all connections in a project (summary view).
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of ConnectionSummary entries
        """
        return self._connection_repo.list_connection_summaries_by_project(project_id)
    
    def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID."""
        return self._connection_repo.get(connection_id)

    def get_connection_detail(self, connection_id: str) -> Optional[Connection]:
        """
        Load the full connection entity (for commands that need details).
        """
        return self._connection_repo.load_connection_detail(connection_id)

