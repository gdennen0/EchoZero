"""
Connection repository interface

Defines the contract for connection data access.

SIMPLIFIED: Connections now reference blocks directly by block_id + port_name.
No separate Port entities.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.features.connections.domain.connection import Connection
from src.features.connections.domain.connection_summary import ConnectionSummary


class ConnectionRepository(ABC):
    """Repository interface for Connection entities"""
    
    @abstractmethod
    def create(self, connection: Connection) -> Connection:
        """
        Create a new connection.
        
        Args:
            connection: Connection entity to create
            
        Returns:
            Created connection with generated ID
            
        Raises:
            ValueError: If connection violates constraints:
                - Target input already connected
                - Ports are not compatible
                - Source/target blocks don't exist
        """
        pass
    
    @abstractmethod
    def get(self, connection_id: str) -> Optional[Connection]:
        """
        Get connection by ID.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Connection entity or None if not found
        """
        pass
    
    @abstractmethod
    def delete(self, connection_id: str) -> None:
        """
        Delete connection by ID.
        
        Args:
            connection_id: Connection identifier
        """
        pass
    
    @abstractmethod
    def delete_by_target(self, target_block_id: str, target_input_name: str) -> None:
        """
        Delete connection by target block and input port name.
        
        This is used when disconnecting an input port.
        
        Args:
            target_block_id: Target block identifier
            target_input_name: Target input port name
        """
        pass
    
    @abstractmethod
    def list_by_project(self, project_id: str) -> List[Connection]:
        """
        List all connections in a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of connections in the project
        """
        pass
    
    @abstractmethod
    def list_connection_summaries_by_project(self, project_id: str) -> List[ConnectionSummary]:
        """
        List lightweight connection summaries for a project.
        """
        pass
    
    @abstractmethod
    def list_by_block(self, block_id: str) -> List[Connection]:
        """
        List all connections involving a block.
        
        Args:
            block_id: Block identifier
            
        Returns:
            List of connections (both incoming and outgoing)
        """
        pass
    
    @abstractmethod
    def list_connection_summaries_by_block(self, block_id: str) -> List[ConnectionSummary]:
        """List summary metadata for connections involving a block."""
        pass
    
    @abstractmethod
    def find_by_target(self, target_block_id: str, target_input_name: str) -> Optional[Connection]:
        """
        Find a connection by target block and input port name.
        
        Returns the first matching connection (for backward compatibility).
        Use list_by_target() to get all connections to an input port.
        
        Args:
            target_block_id: Target block identifier
            target_input_name: Target input port name
            
        Returns:
            Connection entity or None if not found
        """
        pass
    
    @abstractmethod
    def list_by_target(self, target_block_id: str, target_input_name: str) -> List[Connection]:
        """
        List all connections by target block and input port name.
        
        Allows multiple connections to the same input port (e.g., Event ports).
        
        Args:
            target_block_id: Target block identifier
            target_input_name: Target input port name
            
        Returns:
            List of Connection entities (empty if none found)
        """
        pass
    
    @abstractmethod
    def load_connection_detail(self, connection_id: str) -> Optional[Connection]:
        """Load full connection details when needed."""
        pass

