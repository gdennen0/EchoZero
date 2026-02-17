"""
SQLite implementation of ConnectionRepository

Handles persistence of Connection entities in SQLite database.
Connections reference blocks directly by block_id + port_name.
"""
from typing import Optional, List
import sqlite3

from src.features.connections.domain.connection import Connection
from src.features.connections.domain.connection_summary import ConnectionSummary
from src.features.connections.domain.connection_repository import ConnectionRepository
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SQLiteConnectionRepository(ConnectionRepository):
    """SQLite implementation of ConnectionRepository"""
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def create(self, connection: Connection) -> Connection:
        """
        Create a new connection.
        
        Args:
            connection: Connection entity to create
            
        Returns:
            Created connection (with generated ID if needed)
            
        Raises:
            ValueError: If connection violates constraints:
                - Source/target blocks don't exist (enforced by FOREIGN KEY)
        """
        # Note: Multiple connections to the same input port are now allowed
        # (removed UNIQUE constraint to support multiple EventDataItems per events input)
        
        with self.db.transaction() as conn:
            try:
                cursor = conn.cursor()
                
                # Look up block names for readability
                cursor.execute("SELECT name FROM blocks WHERE id = ?", (connection.source_block_id,))
                source_row = cursor.fetchone()
                source_block_name = source_row["name"] if source_row else None
                
                cursor.execute("SELECT name FROM blocks WHERE id = ?", (connection.target_block_id,))
                target_row = cursor.fetchone()
                target_block_name = target_row["name"] if target_row else None
                
                # Insert connection with block names for readability
                cursor.execute("""
                    INSERT INTO connections (id, source_block_id, source_block_name, source_output_name, 
                                            target_block_id, target_block_name, target_input_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    connection.id,
                    connection.source_block_id,
                    source_block_name,
                    connection.source_output_name,
                    connection.target_block_id,
                    target_block_name,
                    connection.target_input_name
                ))
                
                Log.debug(
                    f"Created connection: {source_block_name}.{connection.source_output_name} -> "
                    f"{target_block_name}.{connection.target_input_name}"
                )
                return connection
                
            except sqlite3.IntegrityError as e:
                # Handle foreign key violations
                error_msg = str(e).lower()
                if "foreign key" in error_msg:
                    raise ValueError(
                        f"Source block '{connection.source_block_id}' or target block "
                        f"'{connection.target_block_id}' does not exist"
                    )
                else:
                    raise ValueError(f"Failed to create connection: {e}")
    
    def get(self, connection_id: str) -> Optional[Connection]:
        """
        Get connection by ID.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Connection entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source_block_id, source_block_name, source_output_name, 
                       target_block_id, target_block_name, target_input_name
                FROM connections
                WHERE id = ?
            """, (connection_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_connection(row)
    
    def delete(self, connection_id: str) -> None:
        """
        Delete connection by ID.
        
        Args:
            connection_id: Connection identifier
        """
        # Verify connection exists
        existing = self.get(connection_id)
        if existing is None:
            Log.warning(f"Attempted to delete non-existent connection: {connection_id}")
            return
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM connections WHERE id = ?", (connection_id,))
            
            Log.info(f"Deleted connection: {existing}")
    
    def delete_by_target(self, target_block_id: str, target_input_name: str) -> None:
        """
        Delete all connections by target block and input port name.
        
        This is used when disconnecting an input port.
        Deletes all connections to the specified input (multiple connections are now allowed).
        
        Args:
            target_block_id: Target block identifier
            target_input_name: Target input port name
        """
        # Get count of connections to be deleted
        connections = self.list_by_target(target_block_id, target_input_name)
        if not connections:
            Log.warning(
                f"Attempted to delete non-existent connection: "
                f"target {target_block_id}.{target_input_name}"
            )
            return
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM connections
                WHERE target_block_id = ? AND target_input_name = ?
            """, (target_block_id, target_input_name))
            
            Log.info(f"Deleted {len(connections)} connection(s) by target: {target_block_id}.{target_input_name}")
    
    def list_by_project(self, project_id: str) -> List[Connection]:
        """
        List all connections in a project.
        
        This finds connections by joining through blocks that belong to the project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of connections in the project
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT c.id, c.source_block_id, c.source_block_name, c.source_output_name, 
                       c.target_block_id, c.target_block_name, c.target_input_name
                FROM connections c
                INNER JOIN blocks source_block ON c.source_block_id = source_block.id
                INNER JOIN blocks target_block ON c.target_block_id = target_block.id
                WHERE source_block.project_id = ? OR target_block.project_id = ?
            """, (project_id, project_id))
            
            rows = cursor.fetchall()
            return [self._row_to_connection(row) for row in rows]

    def list_connection_summaries_by_project(self, project_id: str) -> List[ConnectionSummary]:
        """
        Return lightweight connection summaries for a project.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT c.id, c.source_block_id, c.source_block_name, c.source_output_name, 
                       c.target_block_id, c.target_block_name, c.target_input_name
                FROM connections c
                INNER JOIN blocks source_block ON c.source_block_id = source_block.id
                INNER JOIN blocks target_block ON c.target_block_id = target_block.id
                WHERE source_block.project_id = ? OR target_block.project_id = ?
            """, (project_id, project_id))
            rows = cursor.fetchall()
            return [self._row_to_summary(row) for row in rows]
    
    def list_by_block(self, block_id: str) -> List[Connection]:
        """
        List all connections involving a block.
        
        This includes both:
        - Connections where block is the source (outgoing)
        - Connections where block is the target (incoming)
        
        Args:
            block_id: Block identifier
            
        Returns:
            List of connections (both incoming and outgoing)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source_block_id, source_block_name, source_output_name, 
                       target_block_id, target_block_name, target_input_name
                FROM connections
                WHERE source_block_id = ? OR target_block_id = ?
            """, (block_id, block_id))
            
            rows = cursor.fetchall()
            return [self._row_to_connection(row) for row in rows]

    def list_connection_summaries_by_block(self, block_id: str) -> List[ConnectionSummary]:
        """
        Return lightweight summaries for connections related to a block.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source_block_id, source_block_name, source_output_name, 
                       target_block_id, target_block_name, target_input_name
                FROM connections
                WHERE source_block_id = ? OR target_block_id = ?
            """, (block_id, block_id))
            rows = cursor.fetchall()
            return [self._row_to_summary(row) for row in rows]
    
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
        connections = self.list_by_target(target_block_id, target_input_name)
        return connections[0] if connections else None
    
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
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, source_block_id, source_block_name, source_output_name, 
                       target_block_id, target_block_name, target_input_name
                FROM connections
                WHERE target_block_id = ? AND target_input_name = ?
            """, (target_block_id, target_input_name))
            
            rows = cursor.fetchall()
            return [self._row_to_connection(row) for row in rows]

    def load_connection_detail(self, connection_id: str) -> Optional[Connection]:
        """
        Load detailed connection metadata when a command needs the full entity.
        """
        return self.get(connection_id)
    
    def _row_to_connection(self, row) -> Connection:
        """
        Convert database row to Connection entity.
        
        Args:
            row: Database row (sqlite3.Row)
            
        Returns:
            Connection entity
        """
        return Connection(
            id=row["id"],
            source_block_id=row["source_block_id"],
            source_output_name=row["source_output_name"],
            target_block_id=row["target_block_id"],
            target_input_name=row["target_input_name"]
        )

    def _row_to_summary(self, row) -> ConnectionSummary:
        return ConnectionSummary(
            id=row["id"],
            source_block_id=row["source_block_id"],
            source_output_name=row["source_output_name"],
            target_block_id=row["target_block_id"],
            target_input_name=row["target_input_name"]
        )

