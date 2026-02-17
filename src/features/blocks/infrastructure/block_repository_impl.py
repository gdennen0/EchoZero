"""
SQLite implementation of BlockRepository

Handles persistence of Block entities in SQLite database.
Port definitions (inputs/outputs) are stored as JSON.
"""
from typing import Optional, List, Dict

from src.features.blocks.domain.block import Block
from src.shared.domain.entities import BlockSummary
from src.features.blocks.domain.port import Port
from src.features.blocks.domain.block_repository import BlockRepository
from src.shared.domain.value_objects.port_type import get_port_type
from src.features.blocks.domain.port_direction import PortDirection
# BlockPosition removed - backend-only architecture doesn't need UI coordinates
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SQLiteBlockRepository(BlockRepository):
    """SQLite implementation of BlockRepository"""
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def create(self, block: Block) -> Block:
        """
        Create a new block.
        
        Args:
            block: Block entity to create
            
        Returns:
            Created block (with generated ID if needed)
            
        Raises:
            ValueError: If block name already exists in project
        """
        # Check if name already exists in project
        existing = self.find_by_name(block.project_id, block.name)
        if existing and existing.id != block.id:
            raise ValueError(f"Block with name '{block.name}' already exists in project")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Serialize unified ports to JSON (using composite keys)
            ports_json = self._serialize_unified_ports(block.ports)
            metadata_to_save = block.metadata or {}
            
            cursor.execute("""
                INSERT INTO blocks (id, project_id, name, type, ports, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                block.id,
                block.project_id,
                block.name,
                block.type,
                ports_json,
                Database.json_encode(metadata_to_save)
            ))
            
            Log.debug(f"Created block: {block.name} in project {block.project_id}")
            return block
    
    def get(self, project_id: str, block_id: str) -> Optional[Block]:
        """
        Get block by ID.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            
        Returns:
            Block entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, name, type, ports, metadata
                FROM blocks
                WHERE id = ? AND project_id = ?
            """, (block_id, project_id))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_block(row)
    
    def update(self, block: Block) -> None:
        """
        Update existing block.
        
        Args:
            block: Block entity to update
            
        Raises:
            ValueError: If block not found
        """
        # Verify block exists
        existing = self.get(block.project_id, block.id)
        if existing is None:
            raise ValueError(f"Block with id '{block.id}' not found in project {block.project_id}")
        
        # Track what changed for logging
        changes = []
        
        # Check name uniqueness (if name changed)
        if existing.name != block.name:
            name_conflict = self.find_by_name(block.project_id, block.name)
            if name_conflict and name_conflict.id != block.id:
                raise ValueError(f"Block with name '{block.name}' already exists in project")
            changes.append(f"name: '{existing.name}' -> '{block.name}'")
        
        if existing.type != block.type:
            changes.append(f"type: '{existing.type}' -> '{block.type}'")
        
        # Check port changes (unified structure)
        existing_ports = existing.ports
        new_ports = block.ports
        
        # Compare ports by composite key
        existing_keys = set(existing_ports.keys())
        new_keys = set(new_ports.keys())
        added_keys = new_keys - existing_keys
        removed_keys = existing_keys - new_keys
        
        for key in sorted(added_keys):
            port = new_ports[key]
            changes.append(f"port.{key}: added (type: {port.port_type.name}, direction: {port.direction.value})")
        
        for key in sorted(removed_keys):
            port = existing_ports[key]
            changes.append(f"port.{key}: removed (was type: {port.port_type.name}, direction: {port.direction.value})")
        
        # Check changed ports (same key, different properties)
        for key in sorted(existing_keys & new_keys):
            existing_port = existing_ports[key]
            new_port = new_ports[key]
            if existing_port.port_type.name != new_port.port_type.name:
                changes.append(f"port.{key}: type '{existing_port.port_type.name}' -> '{new_port.port_type.name}'")
            if existing_port.metadata != new_port.metadata:
                changes.append(f"port.{key}: metadata changed")
        
        # Check metadata changes (show actual value changes)
        if existing.metadata != block.metadata:
            existing_keys = set(existing.metadata.keys())
            new_keys = set(block.metadata.keys())
            added_metadata = new_keys - existing_keys
            removed_metadata = existing_keys - new_keys
            
            # Helper to format value for display
            def format_value(value, max_length=100):
                """Format a value for display, truncating if too long"""
                if value is None:
                    return "None"
                value_str = str(value)
                if len(value_str) > max_length:
                    return value_str[:max_length] + "..."
                return value_str
            
            if added_metadata:
                for key in sorted(added_metadata):
                    value = format_value(block.metadata[key])
                    changes.append(f"metadata.{key}: added '{value}'")
            
            if removed_metadata:
                for key in sorted(removed_metadata):
                    value = format_value(existing.metadata[key])
                    changes.append(f"metadata.{key}: removed (was '{value}')")
            
            # Show changed metadata with old -> new values
            for key in sorted(existing_keys & new_keys):
                if existing.metadata[key] != block.metadata[key]:
                    old_value = format_value(existing.metadata[key])
                    new_value = format_value(block.metadata[key])
                    changes.append(f"metadata.{key}: '{old_value}' -> '{new_value}'")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # CRITICAL: Preserve metadata if block object has empty metadata but database has metadata
            # This prevents accidental metadata loss when updating blocks
            cursor.execute("SELECT metadata FROM blocks WHERE id = ? AND project_id = ?", (block.id, block.project_id))
            row_before = cursor.fetchone()
            metadata_in_db = Database.json_decode(row_before[0]) if row_before and row_before[0] else {}
            metadata_to_save = block.metadata or {}
            
            # If database has metadata but block object doesn't, preserve database metadata
            # This prevents clearing metadata when updating blocks for non-metadata changes
            if metadata_in_db and not metadata_to_save:
                metadata_to_save = metadata_in_db
                Log.warning(f"BlockRepository: Preserved metadata from database for block '{block.name}' - block object had empty metadata")
            
            ports_json = self._serialize_unified_ports(block.ports)
            cursor.execute("""
                UPDATE blocks
                SET name = ?, type = ?, ports = ?, metadata = ?
                WHERE id = ? AND project_id = ?
            """, (
                block.name,
                block.type,
                ports_json,
                Database.json_encode(metadata_to_save),
                block.id,
                block.project_id
            ))
            
            # Log with change details
            if changes:
                changes_str = ", ".join(changes)
                Log.info(f"Updated block: {block.name} - Changes: {changes_str}")
            else:
                # No changes - log at debug level to reduce spam
                Log.debug(f"Updated block: {block.name} - No changes detected")
    
    def delete(self, project_id: str, block_id: str) -> None:
        """
        Delete block by ID.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            
        Raises:
            ValueError: If block not found
        """
        # Verify block exists
        existing = self.get(project_id, block_id)
        if existing is None:
            raise ValueError(f"Block with id '{block_id}' not found in project {project_id}")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM blocks WHERE id = ? AND project_id = ?", (block_id, project_id))
            
            Log.info(f"Deleted block: {existing.name}")
    
    def list_by_project(self, project_id: str) -> List[Block]:
        """
        List all blocks in a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of blocks in the project
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, name, type, ports, metadata
                FROM blocks
                WHERE project_id = ?
                ORDER BY name
            """, (project_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_block(row) for row in rows]

    def list_block_summaries(self, project_id: str) -> List[BlockSummary]:
        """
        Return only the minimal block metadata for lazy listing.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, type
                FROM blocks
                WHERE project_id = ?
                ORDER BY name
            """, (project_id,))
            rows = cursor.fetchall()
            return [BlockSummary(id=row["id"], name=row["name"], type=row["type"]) for row in rows]

    def load_block_detail(self, project_id: str, block_id: str) -> Optional[Block]:
        """
        Load the full block (inputs/outputs) when needed.
        """
        return self.get(project_id, block_id)
    
    def find_by_name(self, project_id: str, name: str) -> Optional[Block]:
        """
        Find block by name within a project.
        
        Args:
            project_id: Project identifier
            name: Block name
            
        Returns:
            Block entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, name, type, ports, metadata
                FROM blocks
                WHERE project_id = ? AND LOWER(name) = LOWER(?)
            """, (project_id, name))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_block(row)
    
    def get_by_id(self, block_id: str) -> Optional[Block]:
        """
        Get block by ID (across all projects).
        
        This is a convenience method for finding blocks when you only have the ID.
        Use get(project_id, block_id) when you know the project for better performance.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Block entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, name, type, ports, metadata
                FROM blocks
                WHERE id = ?
            """, (block_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_block(row)
    
    def _row_to_block(self, row) -> Block:
        """
        Convert database row to Block entity.
        
        Uses unified ports format with composite keys.
        
        Args:
            row: Database row (sqlite3.Row)
            
        Returns:
            Block entity
        """
        # Deserialize unified ports with composite keys
        ports = self._deserialize_unified_ports(row["ports"]) if row["ports"] else {}
        
        raw_metadata = row["metadata"]
        metadata_from_db = Database.json_decode(raw_metadata) if raw_metadata else {}
        
        return Block(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            type=row["type"],
            ports=ports,
            metadata=metadata_from_db
        )
    
    def _serialize_unified_ports(self, ports: dict) -> str:
        """
        Serialize unified ports (Dict[str, Port]) to JSON string.
        
        Uses composite keys: "{direction}:{port_name}"
        
        Args:
            ports: Dictionary mapping composite keys to Port objects
            
        Returns:
            JSON string representation
        """
        port_dict = {}
        for key, port in ports.items():
            port_dict[key] = {
                "port_type": port.port_type.name,
                "direction": port.direction.value,
                "metadata": port.metadata
            }
        return Database.json_encode(port_dict) or "{}"
    
    def _deserialize_unified_ports(self, json_str: Optional[str]) -> dict:
        """
        Deserialize JSON string to unified ports (Dict[str, Port]).
        
        Args:
            json_str: JSON string representation with composite keys
            
        Returns:
            Dictionary mapping composite keys to Port objects
        """
        if not json_str:
            return {}
        
        port_dict = Database.json_decode(json_str)
        if not isinstance(port_dict, dict):
            return {}
        
        result = {}
        for key, port_data in port_dict.items():
            if isinstance(port_data, dict):
                port_type_name = port_data.get("port_type", port_data.get("type", ""))
                direction_str = port_data.get("direction", "")
                metadata = port_data.get("metadata", {})
                
                try:
                    direction = PortDirection.from_string(direction_str)
                    port_type = get_port_type(port_type_name)
                    # Extract port name from composite key
                    port_name = key.split(":", 1)[1] if ":" in key else key
                    result[key] = Port(
                        name=port_name,
                        port_type=port_type,
                        direction=direction,
                        metadata=metadata
                    )
                except (ValueError, KeyError):
                    # Skip invalid ports
                    continue
        
        return result

