"""
SQLite implementation of UIStateRepository

Handles persistence of UI-specific state (block positions, zoom levels, etc.)
that is separate from domain data but needs to be persisted across sessions.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class UIStateRepository:
    """
    Repository for UI state persistence.
    
    UI state is project-specific data like:
    - Block positions on the canvas
    - Zoom levels
    - Panel layouts
    - View settings
    
    This data is cleared when projects are loaded (session-specific).
    """
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def set(self, state_type: str, entity_id: Optional[str], data: Dict[str, Any]) -> None:
        """
        Set UI state for a given type and entity.
        
        Args:
            state_type: Type of UI state (e.g., 'block_position', 'zoom_level')
            entity_id: Optional entity ID this state is associated with (e.g., block_id)
            data: Dictionary of state data to store
        """
        # Generate unique ID
        if entity_id:
            state_id = f"{state_type}:{entity_id}"
        else:
            state_id = state_type
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Check if state exists
            cursor.execute("""
                SELECT id FROM ui_state WHERE id = ?
            """, (state_id,))
            
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing
                cursor.execute("""
                    UPDATE ui_state
                    SET data = ?, updated_at = ?
                    WHERE id = ?
                """, (
                    Database.json_encode(data),
                    datetime.now().isoformat(),
                    state_id
                ))
                Log.debug(f"Updated UI state: {state_type} (entity: {entity_id})")
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO ui_state (id, type, entity_id, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    state_id,
                    state_type,
                    entity_id,
                    Database.json_encode(data),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                Log.debug(f"Created UI state: {state_type} (entity: {entity_id})")
    
    def get(self, state_type: str, entity_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get UI state for a given type and entity.
        
        Args:
            state_type: Type of UI state
            entity_id: Optional entity ID
            
        Returns:
            Dictionary of state data or None if not found
        """
        # Generate ID
        if entity_id:
            state_id = f"{state_type}:{entity_id}"
        else:
            state_id = state_type
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT data FROM ui_state WHERE id = ?
            """, (state_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return Database.json_decode(row[0])
    
    def get_by_type(self, state_type: str) -> List[Dict[str, Any]]:
        """
        Get all UI state entries of a given type.
        
        Args:
            state_type: Type of UI state
            
        Returns:
            List of state data dictionaries
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_id, data FROM ui_state WHERE type = ?
            """, (state_type,))
            
            results = []
            for row in cursor.fetchall():
                state_data = Database.json_decode(row[1])
                state_data['entity_id'] = row[0]  # Include entity_id in result
                results.append(state_data)
            
            return results
    
    def delete(self, state_type: str, entity_id: Optional[str] = None) -> None:
        """
        Delete UI state for a given type and entity.
        
        Args:
            state_type: Type of UI state
            entity_id: Optional entity ID
        """
        # Generate ID
        if entity_id:
            state_id = f"{state_type}:{entity_id}"
        else:
            state_id = state_type
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM ui_state WHERE id = ?
            """, (state_id,))
            
            Log.debug(f"Deleted UI state: {state_type} (entity: {entity_id})")
    
    def clear(self) -> None:
        """
        Clear all UI state.
        
        Called when switching projects or resetting the application.
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ui_state")
            Log.debug("Cleared all UI state")
    
    def serialize_for_project(self, project_id: str) -> Dict[str, Any]:
        """
        Serialize all UI state for a project to dict format for project file.
        
        Returns a dictionary mapping state_type -> list of state entries.
        Each entry contains entity_id and data.
        
        Example:
        {
            "block_position": [
                {"entity_id": "block1", "data": {"x": 100, "y": 200}},
                {"entity_id": "block2", "data": {"x": 300, "y": 400}}
            ]
        }
        
        Args:
            project_id: Project ID (currently not used, but kept for future filtering)
            
        Returns:
            Dictionary of serialized UI state
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT type, entity_id, data FROM ui_state
                ORDER BY type, entity_id
            """)
            
            serialized = {}
            for row in cursor.fetchall():
                state_type = row[0]
                entity_id = row[1]
                state_data = Database.json_decode(row[2])
                
                if state_type not in serialized:
                    serialized[state_type] = []
                
                serialized[state_type].append({
                    "entity_id": entity_id,
                    "data": state_data
                })
            
            Log.debug(f"Serialized {sum(len(entries) for entries in serialized.values())} UI state entries")
            return serialized
    
    def deserialize_from_project(self, project_id: str, serialized: Dict[str, Any]) -> None:
        """
        Restore UI state from serialized dict format.
        
        Takes the same format as serialize_for_project() and restores
        all state to the ui_state table.
        
        Args:
            project_id: Project ID (currently not used, but kept for future filtering)
            serialized: Dictionary of serialized UI state from project file
        """
        if not serialized:
            Log.debug("No UI state to deserialize")
            return
        
        restored_count = 0
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            for state_type, entries in serialized.items():
                if not isinstance(entries, list):
                    Log.warning(f"Invalid UI state format for type '{state_type}': expected list")
                    continue
                
                for entry in entries:
                    if not isinstance(entry, dict):
                        Log.warning(f"Invalid UI state entry for type '{state_type}': expected dict")
                        continue
                    
                    entity_id = entry.get("entity_id")
                    state_data = entry.get("data")
                    
                    if not state_data or not isinstance(state_data, dict):
                        Log.warning(f"Invalid UI state data for type '{state_type}', entity '{entity_id}'")
                        continue
                    
                    # Use existing set() method to restore
                    try:
                        self.set(state_type, entity_id, state_data)
                        restored_count += 1
                    except Exception as e:
                        Log.warning(f"Failed to restore UI state for type '{state_type}', entity '{entity_id}': {e}")
                        # Continue with other entries
        
        if restored_count > 0:
            Log.info(f"Restored {restored_count} UI state entry(ies) from project file")
        else:
            Log.debug("No UI state entries restored")


