"""
SQLite implementation of ActionSetRepository

Handles persistence of ActionSet entities in SQLite database.
"""
from typing import Optional, List
from datetime import datetime

from src.features.projects.domain.action_set import ActionSet
from src.features.projects.domain.action_set_repository import ActionSetRepository
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SQLiteActionSetRepository(ActionSetRepository):
    """SQLite implementation of ActionSetRepository"""
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def create(self, action_set: ActionSet) -> ActionSet:
        """
        Create a new action set.
        
        Args:
            action_set: ActionSet entity to create
            
        Returns:
            Created action set (with generated ID if needed)
            
        Raises:
            ValueError: If action set name already exists
        """
        # Check for duplicate name
        existing = self.find_by_name(action_set.name, action_set.project_id)
        if existing:
            raise ValueError(f"Action set with name '{action_set.name}' already exists")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Insert action set
            cursor.execute("""
                INSERT INTO action_sets (
                    id, name, description, actions, project_id,
                    created_at, modified_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_set.id,
                action_set.name,
                action_set.description,
                Database.json_encode(action_set.to_dict()["actions"]),
                action_set.project_id,
                action_set.created_at.isoformat(),
                action_set.modified_at.isoformat(),
                Database.json_encode(action_set.metadata)
            ))
            
            Log.info(f"Created action set: {action_set.name} (id: {action_set.id})")
            return action_set
    
    def get(self, action_set_id: str) -> Optional[ActionSet]:
        """
        Get action set by ID.
        
        Args:
            action_set_id: Action set identifier
            
        Returns:
            ActionSet entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, actions, project_id,
                       created_at, modified_at, metadata
                FROM action_sets
                WHERE id = ?
            """, (action_set_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_action_set(row)
    
    def find_by_name(self, name: str, project_id: Optional[str] = None) -> Optional[ActionSet]:
        """
        Find action set by name.
        
        Args:
            name: Action set name
            project_id: Optional project ID to scope search
            
        Returns:
            ActionSet entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if project_id:
                cursor.execute("""
                    SELECT id, name, description, actions, project_id,
                           created_at, modified_at, metadata
                    FROM action_sets
                    WHERE name = ? AND project_id = ?
                """, (name, project_id))
            else:
                cursor.execute("""
                    SELECT id, name, description, actions, project_id,
                           created_at, modified_at, metadata
                    FROM action_sets
                    WHERE name = ?
                """, (name,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_action_set(row)
    
    def list_all(self, project_id: Optional[str] = None) -> List[ActionSet]:
        """
        List all action sets.
        
        Args:
            project_id: Optional project ID to filter by
            
        Returns:
            List of ActionSet entities
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if project_id:
                cursor.execute("""
                    SELECT id, name, description, actions, project_id,
                           created_at, modified_at, metadata
                    FROM action_sets
                    WHERE project_id = ?
                    ORDER BY name
                """, (project_id,))
            else:
                cursor.execute("""
                    SELECT id, name, description, actions, project_id,
                           created_at, modified_at, metadata
                    FROM action_sets
                    ORDER BY name
                """)
            
            rows = cursor.fetchall()
            return [self._row_to_action_set(row) for row in rows]
    
    def list_by_project(self, project_id: str) -> List[ActionSet]:
        """
        List all action sets for a project (follows blocks pattern).
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of ActionSet entities for the project
        """
        return self.list_all(project_id=project_id)
    
    def update(self, action_set: ActionSet) -> None:
        """
        Update existing action set.
        
        Args:
            action_set: ActionSet entity to update
            
        Raises:
            ValueError: If action set not found
        """
        # Verify action set exists
        existing = self.get(action_set.id)
        if existing is None:
            raise ValueError(f"Action set with id '{action_set.id}' not found")
        
        # Check for duplicate name (if name changed)
        if action_set.name != existing.name:
            duplicate = self.find_by_name(action_set.name, action_set.project_id)
            if duplicate and duplicate.id != action_set.id:
                raise ValueError(f"Action set with name '{action_set.name}' already exists")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE action_sets
                SET name = ?, description = ?, actions = ?, project_id = ?,
                    modified_at = ?, metadata = ?
                WHERE id = ?
            """, (
                action_set.name,
                action_set.description,
                Database.json_encode(action_set.to_dict()["actions"]),
                action_set.project_id,
                action_set.modified_at.isoformat(),
                Database.json_encode(action_set.metadata),
                action_set.id
            ))
            
            Log.info(f"Updated action set: {action_set.name} (id: {action_set.id})")
    
    def delete(self, action_set_id: str) -> None:
        """
        Delete action set by ID.
        
        Args:
            action_set_id: Action set identifier
            
        Raises:
            ValueError: If action set not found
        """
        # Verify action set exists
        existing = self.get(action_set_id)
        if existing is None:
            raise ValueError(f"Action set with id '{action_set_id}' not found")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM action_sets WHERE id = ?", (action_set_id,))
            
            Log.info(f"Deleted action set: {existing.name} (id: {action_set_id})")
    
    def delete_by_project(self, project_id: str) -> int:
        """
        Delete all action sets for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Number of action sets deleted
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM action_sets WHERE project_id = ?", (project_id,))
            deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                Log.debug(f"Deleted {deleted_count} action set(s) for project {project_id}")
            
            return deleted_count
    
    def _row_to_action_set(self, row) -> ActionSet:
        """
        Convert database row to ActionSet entity.
        
        Args:
            row: Database row (sqlite3.Row)
            
        Returns:
            ActionSet entity
        """
        from src.features.projects.domain.action_set import ActionItem
        
        # Parse actions from JSON
        actions_data = Database.json_decode(row["actions"]) if row["actions"] else []
        actions = [ActionItem.from_dict(action_data) for action_data in actions_data]
        
        metadata = {}
        if row.get("metadata"):
            metadata = Database.json_decode(row["metadata"]) or {}
        
        return ActionSet(
            id=row["id"],
            name=row["name"],
            description=row.get("description") or "",
            actions=actions,
            project_id=row.get("project_id"),
            created_at=datetime.fromisoformat(row["created_at"]),
            modified_at=datetime.fromisoformat(row["modified_at"]),
            metadata=metadata
        )

