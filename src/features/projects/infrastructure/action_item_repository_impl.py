"""
SQLite implementation of ActionItemRepository

Handles persistence of ActionItem entities in SQLite database.
"""
from typing import Optional, List
from datetime import datetime

from src.features.projects.domain.action_set import ActionItem
from src.features.projects.domain.action_item_repository import ActionItemRepository
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SQLiteActionItemRepository(ActionItemRepository):
    """SQLite implementation of ActionItemRepository"""
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def create(self, action_item: ActionItem) -> ActionItem:
        """
        Create a new action item.
        
        Args:
            action_item: ActionItem entity to create
            
        Returns:
            Created action item
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO action_items (
                    id, action_set_id, project_id, action_type, block_id, block_name,
                    action_name, action_description, action_args, order_index,
                    created_at, modified_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_item.id,
                action_item.action_set_id,
                action_item.project_id,
                action_item.action_type,
                action_item.block_id,
                action_item.block_name,
                action_item.action_name,
                action_item.action_description,
                Database.json_encode(action_item.action_args),
                action_item.order_index,
                action_item.created_at.isoformat(),
                action_item.modified_at.isoformat(),
                Database.json_encode(action_item.metadata)
            ))
            
            Log.debug(f"Created action item: {action_item.action_name} for block {action_item.block_name}")
            return action_item
    
    def get(self, action_item_id: str) -> Optional[ActionItem]:
        """Get action item by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, action_set_id, project_id, action_type, block_id, block_name,
                       action_name, action_description, action_args, order_index,
                       created_at, modified_at, metadata
                FROM action_items
                WHERE id = ?
            """, (action_item_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_action_item(row)
    
    def list_by_action_set(self, action_set_id: str) -> List[ActionItem]:
        """List all action items for an action set, ordered by order_index."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, action_set_id, project_id, action_type, block_id, block_name,
                       action_name, action_description, action_args, order_index,
                       created_at, modified_at, metadata
                FROM action_items
                WHERE action_set_id = ?
                ORDER BY order_index
            """, (action_set_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_action_item(row) for row in rows]
    
    def list_by_project(self, project_id: str) -> List[ActionItem]:
        """List all action items for a project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, action_set_id, project_id, action_type, block_id, block_name,
                       action_name, action_description, action_args, order_index,
                       created_at, modified_at, metadata
                FROM action_items
                WHERE project_id = ?
                ORDER BY action_set_id, order_index
            """, (project_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_action_item(row) for row in rows]
    
    def update(self, action_item: ActionItem) -> None:
        """Update existing action item."""
        existing = self.get(action_item.id)
        if existing is None:
            raise ValueError(f"ActionItem with id '{action_item.id}' not found")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE action_items
                SET action_set_id = ?, project_id = ?, action_type = ?, block_id = ?, block_name = ?,
                    action_name = ?, action_description = ?, action_args = ?,
                    order_index = ?, modified_at = ?, metadata = ?
                WHERE id = ?
            """, (
                action_item.action_set_id,
                action_item.project_id,
                action_item.action_type,
                action_item.block_id,
                action_item.block_name,
                action_item.action_name,
                action_item.action_description,
                Database.json_encode(action_item.action_args),
                action_item.order_index,
                action_item.modified_at.isoformat(),
                Database.json_encode(action_item.metadata),
                action_item.id
            ))
            
            Log.debug(f"Updated action item: {action_item.action_name}")
    
    def delete(self, action_item_id: str) -> None:
        """Delete action item by ID."""
        existing = self.get(action_item_id)
        if existing is None:
            raise ValueError(f"ActionItem with id '{action_item_id}' not found")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM action_items WHERE id = ?", (action_item_id,))
            
            Log.debug(f"Deleted action item: {existing.action_name}")
    
    def delete_by_action_set(self, action_set_id: str) -> int:
        """Delete all action items for an action set."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM action_items WHERE action_set_id = ?", (action_set_id,))
            deleted_count = cursor.rowcount
            
            Log.debug(f"Deleted {deleted_count} action item(s) for action set {action_set_id}")
            return deleted_count
    
    def delete_by_project(self, project_id: str) -> int:
        """Delete all action items for a project."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM action_items WHERE project_id = ?", (project_id,))
            deleted_count = cursor.rowcount
            
            Log.debug(f"Deleted {deleted_count} action item(s) for project {project_id}")
            return deleted_count
    
    def reorder(self, action_set_id: str, item_ids: List[str]) -> None:
        """Reorder action items within an action set."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            for index, item_id in enumerate(item_ids):
                cursor.execute("""
                    UPDATE action_items
                    SET order_index = ?, modified_at = ?
                    WHERE id = ? AND action_set_id = ?
                """, (index, datetime.utcnow().isoformat(), item_id, action_set_id))
            
            Log.debug(f"Reordered {len(item_ids)} action item(s) for action set {action_set_id}")
    
    def _row_to_action_item(self, row) -> ActionItem:
        """Convert database row to ActionItem entity."""
        action_args = {}
        if row["action_args"]:
            action_args = Database.json_decode(row["action_args"]) or {}
        
        metadata = {}
        if row["metadata"]:
            metadata = Database.json_decode(row["metadata"]) or {}
        
        # sqlite3.Row uses dictionary-style access, not .get()
        # block_id can be None for project-level actions
        block_id = row["block_id"] if row["block_id"] else None
        
        return ActionItem(
            id=row["id"],
            action_set_id=row["action_set_id"],
            project_id=row["project_id"],
            action_type=row["action_type"],
            block_id=block_id,
            block_name=row["block_name"] or "",
            action_name=row["action_name"],
            action_description=row["action_description"] or "",
            action_args=action_args,
            order_index=row["order_index"] or 0,
            created_at=datetime.fromisoformat(row["created_at"]),
            modified_at=datetime.fromisoformat(row["modified_at"]),
            metadata=metadata
        )

