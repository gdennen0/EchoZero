"""
SQLite implementation of DataItemRepository

Handles persistence of DataItem entities in SQLite database.
Data items are stored with metadata in database, binary data on file system.
"""
from typing import Optional, List
import json
from datetime import datetime

from src.shared.domain.entities import DataItem
from src.shared.domain.entities import AudioDataItem
from src.shared.domain.entities import EventDataItem
from src.shared.domain.entities import DataItemSummary
from src.shared.domain.repositories import DataItemRepository
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log

class SQLiteDataItemRepository(DataItemRepository):
    """
    SQLite implementation of DataItemRepository.
    
    Stores data item metadata in SQLite database.
    Binary data (audio files) stored on file system.
    """
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure data_items table exists"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_items (
                    id TEXT PRIMARY KEY,
                    block_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_path TEXT,
                    metadata TEXT,
                    FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
    
    def create(self, data_item: DataItem) -> DataItem:
        """
        Create a new data item.
        
        Args:
            data_item: DataItem entity to create
            
        Returns:
            Created data item
        """
        
        # Check if data item already exists (pass-through outputs must not be re-inserted)
        existing = None
        try:
            existing = self.get(data_item.id) if getattr(data_item, "id", None) else None
        except Exception:
            existing = None

        if existing is not None:
            return existing
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO data_items (id, block_id, name, type, created_at, file_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    data_item.id,
                    data_item.block_id,
                    data_item.name,
                    data_item.type,
                    data_item.created_at.isoformat(),
                    data_item.file_path,
                    json.dumps(data_item.to_dict())  # Store full serialization
                ))
                
                Log.debug(f"Created data item: {data_item.name} (id: {data_item.id})")
                return data_item
            except Exception as e:
                raise
    
    def get(self, data_item_id: str) -> Optional[DataItem]:
        """
        Get data item by ID.
        
        Args:
            data_item_id: Data item identifier
            
        Returns:
            DataItem entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, block_id, name, type, created_at, file_path, metadata
                FROM data_items
                WHERE id = ?
            """, (data_item_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_data_item(row)

    def load_data_item_detail(self, data_item_id: str) -> Optional[DataItem]:
        """
        Load the full DataItem entity when needed.
        """
        return self.get(data_item_id)
    
    def update(self, data_item: DataItem) -> None:
        """
        Update existing data item.
        
        Args:
            data_item: DataItem entity to update
            
        Raises:
            ValueError: If data item not found
        """
        # Verify exists
        existing = self.get(data_item.id)
        if existing is None:
            raise ValueError(f"Data item with id '{data_item.id}' not found")
        
        # Get event count if EventDataItem
        event_count = None
        if hasattr(data_item, 'get_events'):
            event_count = len(data_item.get_events()) if hasattr(data_item, 'get_events') else None
        
        
        payload = data_item.to_dict()
        
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE data_items
                SET block_id = ?, name = ?, type = ?, file_path = ?, metadata = ?
                WHERE id = ?
            """, (
                data_item.block_id,
                data_item.name,
                data_item.type,
                data_item.file_path,
                json.dumps(payload),
                data_item.id
            ))
            
            # Improved logging with event count
            event_info = f", {event_count} events" if event_count is not None else ""
            Log.info(f"Updated data item: {data_item.name} (id: {data_item.id}, type: {data_item.type}{event_info})")
    
    def delete(self, data_item_id: str) -> None:
        """
        Delete data item by ID.
        
        Args:
            data_item_id: Data item identifier
            
        Raises:
            ValueError: If data item not found
        """
        # Verify exists
        existing = self.get(data_item_id)
        if existing is None:
            raise ValueError(f"Data item with id '{data_item_id}' not found")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM data_items
                WHERE id = ?
            """, (data_item_id,))
            
            Log.info(f"Deleted data item: {data_item_id}")
    
    def list_by_block(self, block_id: str) -> List[DataItem]:
        """
        List all data items for a block.
        
        Args:
            block_id: Block identifier
            
        Returns:
            List of DataItem entities for the block
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, block_id, name, type, created_at, file_path, metadata
                FROM data_items
                WHERE block_id = ?
                ORDER BY created_at
            """, (block_id,))
            
            rows = cursor.fetchall()
            items = [self._row_to_data_item(row) for row in rows]
            return items

    def list_data_item_summaries_by_block(self, block_id: str) -> List[DataItemSummary]:
        """
        Return lightweight summaries of data items for a block.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, block_id, name, type, created_at, file_path
                FROM data_items
                WHERE block_id = ?
                ORDER BY created_at
            """, (block_id,))

            rows = cursor.fetchall()
            return [self._row_to_data_item_summary(row) for row in rows]
    
    def find_by_name(self, block_id: str, name: str) -> Optional[DataItem]:
        """
        Find data item by name within a block.
        
        Args:
            block_id: Block identifier
            name: Data item name
            
        Returns:
            DataItem entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, block_id, name, type, created_at, file_path, metadata
                FROM data_items
                WHERE block_id = ? AND name = ?
            """, (block_id, name))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_data_item(row)
    
    def _row_to_data_item(self, row) -> DataItem:
        """
        Convert database row to DataItem entity.
        
        Args:
            row: Database row tuple
            
        Returns:
            DataItem entity (AudioDataItem or EventDataItem)
        """
        data_item_id, block_id, name, item_type, created_at_str, file_path, metadata_json = row
        
        # Parse metadata JSON (contains full to_dict() output)
        try:
            full_dict = json.loads(metadata_json) if metadata_json else {}
        except (json.JSONDecodeError, RecursionError) as exc:
            retry_success = False
            prev_limit = None
            max_depth = None
            if metadata_json:
                # quick depth scan (brackets/braces) ignoring strings
                cur_depth = 0
                max_depth = 0
                in_str = False
                escape = False
                for ch in metadata_json:
                    if in_str:
                        if escape:
                            escape = False
                        elif ch == '\\\\':
                            escape = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                        continue
                    if ch in "{[":
                        cur_depth += 1
                        if cur_depth > max_depth:
                            max_depth = cur_depth
                    elif ch in "}]":
                        cur_depth -= 1
            if isinstance(exc, RecursionError):
                try:
                    import sys as _sys
                    prev_limit = _sys.getrecursionlimit()
                    new_limit = max(prev_limit, 10000)
                    if new_limit != prev_limit:
                        _sys.setrecursionlimit(new_limit)
                    full_dict = json.loads(metadata_json) if metadata_json else {}
                    retry_success = True
                except Exception:
                    retry_success = False
            if not retry_success:
                full_dict = {}
        
        # If metadata is missing core fields, use row data as fallback
        if not full_dict:
            full_dict = {
                "id": data_item_id,
                "block_id": block_id,
                "name": name,
                "type": item_type,
                "created_at": created_at_str,
                "file_path": file_path,
                "metadata": {}
            }
        
        # Create appropriate subclass based on type
        if item_type == "Audio":
            return AudioDataItem.from_dict(full_dict)
        elif item_type == "Event":
            try:
                return EventDataItem.from_dict(full_dict)
            except RecursionError as exc:
                retry_success = False
                prev_limit = None
                try:
                    import sys as _sys
                    prev_limit = _sys.getrecursionlimit()
                    new_limit = max(prev_limit, 10000)
                    if new_limit != prev_limit:
                        _sys.setrecursionlimit(new_limit)
                    return EventDataItem.from_dict(full_dict)
                except Exception:
                    retry_success = False
                # Fallback to minimal, empty EventDataItem to avoid crash
                from datetime import datetime
                return EventDataItem(
                    id=full_dict.get("id", data_item_id) if isinstance(full_dict, dict) else data_item_id,
                    block_id=full_dict.get("block_id", block_id) if isinstance(full_dict, dict) else block_id,
                    name=full_dict.get("name", name) if isinstance(full_dict, dict) else name,
                    type=full_dict.get("type", "Event") if isinstance(full_dict, dict) else "Event",
                    created_at=datetime.fromisoformat(full_dict.get("created_at", created_at_str)) if isinstance(full_dict, dict) else datetime.fromisoformat(created_at_str),
                    file_path=full_dict.get("file_path", file_path) if isinstance(full_dict, dict) else file_path,
                    event_count=0,
                    metadata=full_dict.get("metadata", {}) if isinstance(full_dict, dict) else {},
                    layers=[]
                )
            except Exception as exc:
                raise
        else:
            # Generic DataItem
            from datetime import datetime
            return DataItem(
                id=full_dict.get("id", data_item_id),
                block_id=full_dict.get("block_id", block_id),
                name=full_dict.get("name", name),
                type=full_dict.get("type", item_type),
                created_at=datetime.fromisoformat(full_dict.get("created_at", created_at_str)),
                file_path=full_dict.get("file_path", file_path),
                metadata=full_dict.get("metadata", {})
            )

    def delete_by_block(self, block_id: str) -> int:
        """
        Delete all data items for a specific block.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Number of data items deleted
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM data_items
                WHERE block_id = ?
            """, (block_id,))
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                Log.debug(f"Deleted {deleted_count} data item(s) from block {block_id}")
            return deleted_count
    
    def delete_by_project(self, project_id: str) -> int:
        """
        Delete all data items for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Number of data items deleted
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            # Delete all data items for blocks in this project
            cursor.execute("""
                DELETE FROM data_items
                WHERE block_id IN (
                    SELECT id FROM blocks WHERE project_id = ?
                )
            """, (project_id,))
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                Log.info(f"Deleted {deleted_count} data item(s) from project {project_id}")
            return deleted_count

    def _row_to_data_item_summary(self, row) -> DataItemSummary:
        """
        Convert database row to DataItemSummary.
        """
        created_at = datetime.fromisoformat(row["created_at"])
        return DataItemSummary(
            id=row["id"],
            block_id=row["block_id"],
            name=row["name"],
            type=row["type"],
            created_at=created_at,
            file_path=row["file_path"]
        )

