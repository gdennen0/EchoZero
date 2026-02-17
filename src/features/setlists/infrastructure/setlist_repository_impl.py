"""
SQLite implementation of SetlistRepository

Handles persistence of Setlist entities in SQLite database.
"""
from typing import Optional, List
from datetime import datetime

from src.features.setlists.domain import Setlist
from src.features.setlists.domain import SetlistRepository
# Removed ExecutionStrategy import - always use full execution
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SQLiteSetlistRepository(SetlistRepository):
    """SQLite implementation of SetlistRepository"""
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def create(self, setlist: Setlist) -> Setlist:
        """
        Create a new setlist (or update existing if project already has one).
        
        One setlist per project - if project already has a setlist, update it.
        
        Args:
            setlist: Setlist entity to create
            
        Returns:
            Created or updated setlist
            
        Raises:
            ValueError: If project_id doesn't match existing setlist
        """
        # Check if project already has a setlist
        existing = self.get_by_project(setlist.project_id)
        if existing:
            # Update existing setlist instead of creating new one
            setlist.id = existing.id
            setlist.created_at = existing.created_at  # Preserve original creation time
            setlist.update_modified()
            self.update(setlist)
            Log.info(f"Updated existing setlist for project {setlist.project_id}")
            return setlist
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Insert setlist
            cursor.execute("""
                INSERT INTO setlists (
                    id, audio_folder_path, project_id, default_actions,
                    created_at, modified_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                setlist.id,
                setlist.audio_folder_path,
                setlist.project_id,
                Database.json_encode(setlist.default_actions),
                setlist.created_at.isoformat(),
                setlist.modified_at.isoformat(),
                Database.json_encode(setlist.metadata)
            ))
            
            Log.info(f"Created setlist for project {setlist.project_id} (id: {setlist.id})")
            return setlist
    
    def get(self, setlist_id: str) -> Optional[Setlist]:
        """
        Get setlist by ID.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            Setlist entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, audio_folder_path, project_id, default_actions,
                       created_at, modified_at, metadata
                FROM setlists
                WHERE id = ?
            """, (setlist_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_setlist(row)
    
    def find_by_folder_path(self, folder_path: str) -> Optional[Setlist]:
        """
        Find setlist by audio folder path.
        
        Args:
            folder_path: Audio folder path
            
        Returns:
            Setlist entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, audio_folder_path, project_id, default_actions,
                       created_at, modified_at, metadata
                FROM setlists
                WHERE audio_folder_path = ?
            """, (folder_path,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_setlist(row)
    
    def get_by_project(self, project_id: str) -> Optional[Setlist]:
        """
        Get the setlist for a project (one setlist per project).
        
        Args:
            project_id: Project identifier
            
        Returns:
            Setlist entity or None if project has no setlist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, audio_folder_path, project_id, default_actions,
                       created_at, modified_at, metadata
                FROM setlists
                WHERE project_id = ?
            """, (project_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_setlist(row)
    
    def list_by_project(self, project_id: str) -> List[Setlist]:
        """
        List all setlists for a project (deprecated - use get_by_project).
        
        Returns at most one setlist since there's one setlist per project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List with at most one setlist entity
        """
        setlist = self.get_by_project(project_id)
        return [setlist] if setlist else []
    
    def update(self, setlist: Setlist) -> None:
        """
        Update setlist.
        
        Args:
            setlist: Setlist entity to update
            
        Raises:
            ValueError: If setlist not found
        """
        # Verify setlist exists
        existing = self.get(setlist.id)
        if existing is None:
            raise ValueError(f"Setlist with id '{setlist.id}' not found")
        
        # No need to check folder path uniqueness - one setlist per project is enforced by DB constraint
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE setlists
                SET audio_folder_path = ?, project_id = ?,
                    default_actions = ?, modified_at = ?, metadata = ?
                WHERE id = ?
            """, (
                setlist.audio_folder_path,
                setlist.project_id,
                Database.json_encode(setlist.default_actions),
                setlist.modified_at.isoformat(),
                Database.json_encode(setlist.metadata),
                setlist.id
            ))
            
            # Only log folder if it's not empty (setlists can exist without folders)
            if setlist.audio_folder_path:
                Log.info(f"Updated setlist for folder: {setlist.audio_folder_path} (id: {setlist.id})")
            else:
                Log.info(f"Updated setlist (id: {setlist.id})")
    
    def delete(self, setlist_id: str) -> None:
        """
        Delete setlist.
        
        Args:
            setlist_id: Setlist identifier to delete
        """
        # Verify setlist exists
        existing = self.get(setlist_id)
        if existing is None:
            raise ValueError(f"Setlist with id '{setlist_id}' not found")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM setlists WHERE id = ?", (setlist_id,))
            
            Log.info(f"Deleted setlist for folder: {existing.audio_folder_path} (id: {setlist_id})")
    
    def _row_to_setlist(self, row) -> Setlist:
        """
        Convert database row to Setlist entity.
        
        Args:
            row: Database row (sqlite3.Row)
            
        Returns:
            Setlist entity
        """
        # sqlite3.Row uses dictionary-style access, not .get()
        audio_folder_path = row["audio_folder_path"] if "audio_folder_path" in row.keys() else ""
        project_id = row["project_id"] if "project_id" in row.keys() else ""
        
        default_actions = {}
        if "default_actions" in row.keys() and row["default_actions"]:
            default_actions = Database.json_decode(row["default_actions"]) or {}
        
        metadata = {}
        if "metadata" in row.keys() and row["metadata"]:
            metadata = Database.json_decode(row["metadata"])
        
        return Setlist(
            id=row["id"],
            audio_folder_path=audio_folder_path,
            project_id=project_id,
            default_actions=default_actions,
            created_at=datetime.fromisoformat(row["created_at"]),
            modified_at=datetime.fromisoformat(row["modified_at"]),
            metadata=metadata
        )

