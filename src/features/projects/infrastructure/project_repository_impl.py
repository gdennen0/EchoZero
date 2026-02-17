"""
SQLite implementation of ProjectRepository

Handles persistence of Project entities in SQLite database.
"""
from typing import Optional, List
from datetime import datetime

from src.features.projects.domain.project import Project
from src.features.projects.domain.project_repository import ProjectRepository
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SQLiteProjectRepository(ProjectRepository):
    """SQLite implementation of ProjectRepository"""
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def create(self, project: Project) -> Project:
        """
        Create a new project.
        
        Args:
            project: Project entity to create
            
        Returns:
            Created project (with generated ID if needed)
            
        Raises:
            ValueError: If project name already exists
        """
        # Check if name already exists
        existing = self.find_by_name(project.name)
        if existing and existing.id != project.id:
            raise ValueError(f"Project with name '{project.name}' already exists")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Insert project
            cursor.execute("""
                INSERT INTO projects (id, name, version, save_directory, created_at, modified_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                project.id,
                project.name,
                project.version,
                project.save_directory,
                project.created_at.isoformat(),
                project.modified_at.isoformat(),
                Database.json_encode(project.metadata)
            ))
            
            Log.info(f"Created project: {project.name} (id: {project.id})")
            return project
    
    def get(self, project_id: str) -> Optional[Project]:
        """
        Get project by ID.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, version, save_directory, created_at, modified_at, metadata
                FROM projects
                WHERE id = ?
            """, (project_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_project(row)
    
    def update(self, project: Project) -> None:
        """
        Update existing project.
        
        Args:
            project: Project entity to update
            
        Raises:
            ValueError: If project not found
        """
        # Verify project exists
        existing = self.get(project.id)
        if existing is None:
            raise ValueError(f"Project with id '{project.id}' not found")
        
        # Check name uniqueness (if name changed)
        if existing.name != project.name:
            name_conflict = self.find_by_name(project.name)
            if name_conflict and name_conflict.id != project.id:
                raise ValueError(f"Project with name '{project.name}' already exists")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE projects
                SET name = ?, version = ?, save_directory = ?, modified_at = ?, metadata = ?
                WHERE id = ?
            """, (
                project.name,
                project.version,
                project.save_directory,
                project.modified_at.isoformat(),
                Database.json_encode(project.metadata),
                project.id
            ))
            
            Log.info(f"Updated project: {project.name} (id: {project.id})")
    
    def delete(self, project_id: str) -> None:
        """
        Delete project by ID.
        
        Args:
            project_id: Project identifier
            
        Raises:
            ValueError: If project not found
        """
        # Verify project exists
        existing = self.get(project_id)
        if existing is None:
            raise ValueError(f"Project with id '{project_id}' not found")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            
            Log.info(f"Deleted project: {existing.name} (id: {project_id})")
    
    def find_by_name(self, name: str) -> Optional[Project]:
        """
        Find project by name.
        
        Args:
            name: Project name
            
        Returns:
            Project entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, version, save_directory, created_at, modified_at, metadata
                FROM projects
                WHERE name = ?
            """, (name,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_project(row)
    
    def list_recent(self, limit: int = 10) -> List[Project]:
        raise NotImplementedError("Recent projects are no longer stored in the repository")
    
    def _row_to_project(self, row) -> Project:
        """
        Convert database row to Project entity.
        
        Args:
            row: Database row (sqlite3.Row)
            
        Returns:
            Project entity
        """
        return Project(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            save_directory=row["save_directory"],
            created_at=datetime.fromisoformat(row["created_at"]),
            modified_at=datetime.fromisoformat(row["modified_at"]),
            metadata=Database.json_decode(row["metadata"])
        )

