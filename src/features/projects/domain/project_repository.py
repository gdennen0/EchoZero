"""
Project repository interface

Defines the contract for project data access.
Implementations handle persistence details.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.features.projects.domain.project import Project


class ProjectRepository(ABC):
    """Repository interface for Project entities"""
    
    @abstractmethod
    def create(self, project: Project) -> Project:
        """
        Create a new project.
        
        Args:
            project: Project entity to create
            
        Returns:
            Created project with generated ID
            
        Raises:
            ValueError: If project name already exists
        """
        pass
    
    @abstractmethod
    def get(self, project_id: str) -> Optional[Project]:
        """
        Get project by ID.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project entity or None if not found
        """
        pass
    
    @abstractmethod
    def update(self, project: Project) -> None:
        """
        Update existing project.
        
        Args:
            project: Project entity to update
            
        Raises:
            ValueError: If project not found
        """
        pass
    
    @abstractmethod
    def delete(self, project_id: str) -> None:
        """
        Delete project by ID.
        
        Args:
            project_id: Project identifier
            
        Raises:
            ValueError: If project not found
        """
        pass
    
    @abstractmethod
    def find_by_name(self, name: str) -> Optional[Project]:
        """
        Find project by name.
        
        Args:
            name: Project name
            
        Returns:
            Project entity or None if not found
        """
        pass
    
    @abstractmethod
    def list_recent(self, limit: int = 10) -> List[Project]:
        """
        List recently accessed projects.
        
        Args:
            limit: Maximum number of projects to return
            
        Returns:
            List of projects sorted by modified_at (newest first)
        """
        pass

