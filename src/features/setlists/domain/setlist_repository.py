"""
Setlist Repository Interface

Defines the interface for setlist persistence.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.features.setlists.domain import Setlist


class SetlistRepository(ABC):
    """
    Repository interface for setlist persistence.
    
    Handles CRUD operations for Setlist entities.
    """
    
    @abstractmethod
    def create(self, setlist: Setlist) -> Setlist:
        """
        Create a new setlist.
        
        Args:
            setlist: Setlist entity to create
            
        Returns:
            Created setlist (with generated ID if needed)
            
        Raises:
            ValueError: If setlist name already exists
        """
        pass
    
    @abstractmethod
    def get(self, setlist_id: str) -> Optional[Setlist]:
        """
        Get setlist by ID.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            Setlist entity or None if not found
        """
        pass
    
    @abstractmethod
    def find_by_folder_path(self, folder_path: str) -> Optional[Setlist]:
        """
        Find setlist by audio folder path.
        
        Args:
            folder_path: Audio folder path
            
        Returns:
            Setlist entity or None if not found
        """
        pass
    
    @abstractmethod
    def get_by_project(self, project_id: str) -> Optional[Setlist]:
        """
        Get the setlist for a project (one setlist per project).
        
        Args:
            project_id: Project identifier
            
        Returns:
            Setlist entity or None if project has no setlist
        """
        pass
    
    @abstractmethod
    def list_by_project(self, project_id: str) -> List[Setlist]:
        """
        List all setlists for a project (deprecated - use get_by_project).
        
        Kept for backward compatibility but should return at most one setlist.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of setlist entities for the project (max 1)
        """
        pass
    
    @abstractmethod
    def update(self, setlist: Setlist) -> None:
        """
        Update setlist.
        
        Args:
            setlist: Setlist entity to update
            
        Raises:
            ValueError: If setlist not found
        """
        pass
    
    @abstractmethod
    def delete(self, setlist_id: str) -> None:
        """
        Delete setlist.
        
        Args:
            setlist_id: Setlist identifier to delete
        """
        pass

