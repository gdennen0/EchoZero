"""
Action Set Repository Interface

Defines the interface for action set persistence.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.features.projects.domain.action_set import ActionSet


class ActionSetRepository(ABC):
    """
    Repository interface for action set persistence.
    
    Handles CRUD operations for ActionSet entities.
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get(self, action_set_id: str) -> Optional[ActionSet]:
        """
        Get action set by ID.
        
        Args:
            action_set_id: Action set identifier
            
        Returns:
            ActionSet entity or None if not found
        """
        pass
    
    @abstractmethod
    def find_by_name(self, name: str, project_id: Optional[str] = None) -> Optional[ActionSet]:
        """
        Find action set by name.
        
        Args:
            name: Action set name
            project_id: Optional project ID to scope search
            
        Returns:
            ActionSet entity or None if not found
        """
        pass
    
    @abstractmethod
    def list_all(self, project_id: Optional[str] = None) -> List[ActionSet]:
        """
        List all action sets.
        
        Args:
            project_id: Optional project ID to filter by
            
        Returns:
            List of ActionSet entities
        """
        pass
    
    @abstractmethod
    def list_by_project(self, project_id: str) -> List[ActionSet]:
        """
        List all action sets for a project (follows blocks pattern).
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of ActionSet entities for the project
        """
        pass
    
    @abstractmethod
    def update(self, action_set: ActionSet) -> None:
        """
        Update existing action set.
        
        Args:
            action_set: ActionSet entity to update
            
        Raises:
            ValueError: If action set not found
        """
        pass
    
    @abstractmethod
    def delete(self, action_set_id: str) -> None:
        """
        Delete action set by ID.
        
        Args:
            action_set_id: Action set identifier
            
        Raises:
            ValueError: If action set not found
        """
        pass
    
    @abstractmethod
    def delete_by_project(self, project_id: str) -> int:
        """
        Delete all action sets for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Number of action sets deleted
        """
        pass

