"""
Action Item Repository Interface

Defines the interface for action item persistence.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.features.projects.domain.action_set import ActionItem


class ActionItemRepository(ABC):
    """
    Repository interface for action item persistence.
    
    Handles CRUD operations for ActionItem entities.
    ActionItems are stored independently with foreign keys to ActionSet and Project.
    """
    
    @abstractmethod
    def create(self, action_item: ActionItem) -> ActionItem:
        """
        Create a new action item.
        
        Args:
            action_item: ActionItem entity to create
            
        Returns:
            Created action item (with generated ID if needed)
        """
        pass
    
    @abstractmethod
    def get(self, action_item_id: str) -> Optional[ActionItem]:
        """
        Get action item by ID.
        
        Args:
            action_item_id: Action item identifier
            
        Returns:
            ActionItem entity or None if not found
        """
        pass
    
    @abstractmethod
    def list_by_action_set(self, action_set_id: str) -> List[ActionItem]:
        """
        List all action items for an action set.
        
        Args:
            action_set_id: ActionSet identifier
            
        Returns:
            List of ActionItem entities ordered by order_index
        """
        pass
    
    @abstractmethod
    def list_by_project(self, project_id: str) -> List[ActionItem]:
        """
        List all action items for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of ActionItem entities
        """
        pass
    
    @abstractmethod
    def update(self, action_item: ActionItem) -> None:
        """
        Update existing action item.
        
        Args:
            action_item: ActionItem entity to update
            
        Raises:
            ValueError: If action item not found
        """
        pass
    
    @abstractmethod
    def delete(self, action_item_id: str) -> None:
        """
        Delete action item by ID.
        
        Args:
            action_item_id: Action item identifier
            
        Raises:
            ValueError: If action item not found
        """
        pass
    
    @abstractmethod
    def delete_by_action_set(self, action_set_id: str) -> int:
        """
        Delete all action items for an action set.
        
        Args:
            action_set_id: ActionSet identifier
            
        Returns:
            Number of items deleted
        """
        pass
    
    @abstractmethod
    def delete_by_project(self, project_id: str) -> int:
        """
        Delete all action items for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Number of items deleted
        """
        pass
    
    @abstractmethod
    def reorder(self, action_set_id: str, item_ids: List[str]) -> None:
        """
        Reorder action items within an action set.
        
        Args:
            action_set_id: ActionSet identifier
            item_ids: List of action item IDs in desired order
        """
        pass


