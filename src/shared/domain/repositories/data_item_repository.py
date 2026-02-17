"""
Data Item Repository Interface

Defines the contract for data item persistence.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.shared.domain.entities import DataItem
from src.shared.domain.entities import DataItemSummary


class DataItemRepository(ABC):
    """Repository interface for DataItem entities"""
    
    @abstractmethod
    def create(self, data_item: DataItem) -> DataItem:
        """
        Create a new data item.
        
        Args:
            data_item: DataItem entity to create
            
        Returns:
            Created data item with generated ID
        """
        pass
    
    @abstractmethod
    def get(self, data_item_id: str) -> Optional[DataItem]:
        """
        Get data item by ID.
        
        Args:
            data_item_id: Data item identifier
            
        Returns:
            DataItem entity or None if not found
        """
        pass
    
    @abstractmethod
    def update(self, data_item: DataItem) -> None:
        """
        Update existing data item.
        
        Args:
            data_item: DataItem entity to update
            
        Raises:
            ValueError: If data item not found
        """
        pass
    
    @abstractmethod
    def delete(self, data_item_id: str) -> None:
        """
        Delete data item by ID.
        
        Args:
            data_item_id: Data item identifier
            
        Raises:
            ValueError: If data item not found
        """
        pass
    
    @abstractmethod
    def list_by_block(self, block_id: str) -> List[DataItem]:
        """
        List all data items for a block.
        
        Args:
            block_id: Block identifier
            
        Returns:
            List of DataItem entities for the block
        """
        pass
    
    @abstractmethod
    def find_by_name(self, block_id: str, name: str) -> Optional[DataItem]:
        """
        Find data item by name within a block.
        
        Args:
            block_id: Block identifier
            name: Data item name
            
        Returns:
            DataItem entity or None if not found
        """
        pass

    @abstractmethod
    def list_data_item_summaries_by_block(self, block_id: str) -> List[DataItemSummary]:
        """
        List lightweight summaries for data items within a block.
        """
        pass

    @abstractmethod
    def load_data_item_detail(self, data_item_id: str) -> Optional[DataItem]:
        """Load the full DataItem entity when needed."""
        pass

    @abstractmethod
    def delete_by_block(self, block_id: str) -> int:
        """
        Delete all data items for a specific block.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Number of data items deleted
        """
        pass
    
    @abstractmethod
    def delete_by_project(self, project_id: str) -> int:
        """
        Delete all data items for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Number of data items deleted
        """
        pass

