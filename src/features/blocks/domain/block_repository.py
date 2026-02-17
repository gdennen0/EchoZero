"""
Block repository interface

Defines the contract for block data access.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.features.blocks.domain.block import Block
from src.shared.domain.entities import BlockSummary


class BlockRepository(ABC):
    """Repository interface for Block entities"""
    
    @abstractmethod
    def create(self, block: Block) -> Block:
        """
        Create a new block.
        
        Args:
            block: Block entity to create
            
        Returns:
            Created block with generated ID
            
        Raises:
            ValueError: If block name already exists in project
        """
        pass
    
    @abstractmethod
    def get(self, project_id: str, block_id: str) -> Optional[Block]:
        """
        Get block by ID.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            
        Returns:
            Block entity or None if not found
        """
        pass
    
    @abstractmethod
    def update(self, block: Block) -> None:
        """
        Update existing block.
        
        Args:
            block: Block entity to update
            
        Raises:
            ValueError: If block not found
        """
        pass
    
    @abstractmethod
    def delete(self, project_id: str, block_id: str) -> None:
        """
        Delete block by ID.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            
        Raises:
            ValueError: If block not found
        """
        pass
    
    @abstractmethod
    def list_by_project(self, project_id: str) -> List[Block]:
        """
        List all blocks in a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of blocks in the project
        """
        pass

    @abstractmethod
    def list_block_summaries(self, project_id: str) -> List[BlockSummary]:
        """Return minimal metadata for blocks in a project."""
        pass

    @abstractmethod
    def load_block_detail(self, project_id: str, block_id: str) -> Optional[Block]:
        """Fetch the full block details (ports/metadata)."""
        pass
    
    @abstractmethod
    def find_by_name(self, project_id: str, name: str) -> Optional[Block]:
        """
        Find block by name within a project.
        
        Args:
            project_id: Project identifier
            name: Block name
            
        Returns:
            Block entity or None if not found
        """
        pass
    
    @abstractmethod
    def get_by_id(self, block_id: str) -> Optional[Block]:
        """
        Get block by ID (across all projects).
        
        This is a convenience method for finding blocks when you only have the ID.
        Use get(project_id, block_id) when you know the project for better performance.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Block entity or None if not found
        """
        pass

