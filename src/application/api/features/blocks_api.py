"""
Blocks API - Feature-specific facade for block operations.

Provides a focused API for block management and processing.
"""
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from src.application.api.result_types import CommandResult

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class BlocksAPI:
    """
    Blocks feature API.
    
    Provides block management operations:
    - Add/remove blocks
    - Update block settings
    - List blocks
    - Get block status
    
    Usage:
        blocks = BlocksAPI(facade)
        result = blocks.add_block("LoadAudio", name="My Audio")
    """
    
    def __init__(self, facade: "ApplicationFacade"):
        """Initialize with reference to main facade."""
        self._facade = facade
    
    def add_block(
        self, 
        block_type: str, 
        name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> CommandResult:
        """Add a new block to the current project."""
        return self._facade.add_block(block_type, name, settings)
    
    def remove_block(self, block_id: str) -> CommandResult:
        """Remove a block from the current project."""
        return self._facade.remove_block(block_id)
    
    def update_block_settings(self, block_id: str, settings: Dict[str, Any]) -> CommandResult:
        """Update block settings."""
        return self._facade.update_block_settings(block_id, settings)
    
    def get_block(self, block_id: str) -> CommandResult:
        """Get block details."""
        return self._facade.get_block(block_id)
    
    def list_blocks(self) -> CommandResult:
        """List all blocks in current project."""
        return self._facade.list_blocks()
    
    def get_block_types(self) -> CommandResult:
        """Get available block types."""
        return self._facade.get_block_types()
    
    def get_block_status(self, block_id: str) -> CommandResult:
        """Get the status of a specific block."""
        return self._facade.get_block_status(block_id)
