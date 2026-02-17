"""
Block State Helper

Simple helper for unified block state access (read + restore).
Provides "one place" to get and restore all block state without complex abstractions.
Returns/accepts simple dicts - easy to understand and use.

Leverages existing unified storage:
- Block settings: block.metadata (from blocks table)
- Block local state: block_local_state repository  
- Block data items: data_item repository

Unified pattern for:
- SnapshotService (save/restore snapshots)
- ProjectService (save/load projects)
- SetlistService (state switching)
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from datetime import datetime, timezone

from src.features.blocks.domain import BlockRepository
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository
from src.shared.domain.repositories import DataItemRepository
from src.shared.domain.entities import DataItem
from src.utils.message import Log


class BlockStateHelper:
    """
    Simple helper for unified block state access.
    
    Provides "one place" to get and restore all block state without complex abstractions.
    Returns/accepts simple dicts - easy to understand and use.
    
    Unified pattern for save/load/state switching:
    - Get state: get_block_state() / get_project_state()
    - Restore state: restore_block_state() / restore_project_state()
    """
    
    def __init__(
        self,
        block_repo: BlockRepository,
        block_local_state_repo: BlockLocalStateRepository,
        data_item_repo: DataItemRepository,
        project_service=None  # For reusing _build_data_item_from_dict
    ):
        """
        Initialize block state helper.
        
        Args:
            block_repo: Repository for accessing blocks
            block_local_state_repo: Repository for block local state
            data_item_repo: Repository for data items
            project_service: Optional ProjectService instance for reusing helper methods
        """
        self._block_repo = block_repo
        self._block_local_state_repo = block_local_state_repo
        self._data_item_repo = data_item_repo
        self._project_service = project_service
        Log.info("BlockStateHelper: Initialized")
    
    def get_block_state(self, block_id: str) -> Dict[str, Any]:
        """
        Get all state for a block - unified access.
        
        Returns dict with:
        - block_id: Block identifier
        - settings: Block metadata (from block.metadata)
        - local_state: Block local state (from block_local_state repo)
        - data_items: List of serialized data items
        
        This provides "one place" to get block state.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Dict containing all block state
            
        Raises:
            ValueError: If block not found
        """
        block = self._block_repo.get_by_id(block_id)
        if not block:
            raise ValueError(f"Block not found: {block_id}")
        
        return {
            "block_id": block_id,
            "settings": block.metadata.copy() if block.metadata else {},
            "local_state": self._block_local_state_repo.get_inputs(block_id) or {},
            "data_items": [item.to_dict() for item in self._data_item_repo.list_by_block(block_id)]
        }
    
    def get_project_state(self, project_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get state for all blocks in project - unified access.
        
        Returns dict: {block_id: {...state...}, ...}
        
        Args:
            project_id: Project identifier
            
        Returns:
            Dict mapping block_id to block state dict
        """
        blocks = self._block_repo.list_by_project(project_id)
        return {block.id: self.get_block_state(block.id) for block in blocks}
    
    def restore_block_state(
        self,
        block_id: str,
        state: Dict[str, Any],
        project_dir: Optional[Path] = None
    ) -> None:
        """
        Restore block state from unified state dict.
        
        Restores:
        - Block local state (via block_local_state_repo.set_inputs())
        - Data items (via data_item_repo.create())
        
        Note: Metadata overrides are handled separately (via commands for undo support).
        
        Args:
            block_id: Block identifier
            state: State dict with 'local_state' and 'data_items' keys
            project_dir: Optional project directory for resolving relative file paths
        """
        # Restore local state
        if state.get("local_state"):
            self._block_local_state_repo.set_inputs(block_id, state["local_state"])
        
        # Restore data items
        for item_dict in state.get("data_items", []):
            # Deserialize data item
            data_item = self._build_data_item_from_dict(item_dict)
            
            # Resolve file paths if project_dir provided
            if project_dir and data_item.file_path:
                if not os.path.isabs(data_item.file_path):
                    resolved_path = project_dir / data_item.file_path
                    data_item.file_path = str(resolved_path)
            
            # Create data item
            self._data_item_repo.create(data_item)
    
    def restore_project_state(
        self,
        project_id: str,
        project_state: Dict[str, Dict[str, Any]],
        project_dir: Optional[Path] = None
    ) -> None:
        """
        Restore state for all blocks in project - unified access.
        
        Args:
            project_id: Project identifier
            project_state: Dict mapping block_id to block state dict
            project_dir: Optional project directory for resolving relative file paths
        """
        for block_id, state in project_state.items():
            self.restore_block_state(block_id, state, project_dir)
    
    def _build_data_item_from_dict(self, data: dict) -> DataItem:
        """
        Build DataItem from dict (reuses ProjectService logic).
        
        Delegates to ProjectService if available, otherwise uses fallback.
        
        Args:
            data: Dictionary containing data item data
            
        Returns:
            DataItem instance
        """
        if self._project_service:
            return self._project_service._build_data_item_from_dict(data)
        else:
            # Fallback implementation (same as ProjectService)
            item_type = (data.get("type") or "").lower()
            if item_type == "audio":
                from src.shared.domain.entities import AudioDataItem as ExportedAudioDataItem
                return ExportedAudioDataItem.from_dict(data)
            if item_type == "event":
                from src.shared.domain.entities import EventDataItem as ExportedEventDataItem
                return ExportedEventDataItem.from_dict(data)
            
            # Default DataItem
            created_at_str = data.get("created_at")
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
            return DataItem(
                id=data.get("id", ""),
                block_id=data.get("block_id", ""),
                name=data.get("name", "DataItem"),
                type=data.get("type", "Data"),
                created_at=created_at,
                file_path=data.get("file_path"),
                metadata=data.get("metadata", {})
            )

