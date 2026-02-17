"""
Input Filter Manager

Standardized service for previewing and filtering block input data.

This service provides reusable functionality for any block that needs to:
- Preview incoming data items from source blocks
- Filter input data items by user selection
- Manage filter selections per input port

The pattern is used generically across:
- Execution engine (applies filters to input ports during execution)
- UI components (reusable filter widgets for any block panel)
- ApplicationFacade (generic filter application methods)

Filter Format:
- Filter selections are stored as Dict[str, bool] per port
- {output_name: True/False} where True=enabled, False=disabled
- None means no filter (all items pass)
- Empty dict {} means all disabled (no items pass)
"""
from typing import List, Optional, Dict, Any
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.repositories import DataItemRepository
from src.features.connections.domain import ConnectionRepository
from src.utils.message import Log


class DataFilterManager:
    """
    Service for previewing and filtering block input data items.
    
    Provides standardized functionality for:
    - Previewing incoming data items from source blocks
    - Applying filter selections to data items
    - Managing filter selections per input port
    
    Note: Only handles input filtering. Output filtering has been removed.
    Blocks display expected outputs via processor.get_expected_outputs().
    """
    
    def __init__(
        self,
        data_item_repo: DataItemRepository,
        connection_repo: ConnectionRepository
    ):
        """
        Initialize data filter manager.
        
        Args:
            data_item_repo: Repository for data items
            connection_repo: Repository for connections
        """
        self._data_item_repo = data_item_repo
        self._connection_repo = connection_repo
    
    def preview_input_data(
        self,
        block: Block,
        port_name: str
    ) -> List[DataItem]:
        """
        Preview input data items for a specific port.
        
        Loads data items from all source blocks connected to this input port.
        Handles multiple connections to the same input port (e.g., multiple EventDataItems).
        
        Args:
            block: Target block
            port_name: Input port name
        
        Returns:
            List of DataItem instances available for this input port (from all connections)
        """
        # Find all connections for this input port (multiple connections allowed)
        all_connections = self._connection_repo.list_by_block(block.id)
        port_connections = [
            conn for conn in all_connections
            if conn.target_block_id == block.id and conn.target_input_name == port_name
        ]
        
        if not port_connections:
            return []
        
        # Collect items from all source blocks
        all_items = []
        for conn in port_connections:
            source_items = self._data_item_repo.list_by_block(conn.source_block_id)
            matching_items = [
                item for item in source_items
                if item.metadata.get('output_port') == conn.source_output_name
            ]
            all_items.extend(matching_items)
        
        return all_items
    
    def get_source_block_info(
        self,
        block: Block,
        port_name: str,
        facade=None
    ) -> Optional[Dict[str, Any]]:
        """
        Get source block information for an input port.
        
        Returns information about all source blocks including their expected_outputs.
        Handles multiple connections to the same input port.
        
        Args:
            block: Target block
            port_name: Input port name
            facade: Optional ApplicationFacade (needed to get source blocks)
        
        Returns:
            Dictionary with source block info, or None if no connections
        """
        all_connections = self._connection_repo.list_by_block(block.id)
        port_connections = [
            conn for conn in all_connections
            if conn.target_block_id == block.id and conn.target_input_name == port_name
        ]
        
        if not port_connections or not facade:
            return None
        
        source_blocks_info = []
        for conn in port_connections:
            result = facade.describe_block(conn.source_block_id)
            if not result.success or not result.data:
                continue
            
            source_block = result.data
            expected_outputs = source_block.metadata.get('expected_outputs', {})
            expected_names_for_port = expected_outputs.get(conn.source_output_name)
            
            if expected_names_for_port is None:
                expected_outputs_service = getattr(facade, 'expected_outputs_service', None)
                if expected_outputs_service:
                    processor = facade.execution_engine.get_processor(source_block)
                    if processor:
                        all_expected = expected_outputs_service.calculate_expected_outputs(
                            source_block,
                            processor,
                            facade=facade
                        )
                        expected_names_for_port = all_expected.get(conn.source_output_name, [])
                    else:
                        expected_names_for_port = []
                else:
                    expected_names_for_port = []
            
            source_blocks_info.append({
                'source_block_id': conn.source_block_id,
                'source_block_name': source_block.name,
                'source_output_name': conn.source_output_name,
                'expected_outputs': expected_names_for_port
            })
        
        if not source_blocks_info:
            return None
        
        return {'source_blocks': source_blocks_info}
    
    def apply_filter(
        self,
        block: Block,
        port_name: str,
        items: List[DataItem]
    ) -> List[DataItem]:
        """
        Apply filter to data items based on block's saved filter selections.
        
        This is the SINGLE application point for filters - all filter logic
        should go through this method.
        
        Filter semantics:
        - None: No filter set - all items pass
        - {}: Empty dict - all items disabled (none pass)
        - {name: True/False}: Explicit filter - only enabled items pass
        - Items missing output_name are logged and skipped (fail fast)
        
        Args:
            block: Block to get filter selections for
            port_name: Input port name
            items: List of data items to filter
            
        Returns:
            Filtered list of data items
            
        Raises:
            ValueError: If filter format is invalid (not None or Dict[str, bool])
        """
        if not items:
            return items
        
        # Get filter selection from block metadata
        filter_selections = block.metadata.get("filter_selections", {})
        
        # Validate format
        if not isinstance(filter_selections, dict):
            raise ValueError(
                f"Invalid filter_selections format for {block.name}: "
                f"expected dict, got {type(filter_selections).__name__}"
            )
        
        port_filter = filter_selections.get(port_name)
        
        # No filter set - return all items
        if port_filter is None:
            return items
        
        # Validate port filter format - must be Dict[str, bool]
        if not isinstance(port_filter, dict):
            raise ValueError(
                f"Invalid filter format for {block.name}.{port_name}: "
                f"expected Dict[str, bool], got {type(port_filter).__name__}"
            )
        
        # Empty dict = all disabled
        if len(port_filter) == 0:
            Log.debug(
                f"DataFilterManager: Filter for {block.name}.{port_name} is empty (all disabled)"
            )
            return []
        
        # Apply filter
        filtered = []
        passed_no_output_name = 0
        
        for item in items:
            output_name = item.metadata.get('output_name')
            
            if not output_name:
                # Items without output_name cannot be matched against named filter
                # entries, so they pass through (conservative: don't drop unknown data)
                passed_no_output_name += 1
                filtered.append(item)
                continue
            
            # Check if enabled (default to True if not in filter)
            if port_filter.get(output_name, True):
                filtered.append(item)
        
        if passed_no_output_name > 0:
            Log.debug(
                f"DataFilterManager: {passed_no_output_name} items without output_name "
                f"passed through filter for {block.name}.{port_name}"
            )
        
        if len(filtered) != len(items):
            Log.debug(
                f"DataFilterManager: Applied filter to {block.name}.{port_name}: "
                f"{len(items)} -> {len(filtered)} items"
            )
        
        return filtered
    
    def get_filter(self, block: Block, port_name: str) -> Optional[Dict[str, bool]]:
        """
        Get the filter selection for a port.
        
        Args:
            block: Block to get filter for
            port_name: Input port name
            
        Returns:
            Filter dict {output_name: bool}, or None if no filter set
        """
        filter_selections = block.metadata.get("filter_selections", {})
        if not isinstance(filter_selections, dict):
            return None
        return filter_selections.get(port_name)
    
    def is_filter_all_disabled(self, block: Block, port_name: str) -> bool:
        """
        Check if filter would exclude ALL items for a port.
        
        Use this before loading items to avoid unnecessary database queries.
        
        Returns True if:
        - Filter is empty dict {} (explicitly all disabled)
        - Filter has entries but all are False
        
        Returns False if:
        - Filter is None (no filter, all pass)
        - Filter has at least one True entry
        
        Args:
            block: Block to check filter for
            port_name: Input port name
            
        Returns:
            True if filter would exclude all items, False otherwise
        """
        port_filter = self.get_filter(block, port_name)
        
        # No filter = all items pass
        if port_filter is None:
            return False
        
        # Empty dict = all disabled
        if len(port_filter) == 0:
            return True
        
        # Check if any item is enabled
        return not any(port_filter.values())
    
    def get_enabled_output_names(self, block: Block, port_name: str) -> Optional[set]:
        """
        Get the set of output_names that are enabled by the filter.
        
        Use this to filter item IDs BEFORE loading items from the repository,
        avoiding unnecessary database loads.
        
        Args:
            block: Block to check filter for
            port_name: Input port name
            
        Returns:
            Set of enabled output_names, or None if no filter (all pass)
        """
        port_filter = self.get_filter(block, port_name)
        
        # No filter = all items pass
        if port_filter is None:
            return None
        
        # Return set of enabled output_names
        return {name for name, enabled in port_filter.items() if enabled}
