"""
Data State Service

Calculates data state for blocks by comparing timestamps with source blocks.
Determines if block data is fresh, stale, or missing.
"""
from typing import Optional, Dict, List
from datetime import datetime

from src.shared.domain.data_state import DataState
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository
from src.shared.domain.repositories import DataItemRepository
from src.features.connections.domain import ConnectionRepository
from src.features.blocks.domain import BlockRepository
from src.utils.message import Log


class DataStateService:
    """
    Service for calculating data state of blocks.
    
    Determines if block data is:
    - FRESH: Up-to-date with source blocks
    - STALE: Source blocks have newer data, or errors/filter issues
    - NO_DATA: Block has no data or no local state
    
    Note on Complexity:
    This service appears complex (350+ lines, multiple responsibilities) but the
    complexity is necessary because:
    1. Data state calculation requires checking multiple conditions (errors, filters,
       timestamps, connections, ports)
    2. Edge cases are numerous (no inputs, no outputs, missing data, stale data)
    3. The logic is inherently stateful (depends on block graph structure)
    
    Splitting this into multiple services would create more abstractions without
    clear benefit. The complexity is localized to this single service with a
    clear, single purpose: calculate data state.
    """
    
    def __init__(
        self,
        block_local_state_repo: BlockLocalStateRepository,
        data_item_repo: DataItemRepository,
        connection_repo: ConnectionRepository,
        block_repo: Optional[BlockRepository] = None
    ):
        """
        Initialize data state service.
        
        Args:
            block_local_state_repo: Repository for block local state
            data_item_repo: Repository for data items
            connection_repo: Repository for connections
            block_repo: Repository for blocks (optional, for error/filter checks)
        """
        self._block_local_state_repo = block_local_state_repo
        self._data_item_repo = data_item_repo
        self._connection_repo = connection_repo
        self._block_repo = block_repo
    
    def _get_block_display_name(self, block_id: str, project_id: Optional[str] = None) -> str:
        """
        Get block display name for logging, falling back to block_id if not found.
        
        Args:
            block_id: Block identifier
            project_id: Optional project ID for block lookup
            
        Returns:
            Block name if available, otherwise block_id
        """
        if self._block_repo:
            try:
                # Try with project_id first (more efficient)
                if project_id:
                    block = self._block_repo.get(project_id, block_id)
                else:
                    # Fall back to get_by_id if no project_id available
                    block = self._block_repo.get_by_id(block_id)
                
                if block:
                    return f"{block.name} ({block_id})"
            except Exception:
                pass
        return block_id
    
    def get_block_data_state(self, block_id: str, project_id: Optional[str] = None) -> DataState:
        """
        Get overall data state for a block.
        
        Logic:
        - GREEN (FRESH): Block has output data and inputs are up-to-date (or no inputs)
        - ORANGE (STALE): Block has output data but upstream has newer data, or has errors/filter issues
        - RED (NO_DATA): Block has no output data
        
        For blocks with outputs:
        - If inputs are stale → STALE (upstream has newer data to process)
        - If inputs are fresh or no inputs → FRESH (output is current)
        
        For blocks without outputs:
        - Check input states to determine if ready to execute
        
        Args:
            block_id: Block identifier
            project_id: Optional project ID (required for block lookup if checking errors/filters)
            
        Returns:
            DataState enum value
        """
        block_name = self._get_block_display_name(block_id, project_id)
        Log.debug(f"[STATE-DIAG] get_block_data_state called for block {block_name}")
        
        # Check for errors or filter issues first (these override other states)
        if self._block_repo and project_id:
            try:
                block = self._block_repo.get(project_id, block_id)
                if block:
                    # Check for errors in metadata
                    if self._has_error(block):
                        Log.debug(f"[STATE-DIAG] Block {block_name} has error -> STALE")
                        return DataState.STALE
                    
                    # Check for missing filters
                    if self._has_missing_filter(block):
                        Log.debug(f"[STATE-DIAG] Block {block_name} has missing filter -> STALE")
                        return DataState.STALE
            except Exception as e:
                Log.debug(f"DataStateService: Could not check block for errors/filters: {e}")
        
        # First, check if block has output data
        output_items = self._data_item_repo.list_by_block(block_id)
        has_outputs = bool(output_items)
        
        # Get all connections involving this block
        all_connections = self._connection_repo.list_by_block(block_id)
        
        # Filter for incoming connections (where this block is the target)
        incoming_connections = [
            conn for conn in all_connections
            if conn.target_block_id == block_id
        ]
        
        Log.debug(
            f"[STATE-DIAG] Block {block_name}: has_outputs={has_outputs}, "
            f"incoming_connections={len(incoming_connections)}"
        )
        
        if not incoming_connections:
            # Source block (no inputs) - check if it has output data
            if has_outputs:
                return DataState.FRESH  # Has output data
            return DataState.NO_DATA  # No output data
        port_states: List[DataState] = []
        seen_ports = set()
        for conn in incoming_connections:
            # Deduplicate: only check each input port once (fan-in handled inside get_port_data_state)
            if conn.target_input_name in seen_ports:
                continue
            seen_ports.add(conn.target_input_name)
            port_state = self.get_port_data_state(
                block_id, conn.target_input_name, is_input=True,
                connections=all_connections
            )
            port_states.append(port_state)
            Log.debug(
                f"DataStateService: Block {block_name} input port '{conn.target_input_name}' state: {port_state}"
            )
        
        # Get worst input state
        worst_input_state = DataState.worst(port_states)
        Log.debug(
            f"[STATE-DIAG] Block {block_name}: worst_input_state={worst_input_state}, "
            f"has_outputs={has_outputs}, port_states={[str(s) for s in port_states]}"
        )
        
        # If block has outputs, the state depends on whether inputs are stale
        if has_outputs:
            # If inputs are stale or have dangling references (NO_DATA), block needs re-execution.
            # NO_DATA on a connected input port means the referenced DataItems were deleted
            # (e.g., upstream block re-executed), so outputs are based on stale data.
            if worst_input_state in (DataState.STALE, DataState.NO_DATA):
                Log.debug(
                    f"[STATE-DIAG] Block {block_name} -> STALE "
                    f"(has outputs but inputs are {worst_input_state.value})"
                )
                return DataState.STALE
            else:
                Log.debug(
                    f"[STATE-DIAG] Block {block_name} -> FRESH "
                    f"(has outputs and inputs are current)"
                )
                return DataState.FRESH
        else:
            # No outputs - return input state
            Log.debug(
                f"[STATE-DIAG] Block {block_name} -> {worst_input_state} "
                f"(no outputs, using input state)"
            )
            return worst_input_state
    
    def get_port_data_state(
        self,
        block_id: str,
        port_name: str,
        is_input: bool = True,
        connections: Optional[list] = None
    ) -> DataState:
        """
        Get data state for a specific port.
        
        For input ports:
        - Compares local state data item timestamps with source block data item timestamps
        - NO_DATA if no local state or no data items and source has no data
        - STALE if source has newer data or local references are dangling
        - FRESH if local data is up-to-date
        
        For output ports:
        - Checks if block has output data items
        - NO_DATA if no output items
        - FRESH if output items exist (outputs are always fresh relative to block)
        
        Args:
            block_id: Block identifier
            port_name: Port name
            is_input: True for input port, False for output port
            connections: Optional pre-fetched connections list (avoids redundant queries)
            
        Returns:
            DataState enum value
        """
        if not is_input:
            # For output ports, just check if data exists
            output_items = self._data_item_repo.list_by_block(block_id)
            matching_outputs = [
                item for item in output_items
                if item.metadata.get('output_port') == port_name
            ]
            if matching_outputs:
                return DataState.FRESH
            return DataState.NO_DATA
        
        # For input ports, compare local data with upstream source data
        # Get local state for this block
        local_state = self._block_local_state_repo.get_inputs(block_id)
        if not local_state:
            Log.debug(f"[STATE-DIAG]   port '{port_name}': no local state -> NO_DATA")
            return DataState.NO_DATA
        
        # Get reference for this port
        port_ref = local_state.get(port_name)
        if not port_ref:
            Log.debug(f"[STATE-DIAG]   port '{port_name}': no port_ref in local state -> NO_DATA")
            return DataState.NO_DATA
        
        Log.debug(
            f"[STATE-DIAG]   port '{port_name}': port_ref="
            f"{port_ref if isinstance(port_ref, str) else f'list({len(port_ref)} items)'}"
        )
        
        # Get local data items (references may be dangling if source re-executed)
        local_items: List[DataItem] = []
        if isinstance(port_ref, list):
            for item_id in port_ref:
                item = self._data_item_repo.get(item_id)
                if item:
                    local_items.append(item)
        else:
            item = self._data_item_repo.get(port_ref)
            if item:
                local_items.append(item)
        
        Log.debug(
            f"[STATE-DIAG]   port '{port_name}': local_items_resolved={len(local_items)} "
            f"(dangling={len(port_ref) if isinstance(port_ref, list) else 1} - {len(local_items)} refs missing)"
            if len(local_items) < (len(port_ref) if isinstance(port_ref, list) else 1) else
            f"[STATE-DIAG]   port '{port_name}': local_items_resolved={len(local_items)} (all refs valid)"
        )
        
        # Use passed connections or fetch (avoids N+1 queries when called from get_block_data_state)
        all_connections = connections if connections is not None else self._connection_repo.list_by_block(block_id)
        
        # Find ALL source connections for this port (supports fan-in with multiple sources)
        source_connections = [
            conn for conn in all_connections
            if conn.target_block_id == block_id and conn.target_input_name == port_name
        ]
        
        if not source_connections:
            # No connection for this port - if we have local data, it's fresh
            if local_items:
                return DataState.FRESH
            return DataState.NO_DATA
        
        # Handle dangling references: local state has IDs but DataItems were deleted
        # (happens when upstream block re-executes, deleting old items and creating new ones)
        if not local_items:
            for source_conn in source_connections:
                source_block_name = self._get_block_display_name(source_conn.source_block_id)
                source_items = self._data_item_repo.list_by_block(source_conn.source_block_id)
                matching = [
                    item for item in source_items
                    if item.metadata.get('output_port') == source_conn.source_output_name
                ]
                Log.debug(
                    f"[STATE-DIAG]   port '{port_name}': DANGLING refs - checking source "
                    f"{source_block_name}.{source_conn.source_output_name}: {len(matching)} items available"
                )
                if matching:
                    block_name = self._get_block_display_name(block_id)
                    Log.debug(
                        f"[STATE-DIAG]   port '{port_name}' -> STALE "
                        f"(dangling refs but source has {len(matching)} newer item(s))"
                    )
                    return DataState.STALE
            Log.debug(f"[STATE-DIAG]   port '{port_name}' -> NO_DATA (dangling refs, no source data)")
            return DataState.NO_DATA
        
        # Collect source items from ALL connections for this port (handles fan-in)
        matching_source_items: List[DataItem] = []
        source_block_names = []
        for source_conn in source_connections:
            source_items = self._data_item_repo.list_by_block(source_conn.source_block_id)
            matching = [
                item for item in source_items
                if item.metadata.get('output_port') == source_conn.source_output_name
            ]
            matching_source_items.extend(matching)
            if matching:
                source_block_names.append(self._get_block_display_name(source_conn.source_block_id))
        
        block_name = self._get_block_display_name(block_id)
        Log.debug(
            f"DataStateService: Checking port '{port_name}' for block {block_name} - "
            f"source blocks: {source_block_names}, "
            f"source items: {len(matching_source_items)}, "
            f"local items: {len(local_items)}"
        )
        
        if not matching_source_items:
            # No source has data - but we have local data
            # This means sources haven't executed yet or were cleared
            # Since we have valid local data, it's fresh (nothing new to process)
            Log.debug(f"[STATE-DIAG]   port '{port_name}' -> FRESH (sources have no data items)")
            return DataState.FRESH
        
        # Get timestamps for comparison
        source_timestamps = [item.created_at for item in matching_source_items]
        newest_source_time = max(source_timestamps) if source_timestamps else None
        
        local_timestamps = [item.created_at for item in local_items]
        newest_local_time = max(local_timestamps) if local_timestamps else None
        
        # Get item IDs for ID-based comparison (fallback when timestamps unavailable)
        local_item_ids = {item.id for item in local_items}
        source_item_ids = {item.id for item in matching_source_items}
        
        Log.debug(
            f"[STATE-DIAG]   port '{port_name}': comparing - "
            f"local_IDs={[i[:8] for i in local_item_ids]}, source_IDs={[i[:8] for i in source_item_ids]}, "
            f"source_newest={newest_source_time}, local_newest={newest_local_time}"
        )
        
        if not newest_source_time or not newest_local_time:
            # Can't compare timestamps - use ID-based check
            local_items_are_current = local_item_ids.issubset(source_item_ids)
            if local_items_are_current:
                Log.debug(f"[STATE-DIAG]   port '{port_name}' -> FRESH (ID subset check, no timestamps)")
                return DataState.FRESH
            else:
                Log.debug(f"[STATE-DIAG]   port '{port_name}' -> STALE (ID subset check failed, no timestamps)")
                return DataState.STALE
        
        # Primary comparison: check if local items are the newest source items
        # This correctly handles the case where multiple blocks share the same source:
        # - If local items have the same timestamp as newest source items → FRESH
        # - If source has newer items than local items → STALE
        # - If local items are newer than source (shouldn't happen, but handle gracefully) → FRESH
        
        # Check if local items match the newest source items (by timestamp)
        # Find source items with the newest timestamp
        newest_source_items = [
            item for item in matching_source_items
            if item.created_at == newest_source_time
        ]
        newest_source_item_ids = {item.id for item in newest_source_items}
        
        # Check if local items are the newest source items (by ID)
        # This correctly handles the case where multiple blocks share the same source:
        # - If both blocks reference the same newest items → both FRESH
        # - If one block references older items → that one STALE
        local_items_are_newest = local_item_ids.intersection(newest_source_item_ids)
        
        Log.debug(
            f"[STATE-DIAG]   port '{port_name}': newest_source_IDs={[i[:8] for i in newest_source_item_ids]}, "
            f"intersection={[i[:8] for i in local_items_are_newest]}"
        )
        
        if local_items_are_newest:
            Log.debug(f"[STATE-DIAG]   port '{port_name}' -> FRESH (local items are newest source items)")
            return DataState.FRESH
        elif newest_source_time > newest_local_time:
            Log.debug(
                f"[STATE-DIAG]   port '{port_name}' -> STALE "
                f"(source newer: {newest_source_time} > local: {newest_local_time})"
            )
            return DataState.STALE
        else:
            Log.debug(
                f"[STATE-DIAG]   port '{port_name}' -> FRESH "
                f"(local up-to-date: {newest_local_time} >= source: {newest_source_time})"
            )
            return DataState.FRESH
    
    def _has_error(self, block: Block) -> bool:
        """
        Check if block has an error stored in metadata.
        
        Args:
            block: Block entity
            
        Returns:
            True if block has an error
        """
        # Check for error indicators in metadata
        metadata = block.metadata or {}
        
        # Check for common error fields
        if metadata.get("last_error") or metadata.get("error"):
            return True
        
        # Check for execution failed flag
        if metadata.get("execution_failed"):
            return True
        
        return False
    
    def _has_missing_filter(self, block: Block) -> bool:
        """
        Check if block has filter_selections that are empty (no items selected).
        
        A block has a missing filter if:
        - filter_selections exists for an input port but is an empty list
        - This indicates filter UI was opened but nothing was selected
        
        Note: Only checks input ports. Output ports are never filtered.
        
        Args:
            block: Block entity
            
        Returns:
            True if block has empty filter selections for an input port
        """
        metadata = block.metadata or {}
        filter_selections = metadata.get("filter_selections", {})
        
        if not filter_selections:
            return False
        
        # Only check input ports (output ports are never filtered)
        input_ports = list(block.get_inputs().keys())
        
        # Check each input port's filter selection
        for port_name in input_ports:
            if port_name not in filter_selections:
                continue
            selected_ids = filter_selections[port_name]
            # Empty list means filter was set but nothing selected (problematic)
            if isinstance(selected_ids, list) and len(selected_ids) == 0:
                block_name = f"{block.name} ({block.id})"
                Log.debug(
                    f"DataStateService: Block {block_name} input port '{port_name}' has empty filter selection"
                )
                return True
        
        return False
    
    def is_data_stale(self, block_id: str, port_name: str) -> bool:
        """
        Check if data for a port is stale.
        
        Convenience method that returns True if state is STALE.
        
        Args:
            block_id: Block identifier
            port_name: Port name
            
        Returns:
            True if data is stale, False otherwise
        """
        state = self.get_port_data_state(block_id, port_name, is_input=True)
        return state == DataState.STALE

