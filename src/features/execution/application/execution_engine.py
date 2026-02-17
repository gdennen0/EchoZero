"""
Block Execution Engine

Executes blocks in a project graph in the correct order.
Handles data flow between blocks via connections.
"""
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from src.features.blocks.domain import Block
from src.features.connections.domain import Connection
from src.shared.domain.entities import DataItem
from src.features.blocks.domain import BlockRepository
from src.features.connections.domain import ConnectionRepository
from src.shared.domain.repositories import DataItemRepository
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository
from src.application.processing.block_processor import BlockProcessor, ProcessingError, FilterError
from src.features.execution.application.topological_sort import (
    topological_sort_blocks,
    validate_block_graph,
    CyclicDependencyError
)
from src.application.block_registry import get_block_registry
from src.application.events import (
    EventBus,
    ExecutionStarted,
    ExecutionProgress,
    BlockExecuted,
    BlockExecutionFailed,
    ExecutionCompleted,
    BlockChanged,
    BlockUpdated
)
from src.utils.message import Log


@dataclass
class ExecutionResult:
    """Result of block execution"""
    success: bool
    executed_blocks: List[str]  # Block IDs that executed successfully
    failed_blocks: List[str]  # Block IDs that failed
    output_data: Dict[str, Dict[str, DataItem]]  # block_id -> {port_name: DataItem}
    errors: Dict[str, str]  # block_id -> error message
    error_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # block_id -> {block_name, block_type, error, traceback}


class BlockExecutionEngine:
    """
    Execution engine for processing block graphs.
    
    Handles:
    - Topological sorting of blocks for execution order
    - Data flow between blocks via connections
    - Error handling and recovery
    - Progress reporting via events
    """
    
    def __init__(
        self,
        block_repo: BlockRepository,
        connection_repo: ConnectionRepository,
        data_item_repo: DataItemRepository,
        event_bus: EventBus,
        block_local_state_repo: Optional[BlockLocalStateRepository] = None,
        data_filter_manager=None,
        pull_callback: Optional[Callable[[str], Any]] = None
    ):
        """
        Initialize execution engine.
        
        Args:
            block_repo: Repository for accessing blocks
            connection_repo: Repository for accessing connections
            data_item_repo: Repository for storing data items
            event_bus: Event bus for publishing execution events
            block_local_state_repo: Repository for block local state
            data_filter_manager: Optional data filter manager for applying filters
            pull_callback: Optional callback to pull data from upstream (block_id -> result)
        """
        self._block_repo = block_repo
        self._connection_repo = connection_repo
        self._data_item_repo = data_item_repo
        self._event_bus = event_bus
        self._block_local_state_repo = block_local_state_repo
        self._data_filter_manager = data_filter_manager
        self._pull_callback = pull_callback
        self._processors: Dict[str, BlockProcessor] = {}
        Log.info("BlockExecutionEngine: Initialized")
    
    def register_processor(self, processor: BlockProcessor) -> None:
        """
        Register a block processor.
        
        Args:
            processor: BlockProcessor implementation
        """
        block_type = processor.get_block_type()
        if block_type in self._processors:
            Log.warning(f"BlockExecutionEngine: Overwriting processor for block type '{block_type}'")
        
        self._processors[block_type] = processor
        Log.info(f"BlockExecutionEngine: Registered processor for block type '{block_type}'")
    
    def get_processor(self, block: Block) -> Optional[BlockProcessor]:
        """
        Get processor for a block.
        
        Args:
            block: Block entity
            
        Returns:
            BlockProcessor instance or None if no processor registered for this block type
        """
        return self._processors.get(block.type)
    
    def validate_project(self, project_id: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a project's block graph is executable.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get all blocks and connections
        blocks = self._block_repo.list_by_project(project_id)
        connections = self._connection_repo.list_by_project(project_id)
        
        # Validate graph structure
        is_valid, error = validate_block_graph(blocks, connections)
        if not is_valid:
            return False, error
        
        # Check all blocks have processors
        for block in blocks:
            if not self.get_processor(block):
                return False, f"No processor registered for block type '{block.type}' (block: {block.name})"
        
        return True, None
    
    def _cleanup_blocks_after_execution(self, blocks: List[Block]) -> None:
        """
        Clean up all blocks after execution.
        
        Calls cleanup() on each block processor to free memory, close file handles,
        and release library resources (PyTorch, TensorFlow, etc.).
        
        This is critical for batch processing to prevent memory accumulation.
        
        Args:
            blocks: List of blocks that were executed
        """
        import gc
        
        for block in blocks:
            processor = self.get_processor(block)
            if processor and hasattr(processor, 'cleanup'):
                try:
                    processor.cleanup(block)
                    Log.debug(f"BlockExecutionEngine: Cleaned up block {block.name}")
                except Exception as e:
                    Log.warning(f"BlockExecutionEngine: Cleanup failed for {block.name}: {e}")
        
        # Trigger garbage collection for libraries (PyTorch, TensorFlow, etc.)
        gc.collect()
        Log.debug("BlockExecutionEngine: Triggered garbage collection after block cleanup")

    def _gather_inputs_from_local_state(self, block: Block) -> Dict[str, Any]:
        """
        Build execution inputs strictly from persisted block local state.

        Local state stores references only: {input_port: data_item_id | [data_item_id]}
        """
        # Source blocks (no inputs defined) can execute without local state
        # Use the unified port system: block.get_inputs() instead of block.inputs
        block_inputs = block.get_inputs()
        
        if not block_inputs:
            return {}

        if not self._block_local_state_repo:
            raise ProcessingError(
                "Block local state repository not configured",
                block_id=block.id,
                block_name=block.name,
            )

        mapping = self._block_local_state_repo.get_inputs(block.id) or {}

        # If a block declares inputs, require that local state exists (model: pull first)
        if not mapping:
            raise ProcessingError(
                "No local inputs pulled for this block. Use 'Pull Data' first.",
                block_id=block.id,
                block_name=block.name,
            )

        inputs: Dict[str, Any] = {}
        for port_name in block.get_inputs().keys():
            if port_name not in mapping:
                continue
            ref = mapping.get(port_name)
            if isinstance(ref, list):
                items = []
                for data_item_id in ref:
                    item = self._data_item_repo.get(data_item_id)
                    if item is not None:
                        items.append(item)
                # Apply filter if available
                items = self._apply_filter(block, port_name, items)
                inputs[port_name] = items
            else:
                item = self._data_item_repo.get(ref)
                if item is not None:
                    # Apply filter if available
                    # Filter error will be raised by _apply_filter if all items filtered out
                    filtered = self._apply_filter(block, port_name, [item])
                    if filtered:
                        inputs[port_name] = filtered[0]
                    else:
                        # Filter error will be raised by _apply_filter
                        # Don't set input here - let the error propagate
                        pass

        # Validate input types
        from src.application.processing.type_validation import validate_input_types
        
        block_metadata = get_block_registry().get(block.type)
        if block_metadata:
            # Normalize inputs to single items for validation (handle lists)
            normalized_inputs = {}
            for port_name, value in inputs.items():
                if isinstance(value, list):
                    if value:
                        normalized_inputs[port_name] = value[0]  # Validate first item
                else:
                    normalized_inputs[port_name] = value
            
            validation_errors = validate_input_types(
                block_type=block.type,
                inputs=normalized_inputs,
                expected_inputs=block_metadata.inputs
            )
            
            if validation_errors:
                raise ProcessingError(
                    f"Input type validation failed: {', '.join(validation_errors)}",
                    block_id=block.id,
                    block_name=block.name
                )

        return inputs
    
    def _apply_filter(self, block: Block, port_name: str, items: List[DataItem]) -> List[DataItem]:
        """
        Apply filter to input data items using DataFilterManager.
        
        Delegates to DataFilterManager.apply_filter() which is the single
        application point for all filter logic.
        
        Args:
            block: Block to get filter selections for
            port_name: Input port name
            items: List of data items to filter
            
        Returns:
            Filtered list of data items
        """
        if not self._data_filter_manager or not items:
            return items
        
        try:
            return self._data_filter_manager.apply_filter(block, port_name, items)
        except ValueError as e:
            # Invalid filter format - log error and return unfiltered
            Log.error(f"BlockExecutionEngine: Invalid filter: {e}")
            return items
        except Exception as e:
            Log.warning(f"BlockExecutionEngine: Failed to apply filter: {e}")
            return items
    
    def _gather_inputs(
        self,
        block: Block,
        connection_map: Dict[tuple[str, str], tuple[str, str]],
        output_data: Dict[str, Dict[str, DataItem]]
    ) -> Dict[str, DataItem]:
        """
        Gather input data items for a block based on connections.
        Enhanced to check database for cached DataItems if not in current execution.
        
        Args:
            block: Block to gather inputs for
            connection_map: Map of (target_block_id, target_input_name) -> (source_block_id, source_output_name)
            output_data: Map of block_id -> {port_name: DataItem} for executed blocks
            
        Returns:
            Dictionary mapping input port names to DataItem instances
        """
        inputs: Dict[str, DataItem] = {}
        
        # For each input port, check if there's a connection
        for input_port_name in block.get_inputs().keys():
            key = (block.id, input_port_name)
            
            if key in connection_map:
                # Get source block and output port
                source_block_id, source_output_name = connection_map[key]
                
                # Try to get data from current execution first
                if source_block_id in output_data:
                    source_outputs = output_data[source_block_id]
                    if source_output_name in source_outputs:
                        inputs[input_port_name] = source_outputs[source_output_name]
                    else:
                        Log.warning(
                            f"BlockExecutionEngine: Source block {source_block_id} "
                            f"has no output '{source_output_name}'"
                        )
                else:
                    # Fall back to cached DataItems from database
                    cached_data = self._get_cached_output(source_block_id, source_output_name)
                    if cached_data:
                        Log.debug(
                            f"BlockExecutionEngine: Using cached output from block {source_block_id} "
                            f"port '{source_output_name}' for block {block.name}"
                        )
                        inputs[input_port_name] = cached_data
                    else:
                        Log.warning(
                            f"BlockExecutionEngine: Source block {source_block_id} "
                            f"has no output '{source_output_name}' (not executed and no cache)"
                        )
            # If no connection, input is optional (block will receive None/empty)
        
        return inputs
    
    def _get_cached_output(self, block_id: str, port_name: str) -> Optional[DataItem]:
        """
        Get cached DataItem from database for a specific block output port.
        
        Args:
            block_id: Source block identifier
            port_name: Output port name
            
        Returns:
            DataItem or list of DataItems if multiple, or None if not found
        """
        cached = self._data_item_repo.list_by_block(block_id)
        port_items = [item for item in cached 
                      if item.metadata.get('output_port') == port_name]
        
        if not port_items:
            return None
        elif len(port_items) == 1:
            return port_items[0]
        else:
            # Multiple items on same port
            return port_items
    
    def execute_block(
        self,
        block: Block,
        inputs: Optional[Dict[str, DataItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_pull: bool = True,
        auto_save: bool = True
    ) -> Dict[str, DataItem]:
        """
        Execute a single block with full preparation and cleanup.
        
        Execution flow:
        1. Clear local data (inputs and outputs from previous execution)
        2. Pull fresh input data from upstream connections
        3. Process data using the block processor
        4. Update local data with new process results (outputs)
        
        Args:
            block: Block entity to execute
            inputs: Optional input data items (if None, will gather from local state)
            metadata: Optional metadata for processing
            auto_pull: Whether to pull upstream data first (default: True)
            auto_save: Whether to save outputs to database (default: True)
            
        Returns:
            Dictionary mapping output port names to DataItem instances
            
        Raises:
            ProcessingError: If processing fails
            ValueError: If no processor registered for block type
        """
        # =====================================================================
        # STEP 1: CLEAR LOCAL DATA (always happens by default)
        # =====================================================================
        
        # 1a: Clear local state references
        if self._block_local_state_repo:
            try:
                self._block_local_state_repo.clear_inputs(block.id)
                Log.debug(f"ExecutionEngine: [STEP 1a] Cleared local state for '{block.name}'")
            except Exception as e:
                Log.warning(f"ExecutionEngine: [STEP 1a] Failed to clear local state for '{block.name}': {e}")
        
        # 1b: Get processor and call step_clear_local_data (clears owned data items)
        processor = self.get_processor(block)
        if processor:
            try:
                processor.step_clear_local_data(block, metadata=metadata or {})
                Log.debug(f"ExecutionEngine: [STEP 1b] Cleared owned data for '{block.name}'")
            except Exception as e:
                Log.warning(f"ExecutionEngine: [STEP 1b] step_clear_local_data failed for '{block.name}': {e}")
        
        # Step 2: Pull fresh input data from upstream connections
        # Execution must pull data from inputs to ensure fresh data
        if auto_pull and inputs is None:
            # Check if block has inputs that need pulling
            input_ports = block.get_inputs()
            has_inputs = bool(input_ports)
            
            if has_inputs:
                if not self._pull_callback:
                    raise ProcessingError(
                        f"Cannot execute block '{block.name}': pull_callback not configured. "
                        f"Block has inputs but no way to pull upstream data.",
                        block_id=block.id,
                        block_name=block.name
                    )
                
                # Pull data from upstream - this must succeed for blocks with inputs
                try:
                    self._pull_callback(block.id)
                    Log.debug(f"ExecutionEngine: Pulled fresh input data for block '{block.name}'")
                except Exception as e:
                    # Gather diagnostic information about the block and its connections
                    diagnostic_info = []
                    diagnostic_info.append(f"Block: '{block.name}' (ID: {block.id}, Type: {block.type})")
                    diagnostic_info.append(f"Declared inputs: {list(input_ports.keys()) if input_ports else 'none'}")
                    
                    # Check connections
                    try:
                        connections = self._connection_repo.list_by_block(block.id)
                        incoming_connections = [c for c in connections if c.target_block_id == block.id]
                        diagnostic_info.append(f"Incoming connections: {len(incoming_connections)}")
                        for conn in incoming_connections:
                            # Check if source block exists and has data
                            source_block = self._block_repo.get_by_id(conn.source_block_id)
                            source_has_data = False
                            if source_block:
                                source_data_items = self._data_item_repo.list_by_block(conn.source_block_id)
                                source_has_data = len(source_data_items) > 0
                                diagnostic_info.append(
                                    f"  Connection: {conn.source_block_id}.{conn.source_output_name} -> {conn.target_block_id}.{conn.target_input_name}"
                                )
                                diagnostic_info.append(
                                    f"    Source block: '{source_block.name}' (Type: {source_block.type}), Has data items: {source_has_data} ({len(source_data_items)} items)"
                                )
                            else:
                                diagnostic_info.append(
                                    f"  Connection: {conn.source_block_id}.{conn.source_output_name} -> {conn.target_block_id}.{conn.target_input_name}"
                                )
                                diagnostic_info.append(f"    Source block: NOT FOUND (ID: {conn.source_block_id})")
                    except Exception as conn_e:
                        diagnostic_info.append(f"Error gathering connection info: {conn_e}")
                    
                    # Check local state
                    try:
                        if self._block_local_state_repo:
                            local_state = self._block_local_state_repo.get_inputs(block.id)
                            diagnostic_info.append(f"Current local state: {local_state if local_state else 'empty'}")
                    except Exception as state_e:
                        diagnostic_info.append(f"Error checking local state: {state_e}")
                    
                    diagnostic_message = "\n".join(diagnostic_info)
                    Log.error(f"ExecutionEngine: Pull failed for block '{block.name}':\n{diagnostic_message}\n\nOriginal error: {e}")
                    
                    # Pull failed - execution cannot proceed with stale data
                    full_error_msg = (
                        f"Failed to pull data from upstream for block '{block.name}': {e}. "
                        f"Execution requires fresh data from inputs.\n\n"
                        f"Diagnostic Information:\n{diagnostic_message}"
                    )
                    raise ProcessingError(
                        full_error_msg,
                        block_id=block.id,
                        block_name=block.name
                    ) from e
        
        # =====================================================================
        # STEP 3: PRE-PROCESS
        # =====================================================================
        
        # Verify processor (already fetched in Step 1b)
        if not processor:
            processor = self.get_processor(block)
        if not processor:
            raise ValueError(f"No processor registered for block type '{block.type}'")
        
        # Call step_pre_process hook
        try:
            processor.step_pre_process(block, metadata=metadata or {})
            Log.debug(f"ExecutionEngine: [STEP 3] Pre-process complete for '{block.name}'")
        except Exception as e:
            Log.warning(f"ExecutionEngine: [STEP 3] step_pre_process failed for '{block.name}': {e}")
        
        # =====================================================================
        # STEP 4: PROCESS (gather inputs and execute)
        # =====================================================================
        
        # Gather inputs from local state if not provided
        if inputs is None:
            inputs = self._gather_inputs_from_local_state(block)
        
        # Execute main processing logic
        outputs = processor.process(block, inputs, metadata=metadata or {})
        Log.debug(f"ExecutionEngine: [STEP 4] Process complete for '{block.name}'")
        
        # Validate output names
        try:
            warnings = processor.validate_output_names(block, outputs)
            if warnings:
                Log.warning(f"ExecutionEngine: Output validation warnings for {block.name}: {warnings}")
        except Exception as e:
            Log.debug(f"ExecutionEngine: Output validation failed for {block.name}: {e}")
        
        # =====================================================================
        # STEP 5: POST-PROCESS
        # =====================================================================
        
        try:
            outputs = processor.step_post_process(block, outputs, metadata=metadata or {})
            Log.debug(f"ExecutionEngine: [STEP 5] Post-process complete for '{block.name}'")
        except Exception as e:
            Log.warning(f"ExecutionEngine: [STEP 5] step_post_process failed for '{block.name}': {e}")
        
        # =====================================================================
        # STEP 6: SAVE & NOTIFY
        # =====================================================================
        
        if auto_save:
            self._save_block_outputs(block, outputs)
            Log.debug(f"ExecutionEngine: [STEP 6] Saved outputs for '{block.name}'")
            
            # Release in-memory audio data after outputs are persisted to free RAM.
            # Audio files remain on disk and can be lazy-loaded if needed.
            self._release_audio_data(outputs)
            
            # Publish BlockUpdated event to notify UI panels (especially Editor panels)
            # that the block's data has changed and UI should refresh
            if self._event_bus:
                # Build event data - include execution flags if processor set them
                event_data = {
                    "id": block.id,
                    "name": block.name,
                    "type": block.type
                }
                
                # Include execution_triggered flag from block metadata if set by processor
                # This tells UI panels (like EditorPanel) to do a full refresh
                if block.metadata.get("_execution_triggered"):
                    event_data["execution_triggered"] = True
                    # Include layer names if provided
                    if block.metadata.get("_new_layer_names"):
                        event_data["layer_names"] = block.metadata.get("_new_layer_names")
                    # Clear the flags after use
                    block.metadata.pop("_execution_triggered", None)
                    block.metadata.pop("_new_layer_names", None)
                
                self._event_bus.publish(BlockUpdated(
                    project_id=metadata.get("project_id") if metadata else None,
                    data=event_data
                ))
                Log.debug(f"ExecutionEngine: Published BlockUpdated event for '{block.name}'")
        
        return outputs
    
    def _emit_downstream_status_changed(self, project_id: str, source_block_id: str) -> None:
        """
        Emit BlockChanged events for all downstream blocks (including transitive).
        
        When a block's data is updated, all blocks that depend on it (directly or transitively)
        may have stale data and need their status indicators refreshed.
        
        Args:
            project_id: Project identifier
            source_block_id: Source block that was updated
        """
        try:
            # Find all transitive downstream blocks using BFS
            downstream_block_ids = self._find_transitive_downstream_blocks(source_block_id)
            
            # Emit status changed event for each downstream block
            for downstream_block_id in downstream_block_ids:
                self._event_bus.publish(BlockChanged(
                    project_id=project_id,
                    data={
                        "block_id": downstream_block_id,
                        "change_type": "data"
                    }
                ))
                Log.debug(f"BlockExecutionEngine: Emitted BlockChanged for downstream block '{downstream_block_id}' (source: '{source_block_id}')")
            
            # Also emit for the source block itself (its status may have changed too)
            self._event_bus.publish(BlockChanged(
                project_id=project_id,
                data={
                    "block_id": source_block_id,
                    "change_type": "data"
                }
            ))
            
        except Exception as e:
            Log.warning(f"BlockExecutionEngine: Failed to emit downstream status changes for '{source_block_id}': {e}")
    
    def _find_transitive_downstream_blocks(self, source_block_id: str) -> set[str]:
        """
        Find all blocks that depend on the source block, transitively.
        
        Uses BFS to traverse the dependency graph and find all blocks that
        directly or indirectly depend on the source block.
        
        Args:
            source_block_id: Source block identifier
            
        Returns:
            Set of downstream block IDs (including transitive dependencies, excluding source)
        """
        from collections import deque
        
        downstream_blocks = set()
        queue = deque([source_block_id])
        visited = set()
        
        while queue:
            current_block_id = queue.popleft()
            if current_block_id in visited:
                continue
            visited.add(current_block_id)
            
            # Get all connections where current_block is the source
            all_connections = self._connection_repo.list_by_block(current_block_id)
            for conn in all_connections:
                if conn.source_block_id == current_block_id:
                    target_id = conn.target_block_id
                    if target_id not in visited:
                        # Only add to downstream set (not the source itself)
                        downstream_blocks.add(target_id)
                        queue.append(target_id)
        
        return downstream_blocks

    def _capture_filter_intent_for_reconciliation(self, blocks: List[Block]) -> None:
        """
        Capture filter intent before data clearing for post-execution reconciliation.

        For blocks with filter_selections, capture what sources the selected items came from.
        This allows us to re-apply filters to equivalent data after execution.

        Args:
            blocks: List of blocks to process
        """
        for block in blocks:
            filter_selections = block.metadata.get("filter_selections", {})
            if not filter_selections:
                continue

            # Build intent map: port_name -> list of (source_block_id, source_output_name, item_name_pattern)
            filter_intent = {}

            for port_name, selected_output_names in filter_selections.items():
                if not selected_output_names or not isinstance(selected_output_names, list):
                    continue

                # Find connections for this port
                all_connections = self._connection_repo.list_by_block(block.id)
                port_connections = [
                    conn for conn in all_connections
                    if conn.target_block_id == block.id and conn.target_input_name == port_name
                ]

                if not port_connections:
                    continue

                # For each selected output name, find what source it came from
                intent_sources = []
                for output_name in selected_output_names:
                    # Find which connection this output name came from by matching output_name
                    for conn in port_connections:
                        source_items = self._data_item_repo.list_by_block(conn.source_block_id)
                        matching_items = [
                            item for item in source_items
                            if item.metadata.get('output_name') == output_name 
                            and item.metadata.get('output_port') == conn.source_output_name
                        ]

                        if matching_items:
                            # Use first matching item for metadata
                            data_item = matching_items[0]
                            # Capture the source information
                            intent_sources.append({
                                'source_block_id': conn.source_block_id,
                                'source_output_name': conn.source_output_name,
                                'output_name': output_name,  # Store semantic name
                                'item_name': data_item.name,
                                'item_type': data_item.type
                            })
                            break

                if intent_sources:
                    filter_intent[port_name] = intent_sources

            # Store the intent for post-execution reconciliation
            if filter_intent:
                block.metadata["_filter_intent"] = filter_intent
                # Update block in repository
                try:
                    self._block_repo.update(block)
                except Exception as e:
                    Log.debug(f"BlockExecutionEngine: Could not save filter intent for block {block.name}: {e}")

    def _reconcile_filters_after_execution(self, blocks: List[Block]) -> None:
        """
        Reconcile filter selections after execution using captured intent.

        For blocks with _filter_intent, find equivalent data items from the same sources
        and re-apply the filter selections.

        Args:
            blocks: List of blocks to process
        """
        for block in blocks:
            filter_intent = block.metadata.get("_filter_intent")
            if not filter_intent:
                continue

            # Rebuild filter selections based on intent
            new_filter_selections = {}

            for port_name, intent_sources in filter_intent.items():
                matched_output_names = []

                for intent_source in intent_sources:
                    source_block_id = intent_source['source_block_id']
                    source_output_name = intent_source['source_output_name']
                    expected_output_name = intent_source.get('output_name')  # Semantic output name

                    # Find new data items from the same source matching by output_name
                    try:
                        source_items = self._data_item_repo.list_by_block(source_block_id)
                        matching_items = [
                            item for item in source_items
                            if item.metadata.get('output_port') == source_output_name
                            and item.metadata.get('output_name') == expected_output_name
                        ]

                        # Match by semantic output name
                        if matching_items and expected_output_name:
                            matched_output_names.append(expected_output_name)
                        elif matching_items:
                            # Fallback: use output_name from first matching item
                            output_name = matching_items[0].metadata.get('output_name')
                            if output_name:
                                matched_output_names.append(output_name)

                    except Exception as e:
                        Log.debug(f"BlockExecutionEngine: Could not reconcile filter for {source_block_id}.{source_output_name}: {e}")
                        continue

                if matched_output_names:
                    new_filter_selections[port_name] = matched_output_names

            # Apply reconciled selections
            if new_filter_selections:
                block.metadata["filter_selections"] = new_filter_selections
                Log.info(f"BlockExecutionEngine: Reconciled filter selections for block '{block.name}'")

            # Clean up temporary intent data
            if "_filter_intent" in block.metadata:
                del block.metadata["_filter_intent"]

            # Update block in repository
            try:
                self._block_repo.update(block)
            except Exception as e:
                Log.debug(f"BlockExecutionEngine: Could not save reconciled filters for block {block.name}: {e}")
    
    def _save_block_outputs(self, block: Block, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save block outputs to database and update block local state.
        
        Deletes any existing data items for the block before saving new ones
        to prevent duplicate data items from accumulating.
        
        Args:
            block: Block that produced outputs
            outputs: Dictionary mapping output port names to DataItem instances
            
        Returns:
            Dictionary mapping output port names to data item IDs
        """
        output_ids = {}
        
        if not self._data_item_repo:
            Log.warning("ExecutionEngine: No data_item_repo available, outputs not saved")
            return output_ids
        
        # Delete old data items for this block to prevent duplicates
        deleted_count = self._data_item_repo.delete_by_block(block.id)
        if deleted_count > 0:
            Log.debug(f"ExecutionEngine: Cleared {deleted_count} old data item(s) for block '{block.name}'")
        
        # Import helper for default output names
        from src.application.processing.output_name_helpers import make_default_output_name
        
        for port_name, port_data in outputs.items():
            # Handle list of data items
            if isinstance(port_data, list):
                item_ids = []
                for data_item in port_data:
                    # Ensure data item has correct block_id
                    if data_item.block_id != block.id:
                        data_item.block_id = block.id
                    # Store port name in metadata for reconstruction later
                    if 'output_port' not in data_item.metadata:
                        data_item.metadata['output_port'] = port_name
                    # Set output_name if not already set by processor
                    if 'output_name' not in data_item.metadata:
                        data_item.metadata['output_name'] = make_default_output_name(port_name)
                    created_item = self._data_item_repo.create(data_item)
                    item_ids.append(created_item.id)
                output_ids[port_name] = item_ids
                Log.debug(f"ExecutionEngine: Saved {len(item_ids)} data items on port '{port_name}'")
            # Handle single data item
            else:
                data_item = port_data
                if data_item.block_id != block.id:
                    data_item.block_id = block.id
                # Store port name in metadata for reconstruction later
                if 'output_port' not in data_item.metadata:
                    data_item.metadata['output_port'] = port_name
                # Set output_name if not already set by processor
                if 'output_name' not in data_item.metadata:
                    data_item.metadata['output_name'] = make_default_output_name(port_name)
                created_item = self._data_item_repo.create(data_item)
                output_ids[port_name] = created_item.id
                Log.debug(f"ExecutionEngine: Saved data item '{data_item.name}' on port '{port_name}'")
        
        # Update block local state with output references
        # This ensures outputs are available for downstream blocks to pull
        # We merge with existing state to preserve any inputs that might be there
        if self._block_local_state_repo and output_ids:
            try:
                # Get existing local state (preserve any existing data)
                current_state = self._block_local_state_repo.get_inputs(block.id) or {}
                # Update with output references (merge, don't overwrite)
                for port_name, item_id in output_ids.items():
                    current_state[port_name] = item_id
                # Save updated state
                self._block_local_state_repo.set_inputs(block.id, current_state)
                Log.debug(f"ExecutionEngine: Updated local state for block '{block.name}' with {len(output_ids)} output reference(s)")

            except Exception as e:
                Log.warning(f"ExecutionEngine: Failed to update local state for block '{block.name}': {e}")
        
        return output_ids
    
    def _release_audio_data(self, outputs: Dict[str, Any]):
        """
        Release in-memory audio data from output DataItems after persistence.
        
        Audio numpy arrays can be very large (millions of samples). Once outputs
        are saved to the database and audio files exist on disk, there is no reason
        to keep the raw audio in RAM. The AudioDataItem.get_audio_data() method
        will lazy-load from file_path if needed later.
        
        Args:
            outputs: Dictionary mapping output port names to DataItem instances
        """
        from src.shared.domain.entities import AudioDataItem
        
        released_count = 0
        for port_name, port_data in outputs.items():
            items = port_data if isinstance(port_data, list) else [port_data]
            for item in items:
                if isinstance(item, AudioDataItem) and hasattr(item, 'release_audio_data'):
                    item.release_audio_data()
                    released_count += 1
        
        if released_count > 0:
            Log.debug(f"ExecutionEngine: Released in-memory audio data for {released_count} AudioDataItem(s)")
