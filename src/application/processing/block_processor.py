"""
Block Processor Interface

Defines the interface for block implementations to execute processing.
Blocks must implement this interface to be executed by the pipeline.

STANDARDIZED EXECUTION FLOW (5 Steps):
======================================
All blocks follow this standardized execution flow managed by BlockExecutionEngine.
Each step has a corresponding hook method that processors can override.

STEP 1: CLEAR LOCAL DATA (step_clear_local_data)
   - DEFAULT: Clears all owned data items for this block
   - Override to: Preserve specific items, clear UI state, etc.

STEP 2: PULL UPSTREAM DATA (handled by engine)
   - Engine pulls fresh data from upstream connections
   - Cannot be overridden by processors

STEP 3: PRE-PROCESS (step_pre_process)
   - DEFAULT: No-op
   - Override to: Validate inputs, prepare state, etc.

STEP 4: PROCESS (step_process / process)
   - REQUIRED: Main processing logic
   - Transform inputs to outputs

STEP 5: POST-PROCESS (step_post_process)
   - DEFAULT: No-op
   - Override to: Update UI state, set execution flags, etc.

STEP 6: SAVE & NOTIFY (handled by engine)
   - Engine saves outputs and publishes BlockUpdated
   - Cannot be overridden by processors

Metadata Available:
-------------------
- data_item_repo: Repository for data item CRUD operations
- ui_state_service: Service for UI state access (get/set layer state, etc.)
- execution_mode: 'executable' or 'live'
- project_id: Current project identifier
- progress_tracker: For reporting execution progress
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class BlockProcessor(ABC):
    """
    Abstract base class for block implementations.
    
    Blocks must implement this interface to be executed by the pipeline.
    This separates the domain Block entity from the actual processing logic.
    
    Step-Based Execution:
    ---------------------
    The execution engine calls these steps IN ORDER:
    
    1. step_clear_local_data() - Clear owned data (DEFAULT: clears all)
    2. [Engine pulls upstream data]
    3. step_pre_process() - Pre-processing hook
    4. step_process() / process() - Main processing (REQUIRED)
    5. step_post_process() - Post-processing hook
    6. [Engine saves outputs and notifies]
    
    Override any step method to customize behavior.
    """
    
    @abstractmethod
    def can_process(self, block: Block) -> bool:
        """
        Check if this processor can handle the given block.
        
        Args:
            block: Block entity to check
            
        Returns:
            True if this processor can handle the block type
        """
        pass
    
    # =========================================================================
    # STEP 1: Clear Local Data
    # =========================================================================
    
    def step_clear_local_data(
        self,
        block: Block,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        STEP 1: Clear owned data items before processing.
        
        DEFAULT BEHAVIOR: Deletes ALL data items owned by this block.
        This ensures a clean slate before creating new outputs.
        
        Override this method to:
        - Preserve specific items (e.g., synced layers)
        - Clear UI state in addition to data items
        - Skip clearing for incremental updates
        
        Args:
            block: Block entity being executed
            metadata: Processing metadata with repos/services
        """
        metadata = metadata or {}
        data_item_repo = metadata.get("data_item_repo")
        
        if not data_item_repo:
            return
        
        try:
            owned_items = data_item_repo.list_by_block(block.id)
            deleted_count = 0
            for item in owned_items:
                try:
                    data_item_repo.delete(item.id)
                    deleted_count += 1
                except Exception as e:
                    Log.warning(f"BlockProcessor: Failed to delete item {item.id}: {e}")
            
            if deleted_count > 0:
                Log.debug(f"BlockProcessor: Cleared {deleted_count} owned data item(s) for '{block.name}'")
        except Exception as e:
            Log.warning(f"BlockProcessor: Failed to clear owned data for '{block.name}': {e}")
    
    # =========================================================================
    # STEP 3: Pre-Process
    # =========================================================================
    
    def step_pre_process(
        self,
        block: Block,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        STEP 3: Pre-processing hook called before main processing.
        
        DEFAULT BEHAVIOR: No-op.
        
        Override this method to:
        - Validate inputs before processing
        - Prepare block state
        - Load cached resources
        
        Args:
            block: Block entity being executed
            metadata: Processing metadata with repos/services
        """
        pass  # Default: no pre-processing needed
    
    # Backwards compatibility alias
    def pre_process(self, block: Block, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Backwards compatibility: calls step_pre_process."""
        self.step_pre_process(block, metadata)
    
    # =========================================================================
    # STEP 4: Process (REQUIRED)
    # =========================================================================
    
    @abstractmethod
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process a block with given inputs.
        
        Args:
            block: Block entity to process
            inputs: Dictionary mapping input port names to DataItem instances
            metadata: Optional metadata for processing (project info, etc.)
            
        Returns:
            Dictionary mapping output port names to DataItem instances
            
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    # Alias for step-based naming consistency
    def step_process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """STEP 4: Alias for process() - for naming consistency."""
        return self.process(block, inputs, metadata)
    
    # =========================================================================
    # STEP 5: Post-Process
    # =========================================================================
    
    def step_post_process(
        self,
        block: Block,
        outputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        STEP 5: Post-processing hook called after main processing.
        
        DEFAULT BEHAVIOR: Returns outputs unchanged.
        
        Override this method to:
        - Update UI state based on outputs
        - Set execution flags in block.metadata
        - Modify outputs before saving
        - Log/track execution results
        
        Args:
            block: Block entity being executed
            outputs: Outputs from process() step
            metadata: Processing metadata with repos/services
            
        Returns:
            Outputs (possibly modified) to be saved
        """
        return outputs  # Default: return unchanged
    
    @abstractmethod
    def get_block_type(self) -> str:
        """
        Get the block type this processor handles.

        Returns:
            Block type identifier (e.g., "LoadAudio", "DetectOnsets")
        """
        pass
    
    @abstractmethod
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for this block type.
        
        Status levels are evaluated in ascending priority order (0, 1, 2...).
        The first level where ANY condition is False becomes the active status.
        If all levels pass, the highest priority level is returned.
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order (lowest priority first)
        """
        pass

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate block configuration before execution.

        This includes validating filter selections, block settings, and other
        configuration that could cause execution to fail.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # Default implementation: no validation
        return []
    
    def cleanup(self, block: Block) -> None:
        """
        Clean up block resources after execution.
        
        Called after execution completes to free memory, close file handles,
        release library resources (PyTorch, TensorFlow, etc.).
        
        Default implementation does nothing. Subclasses should override
        to clean up their specific resources.
        
        Args:
            block: Block entity that was executed
        """
        pass  # Default: no cleanup needed
    
    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """
        Get expected output names for each output port.
        
        This method should be overridden by blocks that have:
        - Multiple outputs on a single port (e.g., separator with 4 stems)
        - Configuration-dependent outputs (e.g., 2-stem vs 4-stem mode)
        
        Note: For blocks that output based on connections (e.g., DetectOnsets),
        use ExpectedOutputsService which handles connection-based calculation.
        
        Args:
            block: Block entity to get expected outputs for
        
        Returns:
            Dictionary mapping port_name -> list of semantic output names
            Example: {"audio": ["audio:vocals", "audio:drums", "audio:bass", "audio:other"]}
            For single-item ports: {"events": ["events:main"]}
        """
        # Default: single output per port using port name
        return {
            port_name: [f"{port_name}:main"]
            for port_name in block.get_outputs().keys()
        }
    
    def validate_output_names(
        self,
        block: Block,
        outputs: Dict[str, Any]
    ) -> List[str]:
        """
        Validate that output DataItems have correct output_name metadata.
        
        Compares actual outputs against expected outputs declared by
        get_expected_outputs(). Returns warnings for mismatches.
        
        Args:
            block: Block entity that was processed
            outputs: Dictionary of output port names to DataItem(s) returned by process()
        
        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []
        # Get expected outputs (static only - connection-based outputs handled by ExpectedOutputsService)
        expected_outputs = self.get_expected_outputs(block)
        
        for port_name, expected_names in expected_outputs.items():
            port_data = outputs.get(port_name)
            if not port_data:
                continue
            
            # Handle list of items
            items = port_data if isinstance(port_data, list) else [port_data]
            
            actual_names = {item.metadata.get('output_name') for item in items if hasattr(item, 'metadata')}
            expected_set = set(expected_names)
            
            missing = expected_set - actual_names
            extra = actual_names - expected_set
            
            if missing:
                warnings.append(f"Port '{port_name}': Missing output_name(s): {missing}")
            if extra:
                warnings.append(f"Port '{port_name}': Unexpected output_name(s): {extra}")
        
        return warnings


class ProcessingError(Exception):
    """Exception raised during block processing"""
    
    def __init__(self, message: str, block_id: Optional[str] = None, block_name: Optional[str] = None):
        """
        Initialize processing error.
        
        Args:
            message: Error message
            block_id: Optional block ID
            block_name: Optional block name
        """
        super().__init__(message)
        self.block_id = block_id
        self.block_name = block_name
        self.message = message
    
    def __str__(self):
        if self.block_name:
            return f"Processing error in block '{self.block_name}': {self.message}"
        elif self.block_id:
            return f"Processing error in block '{self.block_id}': {self.message}"
        return f"Processing error: {self.message}"


class FilterError(ProcessingError):
    """Exception raised when filter results in empty data"""
    
    def __init__(
        self,
        message: str,
        block_id: str,
        block_name: str,
        port_name: str,
        available_items: List[str],
        selected_ids: List[str]
    ):
        """
        Initialize filter error.
        
        Args:
            message: Error message
            block_id: Block ID
            block_name: Block name
            port_name: Port name where filter was applied
            available_items: IDs of items before filtering
            selected_ids: IDs that were selected but don't match
        """
        super().__init__(message, block_id, block_name)
        self.port_name = port_name
        self.available_items = available_items
        self.selected_ids = selected_ids
        self.remedy_action = "open_filter_dialog"

