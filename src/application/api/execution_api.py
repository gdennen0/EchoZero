"""
Execution API

Clean API for block execution operations.
Designed for future UI integration.
"""
from typing import Optional, Dict, Any, List, Callable

from src.features.execution.application import BlockExecutionEngine, ExecutionResult
from src.application.processing.block_processor import BlockProcessor
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.utils.message import Log


class ExecutionAPI:
    """
    Clean API for block execution operations.
    
    Provides a stable interface for UI implementations.
    All methods return DTOs (Data Transfer Objects) for UI consumption.
    
    Note: This wraps BlockExecutionEngine directly (ExecutionService removed as unnecessary wrapper).
    """
    
    def __init__(self, execution_engine: BlockExecutionEngine):
        """
        Initialize Execution API.
        
        Args:
            execution_engine: BlockExecutionEngine instance
        """
        self._engine = execution_engine
        Log.info("ExecutionAPI: Initialized")
    
    def register_processor(self, processor: BlockProcessor) -> None:
        """
        Register a block processor.
        
        Args:
            processor: BlockProcessor implementation
        """
        self._engine.register_processor(processor)
    
    def validate_project(self, project_id: str) -> Dict[str, Any]:
        """
        Validate that a project's block graph is executable.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Dictionary with 'valid' (bool) and 'error' (str or None)
        """
        is_valid, error = self._engine.validate_project(project_id)
        return {
            "valid": is_valid,
            "error": error
        }
    
    def execute_block(
        self,
        block: Block,
        inputs: Optional[Dict[str, DataItem]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a single block (useful for testing).
        
        Args:
            block: Block entity to execute
            inputs: Optional input data items
            metadata: Optional metadata for processing
            
        Returns:
            Dictionary mapping output port names to data item DTOs
            
        Raises:
            ValueError: If no processor registered for block type
        """
        outputs = self._engine.execute_block(block, inputs=inputs, metadata=metadata)
        return {
            port_name: self._data_item_to_dto(data_item)
            for port_name, data_item in outputs.items()
        }
    
    def _execution_result_to_dto(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        Convert ExecutionResult to DTO.
        
        Args:
            result: ExecutionResult instance
            
        Returns:
            Execution result DTO dictionary
        """
        return {
            "success": result.success,
            "executed_blocks": result.executed_blocks,
            "failed_blocks": result.failed_blocks,
            "output_data": {
                block_id: {
                    port_name: self._data_item_to_dto(data_item)
                    for port_name, data_item in ports.items()
                }
                for block_id, ports in result.output_data.items()
            },
            "errors": result.errors
        }
    
    def _data_item_to_dto(self, data_item: DataItem) -> Dict[str, Any]:
        """
        Convert DataItem to DTO.
        
        Args:
            data_item: DataItem instance
            
        Returns:
            Data item DTO dictionary
        """
        dto = {
            "id": data_item.id,
            "block_id": data_item.block_id,
            "name": data_item.name,
            "type": data_item.type,
            "created_at": data_item.created_at.isoformat() if data_item.created_at else None,
            "file_path": data_item.file_path,
            "metadata": data_item.metadata or {}
        }
        
        # Add type-specific fields
        if data_item.type == "Audio":
            from src.shared.domain.entities import AudioDataItem
            if isinstance(data_item, AudioDataItem):
                dto["sample_rate"] = data_item.sample_rate
                dto["length_ms"] = data_item.length_ms
        
        elif data_item.type == "Event":
            from src.shared.domain.entities import EventDataItem
            if isinstance(data_item, EventDataItem):
                dto["event_count"] = data_item.event_count
        
        return dto

