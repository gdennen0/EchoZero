"""
Type validation utilities for block processing.

Provides validation functions to ensure data types match expected port types
at key points: pull inputs, execution, and live edits.
"""
from typing import Dict, List, Optional
from src.shared.domain.entities import DataItem
from src.shared.domain.value_objects.port_type import PortType
from src.application.processing.block_processor import ProcessingError
from src.utils.message import Log


def types_compatible(actual_type: PortType, expected_type: PortType) -> bool:
    """
    Check if actual type is compatible with expected type.
    
    Rules:
    - Exact match: compatible
    - Future: Subtype relationships (Audio subtypes compatible with Audio)
    
    Args:
        actual_type: The actual PortType
        expected_type: The expected PortType
        
    Returns:
        True if compatible, False otherwise
    """
    # For now, only exact matches are compatible
    # Future: Add ANY_TYPE support and subtype relationships
    return actual_type == expected_type


def validate_input_types(
    block_type: str,
    inputs: Dict[str, DataItem],
    expected_inputs: Dict[str, PortType]
) -> List[str]:
    """
    Validate that inputs match expected types.
    
    Args:
        block_type: Block type identifier
        inputs: Actual input DataItems
        expected_inputs: Expected input port types
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    for port_name, data_item in inputs.items():
        expected_type = expected_inputs.get(port_name)
        if not expected_type:
            # Port not declared - allow it (flexible)
            continue
        
        # Get actual type from DataItem
        actual_type_name = data_item.type if hasattr(data_item, 'type') else None
        if not actual_type_name:
            errors.append(f"Port {port_name}: Cannot determine type of DataItem")
            continue
        
        from src.shared.domain.value_objects.port_type import get_port_type
        try:
            actual_type = get_port_type(actual_type_name)
        except Exception as e:
            errors.append(f"Port {port_name}: Invalid type '{actual_type_name}': {e}")
            continue
        
        if not types_compatible(actual_type, expected_type):
            errors.append(
                f"Port {port_name}: Expected {expected_type.name}, got {actual_type_name}"
            )
    
    return errors


def validate_on_pull(
    block_id: str,
    block_type: str,
    connection_map: Dict[str, tuple[str, str]],  # target_input -> (source_block_id, source_output)
    expected_inputs: Dict[str, PortType],
    data_item_repo,
    block_local_state_repo
) -> List[str]:
    """
    Validate types when pulling inputs.
    
    Args:
        block_id: Target block ID
        block_type: Target block type
        connection_map: Map of target inputs to source connections
        expected_inputs: Expected input port types
        data_item_repo: Data item repository
        block_local_state_repo: Block local state repository
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    for target_input, (source_block_id, source_output) in connection_map.items():
        expected_type = expected_inputs.get(target_input)
        if not expected_type:
            continue
        
        # Try to get source output from local state first
        source_local = block_local_state_repo.get_inputs(source_block_id) or {}
        source_ref = source_local.get(source_output)
        
        source_item = None
        if source_ref:
            # Get from local state reference
            if isinstance(source_ref, list):
                if source_ref:
                    source_item = data_item_repo.get(source_ref[0])
            else:
                source_item = data_item_repo.get(source_ref)
        
        if not source_item:
            # Try to get from data_item_repo by output_port
            source_items = data_item_repo.list_by_block(source_block_id)
            matching = [
                item for item in source_items
                if item.metadata.get("output_port") == source_output
            ]
            if matching:
                source_item = matching[0]
        
        if not source_item:
            # Can't validate - source has no data yet
            continue
        
        # Get actual type
        actual_type_name = source_item.type if hasattr(source_item, 'type') else None
        if not actual_type_name:
            errors.append(
                f"Connection {source_block_id}.{source_output} -> {block_id}.{target_input}: "
                f"Cannot determine type of source DataItem"
            )
            continue
        
        from src.shared.domain.value_objects.port_type import get_port_type
        try:
            actual_type = get_port_type(actual_type_name)
        except Exception as e:
            errors.append(
                f"Connection {source_block_id}.{source_output} -> {block_id}.{target_input}: "
                f"Invalid type '{actual_type_name}': {e}"
            )
            continue
        
        if not types_compatible(actual_type, expected_type):
            errors.append(
                f"Connection {source_block_id}.{source_output} -> {block_id}.{target_input}: "
                f"Type mismatch: expected {expected_type.name}, got {actual_type_name}"
            )
    
    return errors

