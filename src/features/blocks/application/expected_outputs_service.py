"""
Expected Outputs Service

Dedicated service for calculating expected outputs for blocks.
Handles connection-based output calculation (e.g., DetectOnsets outputs events matching input audio).

Separates concerns:
- Processors: Define static outputs (Separator: 4 stems, Editor: main/edited)
- Service: Calculates dynamic outputs based on connections (DetectOnsets: events matching input audio)
- UI: Displays what's in block.metadata['expected_outputs']
"""
from typing import Dict, List, Optional
from src.features.blocks.domain import Block
from src.features.connections.domain import ConnectionRepository
from src.features.blocks.domain import BlockRepository
from src.application.processing.block_processor import BlockProcessor
from src.application.processing.output_name_helpers import parse_output_name, make_output_name
from src.utils.message import Log


class ExpectedOutputsService:
    """
    Service for calculating expected outputs for blocks.
    
    Handles:
    - Static outputs (from processor.get_expected_outputs())
    - Connection-based outputs (blocks that output based on inputs)
    """
    
    # Blocks that output based on their inputs
    # Format: {block_type: {input_port: (output_port, transform_func)}}
    # transform_func: (source_output_name) -> output_name
    CONNECTION_BASED_OUTPUTS = {
        "DetectOnsets": {
            "audio": ("events", lambda source: _convert_port_name(source, "audio", "events"))
        },
        "LearnedOnsetDetector": {
            "audio": ("events", lambda source: _convert_port_name(source, "audio", "events"))
        },
        "AudioFilter": {
            "audio": ("audio", lambda source: source)  # Pass-through: preserves upstream names
        },
        "TensorFlowClassify": {
            "events": ("events", lambda source: _add_classified_suffix(source))  # Adds "_classified" suffix
        },
        "PyTorchAudioClassify": {
            "events": ("events", lambda source: _add_classified_suffix(source))  # Adds "_classified" suffix
        },
    }
    
    def __init__(
        self,
        connection_repo: ConnectionRepository,
        block_repo: BlockRepository
    ):
        """
        Initialize expected outputs service.
        
        Args:
            connection_repo: Repository for connections
            block_repo: Repository for blocks
        """
        self._connection_repo = connection_repo
        self._block_repo = block_repo
    
    def calculate_expected_outputs(
        self,
        block: Block,
        processor: BlockProcessor,
        facade=None
    ) -> Dict[str, List[str]]:
        """
        Calculate expected outputs for a block.
        
        First gets static outputs from processor, then applies connection-based
        transformations if applicable. Applies filter selections to only include
        outputs that will actually be processed.
        
        Args:
            block: Block entity
            processor: BlockProcessor for this block
            facade: Optional ApplicationFacade for accessing services
        
        Returns:
            Dictionary mapping port_name -> list of expected output names
            (filtered based on filter_selections if they exist)
        """
        # Get static outputs from processor (no connection info needed)
        static_outputs = processor.get_expected_outputs(block)
        
        # Check if this block type outputs based on connections
        block_type = block.type
        if block_type not in self.CONNECTION_BASED_OUTPUTS:
            # #region agent log
            try:
                import json, time
                if block_type == "AudioFilter":
                    with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                        _f.write(json.dumps({"hypothesisId": "H-D", "location": "expected_outputs_service.py:not-in-connection-based", "message": "AudioFilter NOT in CONNECTION_BASED_OUTPUTS - returning static", "data": {"block_type": block_type, "static_outputs": static_outputs}, "timestamp": int(time.time() * 1000)}) + "\n")
            except Exception:
                pass
            # #endregion
            return static_outputs
        
        # Apply connection-based transformations
        connection_rules = self.CONNECTION_BASED_OUTPUTS[block_type]
        result = static_outputs.copy()
        
        for input_port, (output_port, transform_func) in connection_rules.items():
            # Find all connections to this input port (multiple connections allowed)
            connections = self._connection_repo.list_by_block(block.id)
            input_connections = [
                conn for conn in connections
                if conn.target_block_id == block.id and conn.target_input_name == input_port
            ]
            
            if not input_connections:
                # No connection - keep static output
                continue
            
            # Get filter selections for this input port (if any)
            # filter_selections can be:
            #   - dict format: {name: bool} -- only include names where value is True
            #   - list format: [name] -- all listed names are selected
            filter_selections = block.metadata.get("filter_selections", {})
            raw_selection = filter_selections.get(input_port)
            if isinstance(raw_selection, dict):
                # Dict format: only include keys with True value
                selected_set = {k for k, v in raw_selection.items() if v}
            elif raw_selection:
                selected_set = set(raw_selection)
            else:
                selected_set = None
            
            # Check if filters exist and are empty (all deselected)
            has_empty_filter = selected_set is not None and len(selected_set) == 0
            
            # Collect transformed outputs from all source blocks
            all_transformed_outputs = []
            has_empty_source = False  # Track if any source has empty outputs (all filters off)
            
            for input_connection in input_connections:
                # Get source block
                source_block = self._block_repo.get_by_id(input_connection.source_block_id)
                if not source_block:
                    continue
                
                # Get source block's processor
                if not facade:
                    continue
                
                source_processor = facade.execution_engine.get_processor(source_block)
                if not source_processor:
                    continue
                
                # Recursively calculate source block's expected outputs
                source_expected = self.calculate_expected_outputs(
                    source_block,
                    source_processor,
                    facade=facade
                )
                # Use .get() without default to distinguish None (missing) from [] (empty filter)
                source_outputs = source_expected.get(input_connection.source_output_name)
                
                # Check if source block has empty outputs (all filters off upstream)
                # Empty list [] explicitly means all filters were deselected upstream
                # None means no outputs defined for this port (different case)
                if source_outputs == []:
                    # Source has all filters off - propagate empty outputs downstream
                    has_empty_source = True
                    # Don't break - check all sources, but mark that we have empty source
                    continue
                
                if source_outputs is None or not source_outputs:
                    # No outputs defined (not the same as empty filter) - skip this source
                    continue
                
                # Apply filter selections if they exist
                # Only include source outputs that match the filter
                filtered_source_outputs = source_outputs
                if selected_set is not None:
                    # Filter to only selected outputs
                    # If selected_set is empty, filtered_source_outputs will be empty
                    filtered_source_outputs = [
                        output for output in source_outputs
                        if output in selected_set
                    ]
                
                # Transform filtered source outputs using the transform function
                for source_output in filtered_source_outputs:
                    try:
                        transformed = transform_func(source_output)
                        if transformed and transformed not in all_transformed_outputs:
                            all_transformed_outputs.append(transformed)
                    except Exception as e:
                        Log.debug(f"ExpectedOutputsService: Failed to transform {source_output}: {e}")
                        continue
            
            # Set output port based on filter state
            if has_empty_filter or has_empty_source:
                # All filters deselected (either here or upstream) - explicitly set to empty list
                result[output_port] = []
            elif all_transformed_outputs:
                # Has filtered outputs - use them
                result[output_port] = sorted(all_transformed_outputs)
            else:
                # No filtered outputs but filters weren't explicitly set to empty
                # For connection-based outputs with no filters configured, set to empty instead of "main"
                # This ensures blocks don't show "main" when they depend on filtered inputs
                result[output_port] = []
        
        # #region agent log
        try:
            import json, time
            if block_type == "AudioFilter":
                with open("/Users/gdennen/Projects/EchoZero/.cursor/debug.log", "a") as _f:
                    _f.write(json.dumps({"hypothesisId": "H-D", "location": "expected_outputs_service.py:connection-based-result", "message": "AudioFilter CONNECTION_BASED result", "data": {"result": result, "block_name": block.name}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        
        return result


def _convert_port_name(output_name: str, from_port: str, to_port: str) -> Optional[str]:
    """
    Convert output name from one port type to another.
    
    Example: "audio:vocals" -> "events:vocals"
    
    Args:
        output_name: Source output name (e.g., "audio:vocals")
        from_port: Source port name (e.g., "audio")
        to_port: Target port name (e.g., "events")
    
    Returns:
        Converted output name (e.g., "events:vocals") or None if conversion fails
    """
    try:
        port_name, item_name = parse_output_name(output_name)
        if port_name == from_port:
            return make_output_name(to_port, item_name)
    except ValueError:
        pass
    return None


def _add_classified_suffix(output_name: str) -> Optional[str]:
    """
    Add "_classified" suffix to output name to indicate classified events.
    
    Example: "events:drums" -> "events:drums_classified"
    
    Args:
        output_name: Source output name (e.g., "events:drums")
    
    Returns:
        Output name with "_classified" suffix (e.g., "events:drums_classified") or None if parsing fails
    """
    try:
        port_name, item_name = parse_output_name(output_name)
        # Add "_classified" suffix to item name
        classified_item_name = f"{item_name}_classified"
        return make_output_name(port_name, classified_item_name)
    except ValueError:
        pass
    return None

