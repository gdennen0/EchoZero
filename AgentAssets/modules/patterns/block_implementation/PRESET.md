# Block Implementation Preset

## Block Processor Template

The BlockProcessor follows a 5-step execution flow managed by the execution engine:
1. `step_clear_local_data()` - Clear owned data items
2. Pull upstream data (engine-managed)
3. `step_pre_process()` - Pre-processing hook
4. `process()` - Main processing (required)
5. `step_post_process()` - Post-processing hook

```python
"""
[Block Name] Block Processor

[Brief description of what this block does]
"""
from typing import Dict, Optional, Any
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities.data_item import DataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log


class MyBlockProcessor(BlockProcessor):
    """
    Processor for [BlockType] block type.
    
    [Detailed description of block behavior, inputs, outputs, settings]
    """
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return block.type == "[BlockType]"
    
    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return "[BlockType]"
    
    def get_status_levels(self) -> dict:
        """
        Return status level definitions for this block type.
        
        Returns dict of {status_key: {"label": str, "description": str}}
        """
        return {}
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process [BlockType] block.
        
        Args:
            block: Block entity with metadata
            inputs: Dict of input DataItems (keys match input port names)
            metadata: Optional processing metadata (contains progress tracker)
            
        Returns:
            Dict of output DataItems (keys match output port names)
            
        Raises:
            ProcessingError: If processing fails
        """
        Log.info(f"[BlockType]Processor: Processing block '{block.name}'")
        
        # Validate inputs
        # Process data
        # Create outputs
        # Return outputs dict
        
        raise NotImplementedError("Implement processing logic")
    
    def validate_configuration(self, block: Block) -> Optional[str]:
        """
        Validate block configuration before execution.
        
        Returns None if valid, or an error message string if invalid.
        """
        return None
    
    def get_expected_outputs(self, block: Block) -> Dict[str, str]:
        """
        Return expected output descriptions for this block.
        
        Returns dict of {port_name: description}
        """
        return {}
    
    def cleanup(self, block: Block) -> None:
        """
        Clean up resources for this block.
        
        Called when block is removed or project is unloaded.
        Override if block uses resources (timers, media players, UI windows, etc.)
        """
        # Default: no cleanup needed
        pass


# Auto-register this processor
register_processor_class(MyBlockProcessor)
```

## Command Template (EchoZeroCommand)

Commands use Qt's QUndoCommand for undo/redo. All commands inherit from EchoZeroCommand:

```python
from src.application.commands.base_command import EchoZeroCommand

class MyCommand(EchoZeroCommand):
    """
    Brief description of what the command does.
    
    Redo: What happens when executed/redone
    Undo: What happens when undone
    """
    
    def __init__(self, facade: "ApplicationFacade", param1, param2):
        super().__init__(facade, f"My Operation: {param1}")
        self._param1 = param1
        self._param2 = param2
        self._original_state = None
    
    def redo(self):
        """Execute the operation."""
        if self._original_state is None:
            self._original_state = self._get_current_state()
        self._facade.do_something(self._param1, self._param2)
    
    def undo(self):
        """Reverse the operation."""
        if self._original_state is not None:
            self._facade.restore_state(self._original_state)

# Execute via: facade.command_bus.execute(MyCommand(facade, p1, p2))
```

## Block Registration Checklist

- [ ] Processor class inherits from `BlockProcessor`
- [ ] `can_process()` returns True for block type
- [ ] `get_block_type()` returns block type string
- [ ] `get_status_levels()` returns status definitions
- [ ] `process()` implemented with proper error handling
- [ ] `cleanup()` implemented if resources are used
- [ ] `register_processor_class()` called at module level
- [ ] Module imported in `src/application/blocks/__init__.py`
- [ ] Block type registered in `BlockRegistry` (`src/application/block_registry.py`)
- [ ] Quick actions registered (if applicable)

## Block Registry Entry

Blocks must be registered in `BlockRegistry` (`src/application/block_registry.py`) with:
- Block type name
- Display name
- Input ports (name and type)
- Output ports (name and type)
- Category

Example:
```python
registry.register(BlockTypeMetadata(
    name="My Block",
    type_id="MyBlock",
    inputs={"audio": AUDIO_TYPE},
    outputs={"audio": AUDIO_TYPE, "events": EVENT_TYPE}
))
```

## Quick Actions (Optional)

Register block-specific actions accessible from the UI:

```python
from src.application.blocks.quick_actions import quick_action, ActionCategory

@quick_action("MyBlock", "My Action", category=ActionCategory.EXECUTE)
def my_action(facade, block):
    # Action logic here
    pass
```

## UI Panel (Optional)

If block needs UI configuration:
1. Create panel in `ui/qt_gui/block_panels/`
2. Inherit from `BlockPanelBase` (`ui/qt_gui/block_panels/block_panel_base.py`)
3. Use settings_abstraction if configurable
4. Implement `create_content_widget()` for block-specific UI
5. Implement `refresh()` to load current state
6. Use `set_block_metadata_key()` for undoable metadata updates
7. Use `create_filter_widget()` / `create_expected_outputs_display()` for common components

## Common Patterns

### Input Validation
```python
audio_input = inputs.get("audio")
if audio_input is None:
    raise ProcessingError("Missing required input: audio")
if not isinstance(audio_input, AudioDataItem):
    raise ProcessingError("Input 'audio' must be AudioDataItem")
```

### Settings Access
```python
# Access block metadata for settings
model = block.metadata.get("model", "default")
threshold = block.metadata.get("threshold", 0.5)
```

### Output Creation
```python
outputs = {}
outputs["audio"] = AudioDataItem(audio_data, sample_rate)
outputs["events"] = EventDataItem(events, block_id=block.id)
return outputs
```

### Error Handling
```python
try:
    result = process_audio(audio_data)
except Exception as e:
    raise ProcessingError(f"Failed to process audio: {e}") from e
```

### Progress Tracking in Processors
```python
from src.features.execution.application.progress_helpers import get_progress_tracker

def process(self, block, inputs, metadata=None):
    tracker = get_progress_tracker(metadata)
    if tracker:
        tracker.start("Loading audio")
        # ... load ...
        tracker.update(50, 100, "Processing audio")
        # ... process ...
        tracker.complete()
    return outputs
```

### Resource Cleanup
```python
def cleanup(self, block: Block) -> None:
    """Clean up resources"""
    if hasattr(self, '_timer'):
        self._timer.stop()
        self._timer.deleteLater()
    if hasattr(self, '_player'):
        self._player.stop()
        self._player.deleteLater()
```


