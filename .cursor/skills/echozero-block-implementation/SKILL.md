---
name: echozero-block-implementation
description: Create and implement block processors in EchoZero. Use when making a new block, adding block types, implementing BlockProcessor, block cleanup, quick actions, or when the user asks how to create blocks, add block processors, or register blocks.
---

# Block Implementation

## Quick Start

1. Add entry in `src/application/block_registry.py`
2. Create processor in `src/application/blocks/my_block.py`
3. Import in `src/application/blocks/__init__.py`
4. Add quick actions if needed
5. Create UI panel if needed (inherit BlockPanelBase)

## 5-Step Execution Flow

BlockProcessor flow (engine-managed):
1. `step_clear_local_data()` - Clear owned data items
2. Pull upstream data (engine)
3. `step_pre_process()` - Pre-processing hook
4. `process()` - Main logic (required)
5. `step_post_process()` - Post-processing hook

## Processor Template

```python
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.application.blocks import register_processor_class

class MyBlockProcessor(BlockProcessor):
    def can_process(self, block) -> bool:
        return block.type == "MyBlock"

    def get_block_type(self) -> str:
        return "MyBlock"

    def get_status_levels(self) -> dict:
        return {}

    def process(self, block, inputs, metadata=None):
        # Validate inputs
        audio = inputs.get("audio")
        if audio is None:
            raise ProcessingError("Missing required input: audio")
        # Process, create outputs
        return {"output": result}

    def cleanup(self, block) -> None:
        """Override if block uses timers, media players, UI windows"""
        pass

register_processor_class(MyBlockProcessor)
```

## Block Registry Entry

In `src/application/block_registry.py`:

```python
registry.register(BlockTypeMetadata(
    name="My Block",
    type_id="MyBlock",
    inputs={"audio": AUDIO_TYPE},
    outputs={"audio": AUDIO_TYPE},
    category=BlockCategory.TRANSFORM
))
```

## Checklist

- [ ] Processor inherits BlockProcessor
- [ ] `can_process()`, `get_block_type()`, `process()` implemented
- [ ] `cleanup()` if resources used (timers, players, etc.)
- [ ] `register_processor_class()` at module level
- [ ] Import in `src/application/blocks/__init__.py`
- [ ] Entry in BlockRegistry

## Input Validation Pattern

```python
audio = inputs.get("audio")
if audio is None:
    raise ProcessingError("Missing required input: audio")
if not isinstance(audio, AudioDataItem):
    raise ProcessingError("Input 'audio' must be AudioDataItem")
```

## Progress in Processors

```python
from src.features.execution.application.progress_helpers import get_progress_tracker

def process(self, block, inputs, metadata=None):
    tracker = get_progress_tracker(metadata)
    if tracker:
        tracker.start("Loading data")
        # ... work ...
        tracker.update(50, 100, "Processing")
        tracker.complete()
    return outputs
```

## Quick Actions

```python
from src.application.blocks.quick_actions import quick_action, ActionCategory

@quick_action("MyBlock", "My Action", category=ActionCategory.EXECUTE)
def my_action(facade, block):
    # Read from settings manager
    pass
```

## UI Panel

Inherit from `BlockPanelBase` (`ui/qt_gui/block_panels/block_panel_base.py`):
- Override `create_content_widget()` for block UI
- Use `set_block_metadata_key()` for undoable updates
- Use `create_filter_widget()`, `create_expected_outputs_display()`
- Implement `refresh()`

## Common Pitfalls

- Forgetting BlockRegistry entry and `__init__.py` import
- Missing `cleanup()` when using resources
- Wrong DataItem types (must match port types)
- Silent failures - always raise ProcessingError with clear messages

## Related Skills

- echozero-settings-abstraction - Block configurable settings
- echozero-progress-tracking - Long-running operations
- echozero-ui-components - Block panels, design system
