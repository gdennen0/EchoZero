# Block Implementation Guide

## Step-by-Step Implementation

### Step 1: Define Block Type

**1.1 Determine Block Type**
- What does this block do?
- What are inputs and outputs?
- What category (Input, Analysis, Transform, Output)?

**1.2 Register in BlockRegistry**
- Add entry in `src/application/block_registry.py`
- Define input/output ports with types
- Set display name and category

### Step 2: Create Processor

**2.1 Create Processor File**
- Create `src/application/blocks/my_block.py`
- Use template from PRESET.md
- Implement `BlockProcessor` subclass

**2.2 Implement Core Methods**
- `can_process()` - Check block type
- `get_block_type()` - Return type string
- `get_status_levels()` - Return status level definitions
- `process()` - Main processing logic

**2.3 Implement Optional Methods**
- `validate_configuration()` - Pre-execution config validation
- `get_expected_outputs()` - Expected output descriptions
- `step_pre_process()` / `step_post_process()` - Pre/post hooks

**2.4 Add Error Handling**
- Validate inputs
- Handle processing errors with `ProcessingError`
- Return clear error messages

### Step 3: Register Processor

**3.1 Auto-Registration**
- Call `register_processor_class()` at module level
- Add import to `src/application/blocks/__init__.py`
- Processor auto-registers on import

### Step 4: Implement Processing Logic

**4.1 Input Validation**
- Check required inputs exist
- Validate input types
- Handle missing inputs gracefully

**4.2 Process Data**
- Access block metadata for settings
- Process inputs according to block logic
- Create appropriate output DataItems
- Use progress tracking for long operations (see Step 9)

**4.3 Output Creation**
- Create outputs matching output port names
- Set correct DataItem types
- Handle block ownership (block_id) for owned data

### Step 5: Resource Management

**5.1 Identify Resources**
- Timers, media players, UI windows?
- File handles, network connections?
- Cached data that needs cleanup?

**5.2 Implement cleanup()**
- Stop/close all resources
- Delete Qt objects with deleteLater()
- Clear cached data
- See `modules/commands/cleanup/` for patterns

### Step 6: Add Settings (If Needed)

**6.1 Define Settings Schema**
- Create dataclass with defaults
- See `modules/patterns/settings_abstraction/`

**6.2 Create Settings Manager**
- Inherit from BlockSettingsManager (`src/application/settings/block_settings.py`)
- Add property accessors with validation
- Settings stored in `block.metadata` (not preferences)

**6.3 Integrate in Processor**
- Access settings from block.metadata
- Use settings in processing logic

### Step 7: Create UI Panel (If Needed)

**7.1 Create Panel File**
- Create `ui/qt_gui/block_panels/my_block_panel.py`
- Inherit from `BlockPanelBase` (`ui/qt_gui/block_panels/block_panel_base.py`)

**7.2 Implement Panel**
- Override `create_content_widget()` for block-specific UI
- Use `set_block_metadata_key()` for undoable metadata updates
- Use `set_multiple_metadata()` for batch updates
- Implement `refresh()` method for UI state sync
- Use `create_filter_widget()` for data filter components
- Use `create_expected_outputs_display()` for output info
- Use `add_port_filter_sections()` for auto-generated filter sections

### Step 8: Register Quick Actions (If Needed)

**8.1 Register Actions**
- Add quick actions in `src/application/blocks/quick_actions.py`
- Use `@quick_action()` decorator for block-specific actions
- Use `@common_action()` decorator for actions shared across block types
- Choose appropriate `ActionCategory` (EXECUTE, CONFIGURE, FILE, EDIT, VIEW, EXPORT)

**8.2 Quick Action Pattern**
```python
from src.application.blocks.quick_actions import quick_action, ActionCategory

@quick_action("MyBlock", "My Action", category=ActionCategory.EXECUTE)
def my_action(facade, block):
    # Read from settings manager (single source of truth)
    # Perform action
    pass
```

### Step 9: Add Progress Tracking (If Long Operations)

**9.1 Simple Progress (Subprocess Events)**
- Use `ProgressTracker` from metadata for block-level progress
- See `src/features/execution/application/progress_tracker.py`

```python
from src.features.execution.application.progress_helpers import get_progress_tracker

def process(self, block, inputs, metadata=None):
    tracker = get_progress_tracker(metadata)
    if tracker:
        tracker.start("Loading data")
        # ... work ...
        tracker.update(50, 100, "Processing")
        # ... work ...
        tracker.complete()
```

**9.2 Advanced Progress (Context Managers)**
- Use `ProgressContext` for hierarchical operations
- See `modules/patterns/progress_tracking/` for full guide

### Step 10: Testing

**10.1 Unit Tests**
- Test processor logic
- Test error handling
- Test input validation
- Create in `tests/unit/` or `tests/application/`

**10.2 Integration Tests**
- Test in execution engine
- Test with real data
- Test cleanup
- Create in `tests/integration/`

**10.3 Manual Testing**
- Test in GUI (if panel created)
- Test error scenarios
- Test quick actions

## Common Block Types

### Input Blocks
- Load data from external sources
- No input ports (or minimal)
- Output ports provide data

Example: `LoadAudioBlockProcessor`, `SetlistAudioInputBlockProcessor`

### Analysis Blocks
- Analyze input data
- Produce analysis results (events, classifications)
- May have settings for analysis parameters

Example: `DetectOnsetsBlockProcessor`, `NoteExtractorBasicPitchProcessor`

### Transform Blocks
- Transform input data
- Output transformed data
- May have multiple input/output ports

Example: `SeparatorBlockProcessor`, `AudioPreprocessor`

### Classification Blocks
- Classify audio or events using ML models
- Support PyTorch and TensorFlow backends

Example: `PytorchAudioClassifyBlockProcessor`, `TensorflowClassifyBlockProcessor`

### Output Blocks
- Export or visualize data
- May not have output ports
- May have UI components

Example: `EditorBlockProcessor`, `ExportAudioBlockProcessor`

### Training Blocks
- Train ML models on audio/event data
- Long-running operations requiring progress tracking

Example: `PytorchAudioTrainerBlockProcessor`, `PytorchDrumTrainerBlockProcessor`

## Best Practices

**Keep It Simple**
- One block, one responsibility
- Clear input/output contracts
- Minimal dependencies

**Error Handling**
- Validate inputs early
- Provide clear error messages via ProcessingError
- Don't fail silently
- Use `validate_configuration()` for pre-flight checks

**Resource Management**
- Always implement cleanup() if resources used
- Use deleteLater() for Qt objects
- Clear references to prevent leaks

**Settings**
- Use settings_abstraction pattern (BlockSettingsManager)
- Provide sensible defaults
- Validate settings values
- Store in block.metadata

**Quick Actions**
- Read current values from settings manager (single source of truth)
- Use appropriate ActionCategory
- Keep actions focused and atomic

**Testing**
- Test happy path
- Test error cases
- Test edge cases
- Test cleanup

## Common Pitfalls

**Forgetting Registration**
- Block won't work if not registered
- Check BlockRegistry (`src/application/block_registry.py`) and processor registration
- Ensure import in `src/application/blocks/__init__.py`

**Missing Cleanup**
- Resources accumulate, causing leaks
- Always implement cleanup() if resources used

**Wrong DataItem Types**
- Execution engine validates types
- Match port types exactly

**Not Handling Errors**
- Silent failures are hard to debug
- Always raise ProcessingError with clear messages

**Over-Complicating**
- Start simple, iterate
- Don't add features "just in case"

**Forgetting Quick Actions**
- Common actions should be registered for UI access
- Read settings from manager, not hardcoded values


