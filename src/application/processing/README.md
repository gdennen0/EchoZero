# Block Processing Pipeline

## Overview

The block processing pipeline executes blocks in a project graph in the correct order, handling data flow between blocks via connections.

## Standardized Execution Flow (6 Steps)

ALL blocks follow this standardized step-based execution flow, managed by BlockExecutionEngine.
Each step has a corresponding hook method that processors can override.

```
STEP 1: CLEAR LOCAL DATA (step_clear_local_data)
   - Engine clears local state references
   - Processor's step_clear_local_data() clears owned data items
   - DEFAULT: Deletes ALL owned data items (can override to preserve specific items)

STEP 2: PULL UPSTREAM DATA (handled by engine)
   - Engine pulls fresh data from upstream connections
   - Cannot be overridden by processors

STEP 3: PRE-PROCESS (step_pre_process)
   - Called before main processing
   - DEFAULT: No-op (override to validate inputs, prepare state)

STEP 4: PROCESS (step_process / process)
   - REQUIRED: Main processing logic
   - Transform inputs to outputs

STEP 5: POST-PROCESS (step_post_process)  
   - Called after main processing
   - DEFAULT: Returns outputs unchanged (override to update UI state, set flags)

STEP 6: SAVE & NOTIFY (handled by engine)
   - Engine saves outputs to database
   - Engine publishes BlockUpdated event
```

### Key Principles

- **No special cases**: Every block type uses the same step-based flow
- **Default clearing**: Local data is ALWAYS cleared by default
- **Modular hooks**: Override any step method to customize behavior
- **Separation of concerns**: Engine orchestrates steps, processors implement behavior
- **Metadata for services**: Processors receive repos/services via metadata dict

## Components

### 1. BlockProcessor Interface

**File:** `block_processor.py`

Abstract base class that block implementations must extend:

```python
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem

class MyBlockProcessor(BlockProcessor):
    def can_process(self, block: Block) -> bool:
        return block.type == "MyBlock"
    
    def get_block_type(self) -> str:
        return "MyBlock"
    
    # STEP 1: Override to customize clearing (default deletes all owned items)
    def step_clear_local_data(self, block: Block, metadata=None) -> None:
        # Example: Preserve specific items
        data_item_repo = metadata.get("data_item_repo") if metadata else None
        if data_item_repo:
            for item in data_item_repo.list_by_block(block.id):
                if not item.metadata.get("preserve"):
                    data_item_repo.delete(item.id)
    
    # STEP 3: Override to validate/prepare before processing
    def step_pre_process(self, block: Block, metadata=None) -> None:
        pass  # Optional
    
    # STEP 4: REQUIRED - Main processing logic
    def process(self, block: Block, inputs: Dict[str, DataItem], metadata=None) -> Dict[str, DataItem]:
        # Transform inputs to outputs
        return {}
    
    # STEP 5: Override to update UI state after processing
    def step_post_process(self, block: Block, outputs: Dict[str, DataItem], metadata=None) -> Dict[str, DataItem]:
        # Set execution flags for UI
        block.metadata["_execution_triggered"] = True
        return outputs
```

### Available Metadata

Processors receive these services/repos via the metadata dict:

| Key | Type | Description |
|-----|------|-------------|
| `data_item_repo` | DataItemRepository | CRUD operations for data items |
| `ui_state_service` | UIStateServiceAdapter | Get/set UI state (layers, etc.) |
| `execution_mode` | str | 'executable' or 'live' |
| `project_id` | str | Current project identifier |
| `progress_tracker` | ProgressTracker | For reporting execution progress |

### Step Method Reference

| Method | Default Behavior | When to Override |
|--------|-----------------|------------------|
| `step_clear_local_data()` | Deletes ALL owned data items | Preserve specific items, clear UI state |
| `step_pre_process()` | No-op | Validate inputs, load cached resources |
| `process()` | ABSTRACT - must implement | Always implement this |
| `step_post_process()` | Returns outputs unchanged | Update UI state, set execution flags |

### 2. Topological Sort

**File:** `topological_sort.py`

Determines execution order for blocks:

- **Function:** `topological_sort_blocks(blocks, connections)`
- **Algorithm:** Kahn's algorithm
- **Features:**
  - Handles dependencies between blocks
  - Detects circular dependencies
  - Validates port compatibility

**Usage:**
```python
from src.application.processing.topological_sort import topological_sort_blocks

# Sort blocks for execution
execution_order = topological_sort_blocks(blocks, connections)

# Validate graph
is_valid, error = validate_block_graph(blocks, connections)
```

### 3. Block Execution Engine

**File:** `execution_engine.py`

Main engine for executing block graphs:

- **Class:** `BlockExecutionEngine`
- **Features:**
  - Topological sorting
  - Data flow management
  - Error handling
  - Progress reporting
  - Data item persistence

**Usage:**
```python
from src.application.processing.execution_engine import BlockExecutionEngine

# Initialize engine
engine = BlockExecutionEngine(block_repo, connection_repo, data_item_repo, event_bus)

# Register processors
engine.register_processor(LoadAudioBlockProcessor())

# Execute project
result = engine.execute_project(project_id)
```

### 4. Processor Registration

**Note:** ProcessorRegistry was removed as dead code. Processors are registered directly with BlockExecutionEngine via `register_processor()` method.

## Data Flow

1. **Input Gathering:**
   - For each block, gather inputs from connected source blocks
   - Map connections: `(target_block_id, target_input_name) -> (source_block_id, source_output_name)`
   - Retrieve data from source block outputs

2. **Block Execution:**
   - Get processor for block type
   - Call `processor.process(block, inputs, metadata)`
   - Processor returns outputs: `Dict[str, DataItem]`

3. **Output Storage:**
   - Store output DataItems in repository
   - Make outputs available for downstream blocks
   - Continue to next block in execution order

## Error Handling

- **Validation:** Graph validated before execution
  - Port compatibility checked
  - Circular dependencies detected
  - Missing processors identified

- **Execution Errors:**
  - `ProcessingError` raised by processors
  - Partial execution supported (failed blocks don't stop others)
  - Errors reported in `ExecutionResult`

## Execution Order

Blocks execute in topological order:

1. Blocks with no dependencies execute first
2. After a block executes, dependent blocks become available
3. Process continues until all blocks executed

**Example:**
```
LoadAudio -> DetectOnsets -> DrumClassify
```

Execution order: `LoadAudio`, `DetectOnsets`, `DrumClassify`

## Progress Reporting

Execution engine supports progress callbacks:

```python
def progress_callback(block_id: str, current: int, total: int):
    print(f"Executing block {current}/{total}: {block_id}")

result = engine.execute_project(project_id, progress_callback=progress_callback)
```

## Example: Creating a Block Processor

```python
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.domain.entities.block import Block
from src.domain.entities.data_item import DataItem
from src.domain.entities.audio_data_item import AudioDataItem

class LoadAudioBlockProcessor(BlockProcessor):
    def can_process(self, block: Block) -> bool:
        return block.type == "LoadAudio"
    
    def get_block_type(self) -> str:
        return "LoadAudio"
    
    def process(self, block: Block, inputs: Dict[str, DataItem], metadata=None) -> Dict[str, DataItem]:
        # Get file path from block metadata
        file_path = block.metadata.get("file_path")
        if not file_path:
            raise ProcessingError("No file path specified", block_id=block.id, block_name=block.name)
        
        # Load audio
        audio_item = AudioDataItem(
            block_id=block.id,
            name=f"{block.name}_output"
        )
        audio_item.load_audio(file_path)
        
        # Return outputs
        return {
            "audio": audio_item
        }
```

## Integration

The execution engine is integrated into:
- `ServiceContainer` in `bootstrap.py`
- `ExecutionAPI` in `application_api.py`
- `ExecutionService` for high-level operations

## Testing

Example test for execution:

```python
# Register processor
engine.register_processor(LoadAudioBlockProcessor())

# Execute project
result = engine.execute_project(project_id)

# Check results
assert result.success
assert len(result.executed_blocks) > 0
assert block_id in result.output_data
```

