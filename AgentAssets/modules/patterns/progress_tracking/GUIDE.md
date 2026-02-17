# Progress Tracking Pattern Guide

## Overview

The progress tracking pattern provides a "built-in" way to track progress using Python context managers. This ensures:

- Progress is automatically tracked when entering/exiting contexts
- Exceptions automatically mark operations as failed
- Nesting is natural and hierarchical
- Less boilerplate than manual event emission

## Basic Usage

### 1. Get a Progress Context

```python
from src.application.services import get_progress_context

progress = get_progress_context()
```

### 2. Create an Operation

```python
with progress.operation("operation_type", "Operation Name") as op:
    # ... your code here ...
    pass
# Automatically completed when exiting
```

### 3. Track Nested Levels

```python
with progress.operation("setlist_processing", "Process All Songs") as op:
    for song in songs:
        with op.level("song", song.id, song.name) as song_ctx:
            for action in actions:
                with song_ctx.level("action", str(idx), action.name) as action_ctx:
                    action_ctx.update(message="Executing...")
                    # ... execute action ...
```

## Complete Example: Setlist Processing

```python
from src.application.services import get_progress_context

def process_setlist(self, setlist_id: str):
    progress = get_progress_context()
    
    with progress.setlist_processing(setlist_id, f"Processing {setlist.name}") as op:
        op.set_total(len(songs))
        
        for idx, song in enumerate(songs):
            with op.song(song.id, song.name) as song_ctx:
                song_ctx.set_total(len(actions))
                song_ctx.update(message=f"Starting ({idx + 1}/{len(songs)})")
                
                for action_idx, action in enumerate(actions):
                    with song_ctx.action(str(action_idx), action.name) as action_ctx:
                        action_ctx.update(message="Executing...")
                        
                        # Execute the action
                        result = self._execute_action(action)
                        
                        # Update with result
                        action_ctx.update(
                            message="Completed",
                            result=result  # metadata
                        )
                
                song_ctx.update(message="Completed")
        
        # Operation automatically completed when exiting
```

## Updating Progress

### Set Total Items

```python
with op.level("song", song.id, song.name) as song_ctx:
    song_ctx.set_total(5)  # 5 actions for this song
```

### Update Current Progress

```python
# Set current value directly
song_ctx.update(current=3, message="Processing item 3/5")

# Or increment
song_ctx.increment()  # current += 1
song_ctx.increment(message="Moved to next item")
```

### Add Metadata

```python
song_ctx.update(
    message="Processing audio",
    sample_rate=44100,
    channels=2,
    duration_seconds=180.5
)
```

## Convenience Methods

### For Setlist Processing

```python
with progress.setlist_processing(setlist_id, "Setlist Name") as op:
    with op.song(song_id, song_name) as song_ctx:
        with song_ctx.action(action_id, action_name) as action_ctx:
            pass
```

### For Block Execution

```python
with progress.block_execution(block_id, block_name, block_type="LoadAudio") as op:
    with op.subprocess("load", "Loading audio file") as sub:
        sub.update(current=50, total=100, message="50% loaded")
```

## Exception Handling

Exceptions are automatically handled:

```python
with progress.operation("my_op", "My Operation") as op:
    with op.level("item", "item_1", "Item 1") as ctx:
        raise ValueError("Something went wrong")
        # Level automatically marked as FAILED with error message

# Operation automatically marked as FAILED
```

## Querying Progress State

### Get Active Operations

```python
from src.application.services import get_progress_context

progress = get_progress_context()
active = progress.get_active_operations()

for op in active:
    print(f"{op.name}: {op.status.value}")
```

### Get Specific Operation

```python
state = progress.get_operation(operation_id)
if state:
    overall = state.get_overall_progress()
    print(f"Progress: {overall['percentage']:.1f}%")
    print(f"Completed: {overall['completed']}/{overall['total']}")
```

### Get Historical Operations

```python
history = progress.get_history(operation_type="setlist_processing", limit=10)
for op in history:
    elapsed = op.get_elapsed_seconds()
    print(f"{op.name}: {op.status.value} ({elapsed:.1f}s)")
```

## Listening for Updates (UI)

```python
from src.application.services import get_progress_store

store = get_progress_store()

def on_progress(event_type: str, state: ProgressState):
    if event_type == "updated":
        # Update UI with new state
        print(f"Progress: {state.get_overall_progress()['percentage']:.1f}%")
    elif event_type == "completed":
        print(f"Completed: {state.name}")

store.add_callback(on_progress)

# Later, remove callback
store.remove_callback(on_progress)
```

## Best Practices

### 1. Use Meaningful Names

```python
# Good
with op.song(song.id, Path(song.audio_path).name) as ctx:
    pass

# Less helpful
with op.level("item", "x", "thing") as ctx:
    pass
```

### 2. Update Regularly

```python
for i, chunk in enumerate(chunks):
    ctx.update(current=i, message=f"Processing chunk {i + 1}/{len(chunks)}")
    process_chunk(chunk)
```

### 3. Include Useful Metadata

```python
action_ctx.update(
    message="Completed",
    items_processed=127,
    duration_ms=1250,
    output_file=str(output_path)
)
```

### 4. Use Specific Level Types

```python
# Use the built-in types
op.song(...)       # for songs
song_ctx.action(...) # for actions
action_ctx.block(...) # for blocks
block_ctx.subprocess(...) # for subprocesses

# Or create custom types
op.level("custom_type", id, name)
```

## Integration with Existing Code

### Adding to SetlistService

```python
# In SetlistService.__init__
from src.application.services import get_progress_context
self._progress = get_progress_context()

# In process_setlist method
def process_setlist(self, setlist_id: str, ...):
    with self._progress.setlist_processing(setlist_id, setlist.name) as op:
        # ... existing code wrapped in context managers ...
```

### Adding to BlockProcessor (Simple Progress)

For simple block-level progress, use `ProgressTracker` from metadata:

```python
from src.features.execution.application.progress_helpers import get_progress_tracker

def process(self, block: Block, inputs: Dict[str, DataItem], metadata: Optional[Dict] = None):
    tracker = get_progress_tracker(metadata)
    if tracker:
        tracker.start("Loading data")
    
    # ... load data ...
    
    if tracker:
        tracker.update(50, 100, "Processing audio")
    
    # ... process ...
    
    if tracker:
        tracker.complete()
    
    return outputs
```

### Progress Helpers (Utility Functions)

The `progress_helpers` module (`src/features/execution/application/progress_helpers.py`) provides utility functions:

```python
from src.features.execution.application.progress_helpers import (
    IncrementalProgress,  # Manual step-by-step tracking
    progress_scope,       # Context manager for scoped operations
    yield_progress,       # Report progress at specific points
    track_progress,       # Automatic progress for iterables
    get_progress_tracker, # Extract tracker from metadata
)

# Track progress over an iterable
def process(self, block, inputs, metadata=None):
    tracker = get_progress_tracker(metadata)
    items = list(inputs.values())
    
    for item in track_progress(items, tracker, "Processing items"):
        # ... process each item ...
        pass
```

### Adding to BlockProcessor (Advanced Progress)

For hierarchical operations within a block:

```python
from src.application.services import get_progress_context

def process(self, block: Block, inputs: Dict[str, DataItem], metadata: Optional[Dict] = None):
    progress = get_progress_context()
    
    with progress.block_execution(block.id, block.name, block.type) as op:
        # ... existing processing code ...
```

## Error Information

When an exception occurs, detailed error information is captured:

```python
try:
    with op.level("item", "1", "Item 1") as ctx:
        raise ValueError("Invalid input")
except ValueError:
    pass

# The level now has:
# - status: FAILED
# - error: "Invalid input"
# - error_details: {"exception_type": "ValueError"}
# - completed_at: (timestamp)
```

## Performance Considerations

- Progress updates are lightweight (in-memory store)
- Callbacks are called synchronously - keep them fast
- For high-frequency updates, consider throttling in the UI
- Historical data is limited (default: 100 operations)

