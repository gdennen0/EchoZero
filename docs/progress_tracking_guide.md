# Progress Tracking Guide for Block Developers

## Overview

EchoZero has a standardized progress tracking system that allows blocks to report progress during long-running operations. This guide shows you how to use it.

## The Problem We Solved

Previously, progress bars would show 0%, then jump directly to 100% after the operation completed. This was because blocks weren't reporting intermediate progress during long operations.

## The Solution: Progress Helpers

We created simple, reusable utilities in `src/features/execution/application/progress_helpers.py` that make progress tracking effortless.

## Quick Start Examples

### Example 1: Simple Operation with Steps

Use `progress_scope` and `yield_progress` for operations with defined steps:

```python
from src.features.execution.application.progress_helpers import (
    progress_scope, yield_progress, get_progress_tracker
)

def process(self, block: Block, inputs: Dict, metadata: Optional[Dict] = None) -> Dict:
    progress_tracker = get_progress_tracker(metadata)
    
    # Use progress scope for automatic start/complete
    with progress_scope(progress_tracker, "Loading audio", total=3):
        yield_progress(progress_tracker, 1, "Reading file...")
        data = read_file(path)
        
        yield_progress(progress_tracker, 2, "Decoding audio...")
        audio = decode(data)
        
        yield_progress(progress_tracker, 3, "Creating data item...")
        item = create_item(audio)
    
    return {"audio": item}
```

### Example 2: Processing a List/Batch

Use `track_progress` for automatic progress when iterating over a list:

```python
from src.features.execution.application.progress_helpers import (
    track_progress, get_progress_tracker
)

def process(self, block: Block, inputs: Dict, metadata: Optional[Dict] = None) -> Dict:
    progress_tracker = get_progress_tracker(metadata)
    
    files = ["song1.wav", "song2.wav", "song3.wav"]
    
    # Progress updates automatically for each item!
    for filepath in track_progress(files, progress_tracker, "Exporting audio"):
        export_file(filepath)
    
    return {}
```

### Example 3: Manual Progress Control

Use `IncrementalProgress` when you need manual control:

```python
from src.features.execution.application.progress_helpers import (
    IncrementalProgress, get_progress_tracker
)

def process(self, block: Block, inputs: Dict, metadata: Optional[Dict] = None) -> Dict:
    progress_tracker = get_progress_tracker(metadata)
    
    # Create progress tracker for 100 epochs
    progress = IncrementalProgress(progress_tracker, "Training model", total=100)
    
    for epoch in range(100):
        train_epoch()
        progress.step(f"Epoch {epoch+1}/100 complete")
    
    progress.complete("Training complete")
    return {}
```

## Available Utilities

### `get_progress_tracker(metadata)`
Safely extract progress tracker from metadata. Returns `None` if not available.

```python
progress_tracker = get_progress_tracker(metadata)
```

### `progress_scope(tracker, message, total=None)`
Context manager that automatically calls `start()` and `complete()`.

```python
with progress_scope(tracker, "Loading data", total=5):
    # Do work...
    yield_progress(tracker, 1, "Step 1...")
    yield_progress(tracker, 2, "Step 2...")
```

### `yield_progress(tracker, current, message=None)`
Report progress at a specific point. Use inside `progress_scope()`.

```python
yield_progress(tracker, 3, "Step 3 of 5")
```

### `track_progress(items, tracker, message, total=None)`
Wrap an iterable to automatically report progress for each item.

```python
for item in track_progress(my_list, tracker, "Processing"):
    process_item(item)
```

### `IncrementalProgress(tracker, message, total, start_at=0)`
Manual progress control with `step()`, `set()`, and `complete()` methods.

```python
progress = IncrementalProgress(tracker, "Working", total=100)
progress.step("Step 1 done")  # Auto-increments
progress.set(50, "Halfway done")  # Set to specific value
progress.complete("All done")
```

### `BatchProgress(tracker, items, message, total=None)`
Alternative to `track_progress` using class-based approach.

```python
batch = BatchProgress(tracker, items, "Processing files")
for item in batch:
    process_item(item)  # Progress updates automatically
```

## Best Practices

### 1. Always Get Progress Tracker Safely

```python
# DO THIS
progress_tracker = get_progress_tracker(metadata)

# NOT THIS
progress_tracker = metadata.get("progress_tracker") if metadata else None
```

### 2. Progress Helpers Handle None Gracefully

All helpers check if `progress_tracker` is `None` and no-op safely. You don't need to check:

```python
# This is fine even if tracker is None
with progress_scope(progress_tracker, "Loading", total=3):
    yield_progress(progress_tracker, 1, "Step 1")
```

### 3. Use the Right Helper for the Job

- **Fixed steps?** → `progress_scope` + `yield_progress`
- **Iterating a list?** → `track_progress`
- **Training/epochs?** → `IncrementalProgress`
- **Complex custom logic?** → `IncrementalProgress` with manual `set()`

### 4. Report Progress at Meaningful Points

```python
# GOOD - Reports at each significant step
with progress_scope(tracker, "Processing audio", total=4):
    yield_progress(tracker, 1, "Loading audio...")
    audio = load_audio(path)
    
    yield_progress(tracker, 2, "Applying effects...")
    audio = apply_effects(audio)
    
    yield_progress(tracker, 3, "Generating waveform...")
    waveform = generate_waveform(audio)
    
    yield_progress(tracker, 4, "Saving results...")
    save_results(audio, waveform)

# BAD - Only reports start/end (0% → 100% jump)
with progress_scope(tracker, "Processing audio", total=None):
    audio = load_audio(path)
    audio = apply_effects(audio)
    waveform = generate_waveform(audio)
    save_results(audio, waveform)
```

### 5. Give Descriptive Messages

```python
# GOOD - User knows what's happening
yield_progress(tracker, 2, "Decoding MP3 file...")

# BAD - Too vague
yield_progress(tracker, 2, "Processing...")
```

## Integration Checklist

When adding progress to a block processor:

- [ ] Import progress helpers at the top of the file
- [ ] Get progress tracker with `get_progress_tracker(metadata)`
- [ ] Choose the right helper for your use case
- [ ] Add meaningful progress messages
- [ ] Test that progress actually updates during operation
- [ ] Verify no errors when progress_tracker is None

## Examples in the Codebase

See these files for working examples:

- `src/application/blocks/load_audio_block.py` - Uses `progress_scope`
- `src/application/blocks/export_audio_block.py` - Uses `track_progress`
- `src/application/blocks/export_clips_by_class_block.py` - Uses `IncrementalProgress`
- `src/application/blocks/separator_block.py` - Manual progress tracking (pre-helpers)

## Technical Details

### How It Works

1. `ApplicationFacade.execute_block()` creates a `ProgressTracker` instance
2. Progress tracker is passed in `metadata['progress_tracker']`
3. Progress tracker publishes `SubprocessProgress` events to the event bus
4. `MainWindow` subscribes to these events and updates the UI progress bar
5. Progress bar updates show in the status bar at the bottom of the window

### Progress Tracker API (Low-Level)

If you need direct access (not recommended - use helpers instead):

```python
tracker.start(message, total=None, current=0)
tracker.update(current=None, total=None, message=None, increment=0)
tracker.complete(message=None)
```

### Testing Progress

To test your progress implementation:

1. Run your block with a long operation
2. Watch the status bar at the bottom of the window
3. Progress should update smoothly, not jump from 0% to 100%
4. Check console logs for `ProgressTracker: Publishing SubprocessProgress` messages

## FAQ

**Q: What if progress_tracker is None?**  
A: All helpers handle None gracefully and become no-ops. No need to check.

**Q: Can I use progress tracking outside of block processors?**  
A: Yes! Create a `ProgressTracker` manually with `create_progress_tracker(block, project_id, event_bus)`.

**Q: What if I don't know the total number of steps?**  
A: Set `total=None` for indeterminate progress. The UI will show activity but no percentage.

**Q: Can I nest progress tracking?**  
A: Not currently. Each block has one progress tracker. For sub-operations, just update the message.

**Q: How often should I report progress?**  
A: Every few seconds is good. Too frequent (every millisecond) can slow things down. Too rare defeats the purpose.

## Contributing

When you add progress tracking to a block, please:

1. Follow the patterns in this guide
2. Test that progress actually works
3. Update this guide if you discover better patterns
4. Consider adding examples to `progress_helpers.py` docstrings
