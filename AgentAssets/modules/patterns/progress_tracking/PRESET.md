# Progress Tracking Pattern Preset

## Quick Implementation Checklist

Use this checklist when adding progress tracking to new or existing code.

### Before You Start

- [ ] Identify the operation type (setlist_processing, block_execution, etc.)
- [ ] Identify the levels (e.g., songs -> actions -> blocks)
- [ ] Determine what metadata to track (timing, counts, results)

### Implementation Steps

- [ ] Import progress context
- [ ] Create operation context
- [ ] Add level contexts for each nesting level
- [ ] Call update() with meaningful messages
- [ ] Include relevant metadata
- [ ] Test with successful case
- [ ] Test with error case

---

## Code Templates

### Basic Operation (Advanced Progress)

```python
from src.application.services import get_progress_context

def my_operation():
    progress = get_progress_context()
    
    with progress.operation("operation_type", "Operation Name") as op:
        op.set_total(len(items))
        
        for item in items:
            with op.level("item", item.id, item.name) as ctx:
                ctx.update(message="Processing...")
                # ... process item ...
                ctx.update(message="Complete")
```

### Block Processor (Simple Progress)

```python
from src.features.execution.application.progress_helpers import get_progress_tracker, track_progress

def process(self, block, inputs, metadata=None):
    tracker = get_progress_tracker(metadata)
    
    # Option 1: Manual progress
    if tracker:
        tracker.start("Loading data")
    data = self._load(inputs)
    if tracker:
        tracker.update(50, 100, "Processing")
    result = self._process(data)
    if tracker:
        tracker.complete()
    
    # Option 2: Automatic progress over iterable
    items = list(inputs.values())
    for item in track_progress(items, tracker, "Processing items"):
        self._process_item(item)
    
    return {"output": result}
```

### Setlist Processing

```python
from src.application.services import get_progress_context
from pathlib import Path

def process_setlist(self, setlist_id: str):
    progress = get_progress_context()
    
    with progress.setlist_processing(setlist_id, setlist.name) as op:
        op.set_total(len(songs))
        
        for song in songs:
            with op.song(song.id, Path(song.audio_path).name) as song_ctx:
                song_ctx.set_total(len(actions))
                
                for idx, action in enumerate(actions):
                    with song_ctx.action(str(idx), action.name) as action_ctx:
                        action_ctx.update(message=f"Executing...")
                        self._execute_action(action)
```

### Block Execution (Advanced)

```python
from src.application.services import get_progress_context

def execute_block(self, block: Block):
    progress = get_progress_context()
    
    with progress.block_execution(block.id, block.name, block.type) as op:
        with op.subprocess("load", "Loading data") as load_ctx:
            load_ctx.update(message="Loading inputs...")
            inputs = self._load_inputs(block)
            load_ctx.update(message="Loaded")
        
        with op.subprocess("process", "Processing") as proc_ctx:
            proc_ctx.set_total(len(inputs))
            for idx, item in enumerate(inputs):
                proc_ctx.update(current=idx, message=f"Processing {idx + 1}/{len(inputs)}")
                self._process_item(item)
```

### With Error Handling

```python
from src.application.services import get_progress_context

def process_with_recovery(self, items):
    progress = get_progress_context()
    results = {}
    
    with progress.operation("batch_process", "Batch Processing") as op:
        op.set_total(len(items))
        
        for item in items:
            try:
                with op.level("item", item.id, item.name) as ctx:
                    ctx.update(message="Processing...")
                    result = self._process(item)
                    ctx.update(message="Complete", result=result)
                    results[item.id] = True
            except Exception as e:
                # Level is automatically marked failed
                # Continue with next item
                results[item.id] = False
                continue
    
    return results
```

---

## Common Patterns

### Progress with Percentage

```python
with op.level("download", "file", "Downloading file") as ctx:
    ctx.set_total(100)  # Percentage
    
    for percent in download_with_progress(url):
        ctx.update(current=percent, message=f"Downloading... {percent}%")
```

### Nested Actions with Block Details

```python
with song_ctx.action(str(idx), action.name) as action_ctx:
    action_ctx.update(
        message="Starting",
        block_name=action.block_name,
        block_type=action.block_type
    )
    
    # Execute
    result = self._execute_action(action)
    
    action_ctx.update(
        message="Completed",
        items_processed=result.count,
        duration_ms=result.duration_ms
    )
```

### UI Callback Integration

```python
from src.application.services import get_progress_store, ProgressState

class ProgressDialog:
    def __init__(self):
        self._store = get_progress_store()
        self._store.add_callback(self._on_progress)
    
    def _on_progress(self, event_type: str, state: ProgressState):
        if event_type == "updated":
            overall = state.get_overall_progress()
            self.progress_bar.setValue(int(overall['percentage']))
            self.status_label.setText(state.name)
        elif event_type == "completed":
            self.close()
    
    def cleanup(self):
        self._store.remove_callback(self._on_progress)
```

---

## Validation Checklist

After implementing, verify:

- [ ] Operation starts and completes correctly
- [ ] All levels show proper nesting
- [ ] Errors are captured with useful messages
- [ ] Timing is recorded accurately
- [ ] Metadata is useful for debugging
- [ ] UI updates reflect progress correctly
- [ ] Performance is acceptable (no UI freezing)

---

## Anti-Patterns to Avoid

### Don't Forget to Update

```python
# Bad - no updates
with op.level("item", id, name) as ctx:
    long_running_operation()

# Good - regular updates
with op.level("item", id, name) as ctx:
    for step in steps:
        ctx.update(message=f"Step: {step}")
        process_step(step)
```

### Don't Swallow Exceptions Silently

```python
# Bad - exception swallowed, progress shows success
with op.level("item", id, name) as ctx:
    try:
        risky_operation()
    except:
        pass  # Level shows completed!

# Good - let exception propagate or re-raise
with op.level("item", id, name) as ctx:
    try:
        risky_operation()
    except Exception as e:
        # Level will be marked failed
        raise
```

### Don't Create Progress Context in Loops

```python
# Bad - creates many contexts
for item in items:
    progress = get_progress_context()  # Don't do this
    with progress.operation(...):
        pass

# Good - create once
progress = get_progress_context()
with progress.operation(...) as op:
    for item in items:
        with op.level(...):
            pass
```

