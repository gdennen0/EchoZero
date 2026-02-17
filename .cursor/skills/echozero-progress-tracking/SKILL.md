---
name: echozero-progress-tracking
description: Track progress in EchoZero using ProgressContext or ProgressTracker. Use when adding progress to block processors, setlist processing, long-running operations, or when the user asks about progress bars, progress tracking, or hierarchical progress.
---

# Progress Tracking

## Two Systems

1. **Simple (Block-level):** `ProgressTracker` from metadata - for block processors
2. **Advanced (Hierarchical):** `ProgressContext` - for setlist processing, multi-level ops

## Block Processor (Simple)

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

## Hierarchical (Context Managers)

```python
from src.application.services import get_progress_context

progress = get_progress_context()

with progress.operation("my_op", "My Operation") as op:
    op.set_total(len(items))
    for item in items:
        with op.level("item", item.id, item.name) as ctx:
            ctx.update(message="Processing...")
            ctx.update(current=3, total=5)
            ctx.increment()
```

## Setlist Processing Convenience

```python
with progress.setlist_processing(setlist_id, "Setlist Name") as op:
    with op.song(song_id, song_name) as song_ctx:
        with song_ctx.action(action_id, action_name) as action_ctx:
            action_ctx.update(message="Executing...")
```

## Block Execution Context

```python
with progress.block_execution(block_id, block_name, block_type="LoadAudio") as op:
    with op.subprocess("load", "Loading audio") as sub:
        sub.update(current=50, total=100, message="50%")
```

## Key Behaviors

- Exceptions automatically mark operations as FAILED
- Exiting context automatically completes
- Use `set_total()` for item counts
- Use `update(current=, total=, message=)` or `increment()`
- Add metadata: `ctx.update(message="Done", items_processed=127)`

## Querying State

```python
progress = get_progress_context()
active = progress.get_active_operations()
state = progress.get_operation(operation_id)
history = progress.get_history(operation_type="setlist_processing", limit=10)
```

## Best Practices

- Use meaningful names (song.id, Path(song.audio_path).name)
- Update regularly during loops
- Include useful metadata on completion
- Use built-in level types: `op.song()`, `song_ctx.action()`, `action_ctx.block()`

## Reference

- Full guide: `AgentAssets/modules/patterns/progress_tracking/GUIDE.md`
- API: `AgentAssets/modules/patterns/progress_tracking/API.md`
- Helpers: `src/features/execution/application/progress_helpers.py`
