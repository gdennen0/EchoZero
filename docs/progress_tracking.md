# Progress Tracking

Standard guide for reporting progress in EchoZero blocks. Use the helpers in `src/features/execution/application/progress_helpers.py` so the status bar shows incremental progress instead of 0% then 100%.

## Problem and solution

**Problem:** Progress bars jumped from 0% to 100% because blocks only reported start and complete.

**Solution:** Helpers that report progress at steps or per item. All helpers no-op when `progress_tracker` is `None`.

## Helpers (choose one)

| Helper | Use when |
|--------|----------|
| `get_progress_tracker(metadata)` | Getting the tracker (use in every block that reports progress). |
| `progress_scope(tracker, message, total=N)` | Fixed sequence of steps. |
| `yield_progress(tracker, current, message)` | Report step N inside `progress_scope`. |
| `track_progress(items, tracker, message)` | Iterating a list (one step per item). |
| `IncrementalProgress(tracker, message, total)` | Manual steps (e.g. epochs, custom counts). |
| `BatchProgress(tracker, items, message)` | Class-based alternative to `track_progress`. |

## Patterns

### Fixed steps

```python
from src.features.execution.application.progress_helpers import (
    progress_scope, yield_progress, get_progress_tracker
)

def process(self, block, inputs, metadata=None):
    progress_tracker = get_progress_tracker(metadata)
    with progress_scope(progress_tracker, "Loading audio", total=3):
        yield_progress(progress_tracker, 1, "Reading file...")
        data = load_file()
        yield_progress(progress_tracker, 2, "Decoding...")
        audio = decode(data)
        yield_progress(progress_tracker, 3, "Creating item...")
        return create_item(audio)
```

**Example:** `load_audio_block.py`

### List / batch

```python
from src.features.execution.application.progress_helpers import track_progress, get_progress_tracker

def process(self, block, inputs, metadata=None):
    progress_tracker = get_progress_tracker(metadata)
    for item in track_progress(items, progress_tracker, "Exporting audio"):
        export_item(item)
```

**Examples:** `export_audio_block.py`, `detect_onsets_block.py`

### Epochs / manual steps

```python
from src.features.execution.application.progress_helpers import IncrementalProgress, get_progress_tracker

def process(self, block, inputs, metadata=None):
    progress_tracker = get_progress_tracker(metadata)
    progress = IncrementalProgress(progress_tracker, "Training model", total=epochs)
    for epoch in range(epochs):
        train_epoch()
        progress.step(f"Epoch {epoch+1}/{epochs} - Acc: {acc:.1f}%")
    progress.complete("Training complete")
```

**Examples:** `pytorch_audio_trainer_block.py`, `pytorch_drum_trainer_block.py`, `export_clips_by_class_block.py`

## Best practices

1. **Always use `get_progress_tracker(metadata)`** – do not read `metadata["progress_tracker"]` directly.
2. **Pick the right helper:** fixed steps → `progress_scope` + `yield_progress`; list → `track_progress`; epochs/custom → `IncrementalProgress`.
3. **Report at meaningful points** – avoid a single scope with no intermediate `yield_progress` (that gives 0% → 100%).
4. **Use clear messages** – e.g. "Decoding MP3 file..." not "Processing...".

## How it works

```
Block process(metadata) → get_progress_tracker(metadata) → helpers
  → ProgressTracker → SubprocessProgress (event bus) → MainWindow → status bar
```

Low-level API (prefer helpers): `tracker.start(message, total)`, `tracker.update(current=..., message=...)`, `tracker.complete(message)`.

## Testing

Run the block and watch the status bar. Progress should move incrementally (e.g. 33% → 66% → 100%), not jump from 0% to 100%. Examples: LoadAudio (3 steps), ExportAudio (per file), PyTorch Audio Trainer (per epoch).

## FAQ

- **progress_tracker is None?** Helpers no-op; no need to check.
- **Unknown total?** Use `total=None` for indeterminate progress (no percentage).
- **Use outside block processors?** Create a `ProgressTracker` via `create_progress_tracker(block, project_id, event_bus)`.
- **Nested progress?** Not supported; one tracker per block. Update the message for sub-steps.
- **How often to report?** Every few seconds is enough; avoid per-millisecond updates.

## Reference in codebase

- **Helpers:** `src/features/execution/application/progress_helpers.py`
- **Examples:** `load_audio_block.py`, `export_audio_block.py`, `export_clips_by_class_block.py`, `pytorch_audio_trainer_block.py`, `pytorch_drum_trainer_block.py`, `detect_onsets_block.py`
