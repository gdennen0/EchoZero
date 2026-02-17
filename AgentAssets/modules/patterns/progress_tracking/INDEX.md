# Progress Tracking Pattern Module

## Purpose

Provides a standardized, "built-in" pattern for progress tracking in EchoZero. Uses context managers to make progress tracking automatic and structural rather than something you remember to add.

## When to Use

- When implementing operations that take time (processing, execution)
- When users need visibility into what's happening
- When debugging requires timing and state information
- When adding progress tracking to new or existing code

## Quick Start

```python
# Both import paths work (services re-exports from shared)
from src.application.services import get_progress_context
# or: from src.shared.application.services.progress_context import get_progress_context

progress = get_progress_context()

# Simple operation tracking
with progress.operation("my_operation", "My Operation Name") as op:
    op.set_total(10)  # 10 items to process
    
    for item in items:
        with op.level("item", item.id, item.name) as item_ctx:
            item_ctx.update(message="Processing...")
            # ... do work ...
            # Automatically completed when exiting
            # Automatically failed if exception raised
```

## Two Progress Systems

EchoZero has two complementary progress systems:

1. **Simple (ProgressTracker)** - For block-level subprocess progress events
   - Location: `src/features/execution/application/progress_tracker.py`
   - Helpers: `src/features/execution/application/progress_helpers.py`
   - Publishes `SubprocessProgress` events
   - Used within block processors via metadata

2. **Advanced (ProgressContext)** - For hierarchical operations (setlist processing)
   - Location: `src/shared/application/services/progress_context.py`
   - Store: `src/shared/application/services/progress_store.py`
   - Context manager API with automatic completion/failure
   - Nested levels (operation -> song -> action -> block -> subprocess)

## Contents

- **INDEX.md** - This file (overview and quick start)
- **GUIDE.md** - Detailed usage guide with examples
- **PRESET.md** - Implementation checklist and templates
- **API.md** - Full API reference

## Related Modules

- [`modules/commands/feature/`](../../commands/feature/) - Feature development workflow
- [`modules/patterns/block_implementation/`](../block_implementation/) - Block processors (often use progress)
- [`modules/process/council/`](../../process/council/) - For major changes requiring review

## Documentation Links

- `docs/progress_tracking_guide.md` - Progress tracking usage guide
- `docs/PROGRESS_TRACKING_IMPLEMENTATION_SUMMARY.md` - Implementation summary

## Core Values Alignment

This pattern embodies "the best part is no part" and "simplicity and refinement" by:

- **Structural Enforcement**: Using context managers makes progress tracking automatic - you can't forget it
- **Less Boilerplate**: Clean `with` statements vs. scattered event emissions
- **Exception Safety**: Failures automatically recorded without try/finally blocks
- **Hierarchical by Design**: Nesting is natural and automatic

## Key Components

1. **ProgressContext** (`src/shared/application/services/progress_context.py`) - Main entry point for context manager API
2. **ProgressEventStore** (`src/shared/application/services/progress_store.py`) - Centralized state management backend
3. **ProgressState/ProgressLevel** (`src/shared/application/services/progress_models.py`) - Data models for hierarchical progress
4. **ProgressTracker** (`src/features/execution/application/progress_tracker.py`) - Simple block-level progress
5. **Progress Helpers** (`src/features/execution/application/progress_helpers.py`) - Utility functions (IncrementalProgress, track_progress, etc.)

## Architecture

```
User Code
    |
    v
ProgressContext (context manager API)
    |
    v
ProgressEventStore (state management)
    |
    +---> Callbacks (for UI updates)
    |
    v
Query API (get state from anywhere)

Block Processors
    |
    v
ProgressTracker (simple progress)
    |
    v
SubprocessProgress Events (for UI)
```

