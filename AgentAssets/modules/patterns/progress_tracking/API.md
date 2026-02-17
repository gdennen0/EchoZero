# Progress Tracking API Reference

## Module Imports

### Advanced Progress (Context Managers)

```python
# Re-exported via src.application.services for convenience
from src.application.services import (
    # Main context manager
    ProgressContext,
    get_progress_context,
    
    # Store (for direct access/callbacks)
    ProgressEventStore,
    get_progress_store,
    reset_progress_store,
    
    # Models
    ProgressState,
    ProgressLevel,
    ProgressStatus,
    
    # Context types
    LevelContext,
    OperationContext,
)

# Or import directly from shared layer
from src.shared.application.services.progress_context import ProgressContext, get_progress_context
from src.shared.application.services.progress_store import ProgressEventStore, get_progress_store
from src.shared.application.services.progress_models import ProgressState, ProgressLevel, ProgressStatus
```

### Simple Progress (Block-Level)

```python
from src.features.execution.application.progress_tracker import (
    ProgressTracker,
    ProgressTrackerContext,
    create_progress_tracker,
)

from src.features.execution.application.progress_helpers import (
    IncrementalProgress,   # Manual step-by-step tracking
    progress_scope,        # Context manager for scoped operations
    yield_progress,        # Report progress at specific points
    track_progress,        # Automatic progress for iterables
    get_progress_tracker,  # Extract tracker from metadata
)
```

---

## ProgressContext

Main entry point for progress tracking.

### Constructor

```python
ProgressContext(store: Optional[ProgressEventStore] = None)
```

- `store`: Optional custom progress store (uses global singleton if not provided)

### Methods

#### operation()

```python
@contextmanager
def operation(
    operation_type: str,
    name: str = "",
    operation_id: Optional[str] = None,
    **metadata
) -> Generator[OperationContext, None, None]
```

Start a new progress-tracked operation.

- `operation_type`: Type of operation (e.g., "setlist_processing")
- `name`: Display name for the operation
- `operation_id`: Optional custom operation ID (auto-generated if not provided)
- `**metadata`: Additional metadata

#### setlist_processing()

```python
@contextmanager
def setlist_processing(
    setlist_id: str,
    name: str = "Setlist Processing",
    **metadata
) -> Generator[OperationContext, None, None]
```

Convenience method for setlist processing operations.

#### block_execution()

```python
@contextmanager
def block_execution(
    block_id: str,
    block_name: str,
    block_type: str = "",
    **metadata
) -> Generator[OperationContext, None, None]
```

Convenience method for block execution operations.

#### get_active_operations()

```python
def get_active_operations() -> List[ProgressState]
```

Get all currently active operations.

#### get_operation()

```python
def get_operation(operation_id: str) -> Optional[ProgressState]
```

Get a specific operation by ID.

#### get_history()

```python
def get_history(
    operation_type: Optional[str] = None,
    limit: int = 100
) -> List[ProgressState]
```

Get historical operations.

---

## OperationContext

Context for a top-level operation. Extends LevelContext.

### Methods

#### get_state()

```python
def get_state() -> Optional[ProgressState]
```

Get current operation state.

#### set_total()

```python
def set_total(total: int) -> None
```

Set total count for the operation.

---

## LevelContext

Context for a progress level within an operation.

### Properties

- `level_id: str` - The level ID
- `operation_id: str` - The operation ID

### Methods

#### set_total()

```python
def set_total(total: int) -> None
```

Set the total count for this level.

#### update()

```python
def update(
    current: Optional[int] = None,
    total: Optional[int] = None,
    message: Optional[str] = None,
    increment: int = 0,
    **metadata
) -> None
```

Update progress for this level.

- `current`: Set current progress to this value
- `total`: Update total (if changed)
- `message`: Update status message
- `increment`: Increment current by this amount (ignored if current is set)
- `**metadata`: Additional metadata to merge

#### increment()

```python
def increment(message: Optional[str] = None) -> None
```

Increment progress by 1.

#### level()

```python
@contextmanager
def level(
    level_type: str,
    level_id: str,
    name: str,
    total: int = 0,
    **metadata
) -> Generator['LevelContext', None, None]
```

Create a nested progress level.

#### Convenience Methods

```python
def song(song_id: str, name: str, total: int = 0, **metadata) -> Generator[LevelContext, None, None]
def action(action_id: str, name: str, total: int = 0, **metadata) -> Generator[LevelContext, None, None]
def block(block_id: str, name: str, block_type: str = "", total: int = 0, **metadata) -> Generator[LevelContext, None, None]
def subprocess(subprocess_id: str, name: str, total: int = 0, **metadata) -> Generator[LevelContext, None, None]
```

---

## ProgressEventStore

Centralized store for progress state.

### Constructor

```python
ProgressEventStore(max_history: int = 100)
```

- `max_history`: Maximum number of completed operations to keep in history

### Callback Methods

#### add_callback()

```python
def add_callback(callback: Callable[[str, ProgressState], None]) -> None
```

Add a callback for progress updates. Event types: "started", "updated", "completed", "failed", "cancelled"

#### remove_callback()

```python
def remove_callback(callback: Callable[[str, ProgressState], None]) -> None
```

Remove a progress callback.

### Operation Methods

#### start_operation()

```python
def start_operation(
    operation_type: str,
    name: str = "",
    operation_id: Optional[str] = None,
    **metadata
) -> ProgressState
```

Start tracking a new operation.

#### get_state()

```python
def get_state(operation_id: str) -> Optional[ProgressState]
```

Get current progress state for an operation.

#### get_all_active()

```python
def get_all_active() -> List[ProgressState]
```

Get all currently active operations.

#### complete_operation()

```python
def complete_operation(
    operation_id: str,
    error: Optional[str] = None
) -> Optional[ProgressState]
```

Mark an operation as complete.

#### cancel_operation()

```python
def cancel_operation(operation_id: str) -> Optional[ProgressState]
```

Cancel an operation.

### Level Methods

#### add_level()

```python
def add_level(
    operation_id: str,
    level_id: str,
    level_type: str,
    name: str,
    parent_id: Optional[str] = None,
    total: int = 0,
    **metadata
) -> Optional[ProgressLevel]
```

Add a new progress level to an operation.

#### start_level()

```python
def start_level(
    operation_id: str,
    level_id: str,
    message: Optional[str] = None
) -> Optional[ProgressLevel]
```

Mark a level as started (running).

#### update_level()

```python
def update_level(
    operation_id: str,
    level_id: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
    message: Optional[str] = None,
    increment: int = 0,
    **metadata
) -> Optional[ProgressLevel]
```

Update progress for a level.

#### complete_level()

```python
def complete_level(
    operation_id: str,
    level_id: str,
    message: Optional[str] = None
) -> Optional[ProgressLevel]
```

Mark a level as completed.

#### fail_level()

```python
def fail_level(
    operation_id: str,
    level_id: str,
    error: str,
    error_details: Optional[Dict[str, Any]] = None
) -> Optional[ProgressLevel]
```

Mark a level as failed.

### History Methods

#### get_history()

```python
def get_history(
    operation_type: Optional[str] = None,
    limit: int = 100
) -> List[ProgressState]
```

Get historical progress states.

#### clear_history()

```python
def clear_history() -> None
```

Clear all historical progress states.

---

## ProgressState

Complete progress state for a processing operation.

### Properties

- `operation_id: str` - Unique operation identifier
- `operation_type: str` - Type of operation
- `name: str` - Display name
- `status: ProgressStatus` - Current status
- `started_at: Optional[datetime]` - When operation started
- `completed_at: Optional[datetime]` - When operation completed
- `error: Optional[str]` - Error message (if failed)
- `error_details: Optional[Dict[str, Any]]` - Detailed error info
- `levels: Dict[str, ProgressLevel]` - All progress levels
- `root_level_ids: List[str]` - Top-level item IDs
- `metadata: Dict[str, Any]` - Additional metadata

### Methods

#### get_level()

```python
def get_level(level_id: str) -> Optional[ProgressLevel]
```

Get progress level by ID.

#### get_elapsed_seconds()

```python
def get_elapsed_seconds() -> Optional[float]
```

Get elapsed time in seconds.

#### get_overall_progress()

```python
def get_overall_progress() -> Dict[str, Any]
```

Get overall progress summary. Returns dict with: total, completed, failed, pending, percentage, elapsed_seconds.

#### to_dict()

```python
def to_dict() -> Dict[str, Any]
```

Convert to dictionary for serialization.

---

## ProgressLevel

Progress at a specific level.

### Properties

- `level_id: str` - Level identifier
- `level_type: str` - Type of level
- `name: str` - Display name
- `status: ProgressStatus` - Current status
- `current: int` - Current progress
- `total: int` - Total items
- `percentage: float` - Completion percentage
- `message: str` - Status message
- `started_at: Optional[datetime]` - When level started
- `completed_at: Optional[datetime]` - When level completed
- `error: Optional[str]` - Error message
- `error_details: Optional[Dict[str, Any]]` - Error details
- `parent_id: Optional[str]` - Parent level ID
- `children: Dict[str, ProgressLevel]` - Child levels
- `metadata: Dict[str, Any]` - Additional metadata

### Methods

#### update()

```python
def update(
    current: Optional[int] = None,
    total: Optional[int] = None,
    message: Optional[str] = None,
    increment: int = 0,
    **metadata
) -> None
```

Update progress level.

#### get_elapsed_seconds()

```python
def get_elapsed_seconds() -> Optional[float]
```

Get elapsed time in seconds.

#### get_duration_str()

```python
def get_duration_str() -> str
```

Get human-readable duration string (e.g., "1m 30s").

#### to_dict()

```python
def to_dict() -> Dict[str, Any]
```

Convert to dictionary for serialization.

---

## ProgressStatus

Enum for progress status values.

```python
class ProgressStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

## Event: OperationProgress

Domain event for operation progress updates.

### Data Fields

- `operation_id: str` - Unique operation identifier
- `operation_type: str` - Type of operation
- `event_type: str` - Type of update ("started", "updated", "completed", "failed")
- `state: Dict` - Full ProgressState as dict
- `level_id: str` - ID of level that changed (for "updated" events)
- `level_type: str` - Type of level that changed
- `level_name: str` - Name of level that changed
- `message: str` - Current status message
- `percentage: float` - Overall completion percentage
- `elapsed_seconds: float` - Time elapsed since operation started

---

## Simple Progress System (Block-Level)

### ProgressTracker

Location: `src/features/execution/application/progress_tracker.py`

Simple progress tracker for block processors. Publishes `SubprocessProgress` domain events.

```python
class ProgressTracker:
    def start(self, message: str = "") -> None
    def update(self, current: int, total: int, message: str = "") -> None
    def complete(self, message: str = "") -> None
```

### ProgressTrackerContext

Context for creating progress trackers:

```python
@dataclass
class ProgressTrackerContext:
    block: Block
    project_id: str
    event_bus: EventBus
```

### create_progress_tracker()

Factory function:

```python
def create_progress_tracker(context: ProgressTrackerContext) -> ProgressTracker
```

---

## Progress Helpers

Location: `src/features/execution/application/progress_helpers.py`

Utility functions for common progress patterns in block processors.

### get_progress_tracker()

```python
def get_progress_tracker(metadata: Optional[Dict]) -> Optional[ProgressTracker]
```

Extract a ProgressTracker from processor metadata. Returns None if not available.

### IncrementalProgress

Manual step-by-step progress tracking:

```python
class IncrementalProgress:
    def __init__(self, tracker: ProgressTracker, total: int, base_message: str = "")
    def step(self, message: str = "") -> None  # Increment by 1
    def complete(self, message: str = "") -> None
```

### track_progress()

Automatic progress tracking for iterables:

```python
def track_progress(
    iterable,
    tracker: Optional[ProgressTracker],
    message: str = "Processing"
) -> Generator
```

Wraps an iterable and automatically reports progress as items are consumed.

### progress_scope()

Context manager for scoped progress operations:

```python
@contextmanager
def progress_scope(
    tracker: Optional[ProgressTracker],
    start_message: str = "",
    complete_message: str = ""
) -> Generator
```

### yield_progress()

Report progress at specific points:

```python
def yield_progress(
    tracker: Optional[ProgressTracker],
    current: int,
    total: int,
    message: str = ""
) -> None
```
