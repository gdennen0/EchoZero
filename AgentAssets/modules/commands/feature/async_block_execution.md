# Async Block Execution with Progress Tracking

**Status:** Approach Document  
**Date:** December 2025  
**Problem:** Blocks execute synchronously, blocking UI and preventing user interaction during long-running operations.

## Problem Statement

### Current State
1. **Execution happens in background thread** (`ExecutionThread`) - UI stays responsive
2. **But blocks execute sequentially** - each block blocks until completion
3. **No per-block progress** - only block-level progress (block 1/5, block 2/5, etc.)
4. **No execution state tracking** - can't tell which blocks are executing
5. **No edit protection** - users can edit blocks during execution (unsafe)
6. **No execution order safety** - dependencies not enforced during concurrent execution

### User Pain Points
- Can't see progress of long-running blocks (e.g., SeparatorBlock with Demucs)
- Can't edit other blocks while one is executing
- No feedback on which block is currently running
- Can accidentally modify executing blocks (data corruption risk)

### Technical Requirements
1. **Move compute away from main thread** - already done (ExecutionThread)
2. **Track execution state per block** - idle, queued, executing, completed, failed
3. **Report progress within blocks** - percentage completion (0-100%)
4. **Prevent edits to executing blocks** - safety measure
5. **Ensure execution order safety** - respect dependencies
6. **Display progress in main window** - status bar already exists

## Core Principles Applied

### "Best Part is No Part"
- **Reuse existing ExecutionThread pattern** - extend, don't replace
- **Reuse existing event system** - extend events, don't create new system
- **Reuse existing progress bar** - extend StatusBarProgress, don't create new UI
- **No new abstractions** - use simple state tracking

### "Simplicity and Refinement"
- **Simple state machine** - enum for block execution state
- **Explicit progress reporting** - block processors call progress callback
- **Clear safety checks** - simple "is_executing" check before edits
- **No complex queuing** - use existing topological sort

## Proposed Solution

### Architecture Overview

```
Main Thread (UI)
    │
    ├─> ExecutionManager (tracks state)
    │       │
    │       ├─> BlockExecutionState (per block)
    │       │       - idle, queued, executing, completed, failed
    │       │
    │       └─> ExecutionQueue (topological order)
    │
    └─> ExecutionThread (per block)
            │
            ├─> BlockProcessor.process()
            │       │
            │       └─> progress_callback(percentage)  # NEW
            │
            └─> Events published:
                    - BlockExecutionStarted(block_id)
                    - BlockExecutionProgress(block_id, percentage)  # NEW
                    - BlockExecuted(block_id)
                    - BlockExecutionFailed(block_id)
```

### Key Components

#### 1. Block Execution State Tracking

**Location:** `src/application/processing/execution_state.py` (new)

```python
from enum import Enum
from typing import Dict, Optional

class BlockExecutionState(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

class ExecutionStateManager:
    """Tracks execution state for all blocks in a project."""
    
    def __init__(self):
        self._states: Dict[str, BlockExecutionState] = {}
        self._progress: Dict[str, int] = {}  # block_id -> percentage (0-100)
    
    def set_state(self, block_id: str, state: BlockExecutionState):
        """Set execution state for a block."""
        self._states[block_id] = state
        if state == BlockExecutionState.IDLE:
            self._progress[block_id] = 0
    
    def get_state(self, block_id: str) -> BlockExecutionState:
        """Get execution state for a block."""
        return self._states.get(block_id, BlockExecutionState.IDLE)
    
    def is_executing(self, block_id: str) -> bool:
        """Check if block is currently executing."""
        state = self.get_state(block_id)
        return state == BlockExecutionState.EXECUTING
    
    def set_progress(self, block_id: str, percentage: int):
        """Set progress percentage for a block (0-100)."""
        if 0 <= percentage <= 100:
            self._progress[block_id] = percentage
    
    def get_progress(self, block_id: str) -> int:
        """Get progress percentage for a block."""
        return self._progress.get(block_id, 0)
    
    def clear_project(self, project_id: str):
        """Clear all state for a project."""
        # Called when project is unloaded
        self._states.clear()
        self._progress.clear()
```

**Why Simple:**
- Single class, no dependencies
- Clear state enum
- Explicit methods, no magic
- Easy to test

#### 2. Extended Block Processor Interface

**Location:** `src/application/processing/block_processor.py` (modify)

```python
class BlockProcessor(ABC):
    # ... existing methods ...
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int], None]] = None  # NEW
    ) -> Dict[str, DataItem]:
        """
        Process a block with given inputs.
        
        Args:
            block: Block entity to process
            inputs: Dictionary mapping input port names to DataItem instances
            metadata: Optional metadata for processing
            progress_callback: Optional callback for progress updates (0-100)
            
        Returns:
            Dictionary mapping output port names to DataItem instances
        """
        # Default implementation (backward compatible)
        return self._process_impl(block, inputs, metadata, progress_callback)
    
    @abstractmethod
    def _process_impl(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[int], None]]
    ) -> Dict[str, DataItem]:
        """Internal implementation (subclasses override this)."""
        pass
```

**Why Backward Compatible:**
- Existing processors continue to work
- New processors can use progress_callback
- No breaking changes

#### 3. Per-Block Execution Thread

**Location:** `ui/qt_gui/core/block_execution_thread.py` (new)

```python
class BlockExecutionThread(QThread):
    """Executes a single block in background thread."""
    
    # Signals
    execution_started = pyqtSignal(str)  # block_id
    execution_progress = pyqtSignal(str, int)  # block_id, percentage
    execution_complete = pyqtSignal(str, bool)  # block_id, success
    execution_failed = pyqtSignal(str, str)  # block_id, error
    
    def __init__(self, facade, block_id: str, state_manager, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.block_id = block_id
        self.state_manager = state_manager
        self._should_cancel = False
    
    def run(self):
        """Execute block in background thread."""
        try:
            # Set state to executing
            self.state_manager.set_state(self.block_id, BlockExecutionState.EXECUTING)
            self.execution_started.emit(self.block_id)
            
            # Create progress callback
            def progress_callback(percentage: int):
                self.state_manager.set_progress(self.block_id, percentage)
                self.execution_progress.emit(self.block_id, percentage)
            
            # Execute block (with progress callback)
            result = self.facade.execute_single_block_with_progress(
                self.block_id,
                progress_callback=progress_callback
            )
            
            if result.success:
                self.state_manager.set_state(self.block_id, BlockExecutionState.COMPLETED)
                self.execution_complete.emit(self.block_id, True)
            else:
                self.state_manager.set_state(self.block_id, BlockExecutionState.FAILED)
                self.execution_failed.emit(self.block_id, result.message or "Unknown error")
                
        except Exception as e:
            self.state_manager.set_state(self.block_id, BlockExecutionState.FAILED)
            self.execution_failed.emit(self.block_id, str(e))
```

**Why Simple:**
- One thread per block (clear ownership)
- Reuses existing ExecutionThread pattern
- Explicit state management
- Clear error handling

#### 4. Execution Manager (Orchestrator)

**Location:** `src/application/processing/execution_manager.py` (new)

```python
class ExecutionManager:
    """Manages concurrent block execution with dependency safety."""
    
    def __init__(self, facade, state_manager, event_bus):
        self.facade = facade
        self.state_manager = state_manager
        self.event_bus = event_bus
        self._active_threads: Dict[str, BlockExecutionThread] = {}
        self._execution_order: List[str] = []
        self._current_index = 0
    
    def execute_project(self, project_id: str):
        """Execute project with dependency-aware concurrent execution."""
        # Get execution order (topological sort)
        blocks_result = self.facade.list_blocks()
        connections_result = self.facade.list_connections()
        
        execution_order = topological_sort_blocks(blocks_result.data, connections_result.data)
        self._execution_order = [block.id for block in execution_order]
        self._current_index = 0
        
        # Execute blocks in order (respecting dependencies)
        self._execute_next_block()
    
    def _execute_next_block(self):
        """Execute next block in order."""
        if self._current_index >= len(self._execution_order):
            # All blocks executed
            return
        
        block_id = self._execution_order[self._current_index]
        
        # Check if dependencies are ready
        if not self._dependencies_ready(block_id):
            # Wait for dependencies (will be called again when dependency completes)
            return
        
        # Create and start execution thread
        thread = BlockExecutionThread(self.facade, block_id, self.state_manager)
        thread.execution_complete.connect(self._on_block_complete)
        thread.execution_failed.connect(self._on_block_failed)
        
        self._active_threads[block_id] = thread
        self.state_manager.set_state(block_id, BlockExecutionState.QUEUED)
        
        thread.start()
        
        # Move to next block (if dependencies allow)
        self._current_index += 1
        self._execute_next_block()  # Recursive (but safe - limited depth)
    
    def _dependencies_ready(self, block_id: str) -> bool:
        """Check if all dependencies for a block are completed."""
        # Get block's dependencies from connections
        # Check if all dependency blocks are completed
        # (Implementation details)
        return True  # Simplified
    
    def _on_block_complete(self, block_id: str, success: bool):
        """Called when a block execution completes."""
        # Clean up thread
        if block_id in self._active_threads:
            self._active_threads[block_id].deleteLater()
            del self._active_threads[block_id]
        
        # Try to execute next block
        self._execute_next_block()
```

**Why Simple:**
- Sequential execution with dependency checking (not true concurrency yet)
- Clear state transitions
- Reuses existing topological sort
- Easy to extend later

#### 5. New Events

**Location:** `src/application/events/events.py` (extend)

```python
@dataclass
class BlockExecutionStarted(DomainEvent):
    """Event raised when a single block starts executing."""
    name: ClassVar[str] = "BlockExecutionStarted"

@dataclass
class BlockExecutionProgress(DomainEvent):
    """Event raised for progress updates within a block (0-100%)."""
    name: ClassVar[str] = "BlockExecutionProgress"
```

**Why Simple:**
- Extends existing event system
- No new infrastructure
- Consistent with existing events

#### 6. Edit Protection

**Location:** Multiple files (add checks)

```python
# In ApplicationFacade
def update_block_metadata(self, identifier: str, ...):
    if self.execution_state_manager.is_executing(identifier):
        return CommandResult.error_result(
            message="Cannot edit block while it is executing"
        )
    # ... existing code ...

# In BlockPanelBase
def set_block_metadata_key(self, key: str, value: any):
    if self.facade.execution_state_manager.is_executing(self.block_id):
        self.set_status_message("Block is executing - edits disabled", error=True)
        return False
    # ... existing code ...
```

**Why Simple:**
- Single check before edits
- Clear error messages
- No complex locking

#### 7. UI Updates

**Location:** `ui/qt_gui/main_window.py` (extend)

```python
def _on_block_execution_progress(self, event):
    """Handle block-level progress updates."""
    block_id = event.data.get('block_id')
    percentage = event.data.get('percentage', 0)
    
    # Update progress bar with block name and percentage
    block = self.facade.get_block(block_id)
    if block:
        message = f"Executing {block.name}..."
        self.progress_bar.show_progress(message, percentage, 100)
```

**Why Simple:**
- Reuses existing progress bar
- Clear message with block name
- Percentage-based progress

## Implementation Plan

### Phase 1: Foundation (MVP)
1. **Add ExecutionStateManager** - simple state tracking
2. **Extend BlockProcessor interface** - add progress_callback parameter (backward compatible)
3. **Add BlockExecutionProgress event** - new event type
4. **Update ExecutionEngine** - pass progress_callback to processors
5. **Update UI** - subscribe to BlockExecutionProgress events

### Phase 2: Safety
6. **Add edit protection** - check is_executing before edits
7. **Update ApplicationFacade** - expose execution state
8. **Update BlockPanelBase** - disable edits during execution
9. **Update NodeEditor** - show execution state on blocks

### Phase 3: Per-Block Threading (Future)
10. **Add BlockExecutionThread** - per-block execution
11. **Add ExecutionManager** - orchestrate concurrent execution
12. **Dependency checking** - ensure safe execution order

## Safety Measures

### 1. Execution Order Safety
- **Topological sort** - blocks execute in dependency order
- **Dependency checking** - block waits for dependencies before executing
- **Sequential by default** - no true concurrency initially (simpler)

### 2. Edit Protection
- **State check before edits** - simple is_executing() check
- **Clear error messages** - "Block is executing - edits disabled"
- **UI feedback** - disable controls, show status message

### 3. Progress Reporting
- **Optional callback** - processors can ignore if not needed
- **Thread-safe events** - Qt signals for UI updates
- **Percentage-based** - clear 0-100% progress

### 4. Error Handling
- **State cleanup** - failed blocks set to FAILED state
- **Thread cleanup** - proper deleteLater() calls
- **Event publishing** - errors published via events

## Benefits

1. **User can continue working** - edit other blocks while one executes
2. **Clear progress feedback** - see percentage completion
3. **Safety** - can't corrupt executing blocks
4. **Simple** - reuses existing patterns, minimal new code
5. **Backward compatible** - existing processors work unchanged

## Costs

1. **LOC:** ~500 lines (state manager, events, UI updates)
2. **Dependencies:** None (uses existing Qt, Python stdlib)
3. **Testing:** ~10 new tests (state manager, edit protection)
4. **Maintenance:** Low (simple state machine, clear code)

## Alternatives Considered

### Alternative 1: True Concurrent Execution
- **Approach:** Execute all independent blocks simultaneously
- **Why Not:** Too complex, requires complex dependency tracking, race conditions
- **Decision:** Sequential with dependency checking is simpler, safer

### Alternative 2: Process-Based Execution
- **Approach:** Use multiprocessing instead of threading
- **Why Not:** Adds complexity, serialization overhead, harder debugging
- **Decision:** Threading is sufficient, simpler

### Alternative 3: Job Queue System
- **Approach:** Use Celery or similar job queue
- **Why Not:** Massive dependency, overkill for single-user desktop app
- **Decision:** Simple state manager is sufficient

## Success Criteria

1. ✅ User can edit non-executing blocks during execution
2. ✅ Progress percentage displayed in status bar
3. ✅ Edits to executing blocks are prevented with clear error
4. ✅ Execution order respects dependencies
5. ✅ No breaking changes to existing processors
6. ✅ Simple, maintainable code

## Next Steps

1. **Review and approval** - get feedback on approach
2. **Implement Phase 1** - foundation (state manager, events, UI)
3. **Test with one block** - verify progress reporting works
4. **Implement Phase 2** - safety (edit protection)
5. **Test full workflow** - verify user can edit while executing
6. **Iterate** - refine based on usage

---

**The best part is no part. But progress tracking is necessary for user experience. This approach adds minimal complexity while solving the real problem.**


