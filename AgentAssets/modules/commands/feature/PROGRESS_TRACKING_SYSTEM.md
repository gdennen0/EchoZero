# Centralized Progress Tracking System Proposal

## Status: IMPLEMENTED

The recommended approach (Context Manager with Event Store backend) has been implemented.
See `AgentAssets/modules/patterns/progress_tracking/` for the pattern documentation.

### Files Created

- `src/shared/application/services/progress_models.py` - Core models
- `src/shared/application/services/progress_store.py` - Event store backend
- `src/shared/application/services/progress_context.py` - Context manager API
- `src/application/events/events.py` - Progress-related events

---

## Problem Statement

The current "Process all" dialog in setlist processing provides basic progress information but lacks detailed, verbose information that would be useful for:
- Understanding what's happening at each stage
- Debugging failures
- Estimating time remaining
- Monitoring resource usage
- Tracking performance metrics

We need a centralized system that can:
1. Start small (trial run for setlist processing)
2. Scale application-wide
3. Provide very verbose, data-rich progress information
4. Be accessible from various endpoints (UI, CLI, API)

## Core Requirements

### Must Have
- **Verbose Information**: Detailed status at every level (setlist → song → action → block → subprocess)
- **Centralized**: Single source of truth for progress state
- **Accessible**: Query-able from multiple endpoints
- **Non-Breaking**: Works alongside existing progress tracking
- **Incremental**: Start small, expand gradually

### Should Have
- **Historical Data**: Track past processing runs
- **Performance Metrics**: Timing, resource usage
- **Error Context**: Detailed error information with context
- **Cancellation Support**: Ability to cancel and see cancellation state

### Nice to Have
- **Predictive**: Time estimates based on historical data
- **Export**: Ability to export progress logs
- **Filtering**: Filter progress by various criteria

## Approach 1: Event-Based Progress Store (Recommended for MVP)

### Concept
Extend the existing event system with a progress store that accumulates and provides query-able progress state.

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│ ProgressEventStore (Singleton)                          │
│ - Stores current progress state                          │
│ - Provides query methods                                │
│ - Publishes events for UI updates                       │
└─────────────────────────────────────────────────────────┘
           ▲                    │
           │                    │
    ┌──────┴──────┐    ┌────────┴────────┐
    │             │    │                │
┌───▼───┐   ┌────▼───┐│  ┌─────────────▼──────────┐
│Blocks │   │Setlist ││  │  ProgressQueryService   │
│       │   │Service ││  │  - get_current_state()  │
└───┬───┘   └────┬───┘│  │  - get_history()        │
    │            │    │  │  - get_metrics()         │
    │            │    │  └─────────────────────────┘
    └────────────┴────┘
           │
    ┌──────▼──────┐
    │ EventBus   │
    └────────────┘
```

### Implementation

**1. ProgressState Model**
```python
@dataclass
class ProgressState:
    """Complete progress state for a processing operation"""
    operation_id: str  # Unique ID for this operation
    operation_type: str  # "setlist_processing", "block_execution", etc.
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    
    # Hierarchical progress
    overall: ProgressLevel
    levels: Dict[str, ProgressLevel]  # song_id, block_id, etc.
    
    # Metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
    metrics: Dict[str, Any]  # timing, resource usage, etc.

@dataclass
class ProgressLevel:
    """Progress at a specific level (song, action, block, etc.)"""
    level_id: str
    level_type: str  # "song", "action", "block", "subprocess"
    name: str
    status: str
    current: int
    total: int
    percentage: float
    message: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
    children: Dict[str, ProgressLevel]  # Nested levels
    metadata: Dict[str, Any]  # Additional context
```

**2. ProgressEventStore**
```python
class ProgressEventStore:
    """Centralized store for progress state"""
    
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._states: Dict[str, ProgressState] = {}
        self._history: List[ProgressState] = []
        self._lock = threading.Lock()
        
        # Subscribe to progress events
        event_bus.subscribe(SetlistProgressEvent, self._on_progress_event)
        event_bus.subscribe(BlockProgressEvent, self._on_progress_event)
        event_bus.subscribe(SubprocessProgress, self._on_progress_event)
    
    def get_current_state(self, operation_id: str) -> Optional[ProgressState]:
        """Get current progress state for an operation"""
        with self._lock:
            return self._states.get(operation_id)
    
    def get_all_active(self) -> List[ProgressState]:
        """Get all currently active operations"""
        with self._lock:
            return [
                state for state in self._states.values()
                if state.status in ["pending", "running"]
            ]
    
    def get_history(self, operation_type: Optional[str] = None, limit: int = 100) -> List[ProgressState]:
        """Get historical progress states"""
        history = self._history
        if operation_type:
            history = [s for s in history if s.operation_type == operation_type]
        return history[-limit:]
    
    def _on_progress_event(self, event: DomainEvent):
        """Update state from progress event"""
        # Parse event and update state
        # Maintain hierarchical structure
        pass
```

**3. Rich Progress Events**
```python
@dataclass
class SetlistProgressEvent(DomainEvent):
    """Verbose setlist processing progress"""
    name: ClassVar[str] = "SetlistProgress"
    
    # Operation context
    operation_id: str
    setlist_id: str
    setlist_name: str
    
    # Current level
    level: str  # "setlist", "song", "action", "block"
    level_id: str  # song_id, action_index, block_id
    
    # Progress
    current: int
    total: int
    percentage: float
    
    # Status
    status: str  # "pending", "running", "completed", "failed"
    message: str
    
    # Context
    song_name: Optional[str] = None
    action_name: Optional[str] = None
    block_name: Optional[str] = None
    block_type: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None
    estimated_remaining_seconds: Optional[float] = None
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)  # CPU, memory, etc.
    
    # Error context
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
```

### Pros
- ✅ Leverages existing event system
- ✅ Non-breaking (additive)
- ✅ Query-able state
- ✅ Can start small (just setlist)
- ✅ Scales naturally

### Cons
- ⚠️ Requires event subscription management
- ⚠️ State synchronization complexity

### Trial Implementation Scope
1. Create `ProgressState` and `ProgressLevel` models
2. Create `ProgressEventStore` (singleton)
3. Add `SetlistProgressEvent` with verbose data
4. Update `SetlistService` to emit rich events
5. Update `SetlistProcessingDialog` to query store
6. Add query endpoint for CLI/API

---

## Approach 2: Progress Context Manager

### Concept
Context manager pattern that automatically tracks progress through nested operations.

### Architecture
```python
with ProgressContext("setlist_processing", setlist_id=setlist_id) as ctx:
    ctx.set_overall(total_songs=10, message="Processing setlist")
    
    for song in songs:
        with ctx.enter_level("song", song_id=song.id, name=song.name) as song_ctx:
            song_ctx.set_total(total_actions=5)
            
            for action in actions:
                with song_ctx.enter_level("action", action_index=idx, name=action.name) as action_ctx:
                    action_ctx.update(current=1, message="Executing...")
                    # ... execute action ...
                    action_ctx.complete()
```

### Implementation

**ProgressContext**
```python
class ProgressContext:
    """Context manager for tracking progress"""
    
    def __init__(
        self,
        operation_type: str,
        operation_id: Optional[str] = None,
        store: Optional[ProgressEventStore] = None,
        **metadata
    ):
        self.operation_type = operation_type
        self.operation_id = operation_id or str(uuid.uuid4())
        self.store = store or get_progress_store()
        self.metadata = metadata
        self._levels: List[ProgressLevel] = []
        self._current_level: Optional[ProgressLevel] = None
    
    def __enter__(self):
        self.store.start_operation(self.operation_id, self.operation_type, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "completed"
        self.store.complete_operation(self.operation_id, status, error=str(exc_val) if exc_val else None)
    
    def enter_level(self, level_type: str, level_id: str, name: str, **metadata):
        """Enter a nested progress level"""
        level = ProgressLevel(
            level_id=level_id,
            level_type=level_type,
            name=name,
            metadata=metadata
        )
        self._levels.append(level)
        self._current_level = level
        self.store.enter_level(self.operation_id, level)
        return LevelContext(self, level)
    
    def update(self, current: Optional[int] = None, total: Optional[int] = None, message: Optional[str] = None):
        """Update current level progress"""
        if self._current_level:
            if current is not None:
                self._current_level.current = current
            if total is not None:
                self._current_level.total = total
            if message:
                self._current_level.message = message
            self.store.update_level(self.operation_id, self._current_level)
```

### Pros
- ✅ Clean API
- ✅ Automatic nesting
- ✅ Exception-safe
- ✅ Easy to use

### Cons
- ⚠️ Requires refactoring existing code
- ⚠️ Context manager overhead
- ⚠️ Less flexible for async operations

---

## Approach 3: Progress Reporter Pattern

### Concept
Injectable progress reporter that can be passed through call chains.

### Architecture
```python
class ProgressReporter:
    """Injectable progress reporter"""
    
    def report(
        self,
        level: str,
        level_id: str,
        status: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        **metadata
    ):
        """Report progress at any level"""
        pass
    
    def get_state(self) -> ProgressState:
        """Get current state"""
        pass
```

### Usage
```python
def process_setlist(setlist_id: str, reporter: ProgressReporter):
    reporter.report("setlist", setlist_id, "running", total=len(songs))
    
    for song in songs:
        reporter.report("song", song.id, "running", message=f"Processing {song.name}")
        # ... process song ...
        reporter.report("song", song.id, "completed")
```

### Pros
- ✅ Very flexible
- ✅ No refactoring needed (injectable)
- ✅ Works with existing code

### Cons
- ⚠️ Requires passing reporter through call chains
- ⚠️ Can be verbose to use
- ⚠️ Easy to forget to report

---

## Approach 4: Hybrid Approach (Recommended for Full Implementation)

### Concept
Combine Approach 1 (Event Store) with Approach 3 (Reporter) for maximum flexibility.

### Architecture
- **ProgressEventStore**: Centralized state management
- **ProgressReporter**: Convenient API for reporting
- **ProgressContext**: Optional context manager for new code
- **Event System**: Backend for state updates

### Implementation Strategy

**Phase 1: MVP (Setlist Only)**
1. Create `ProgressState` and `ProgressLevel` models
2. Create `ProgressEventStore` (singleton)
3. Create `SetlistProgressReporter` (specific to setlist)
4. Update `SetlistService` to use reporter
5. Update `SetlistProcessingDialog` to query store
6. Add verbose progress events

**Phase 2: Block Execution**
1. Extend `ProgressReporter` for block execution
2. Integrate with `ExecutionEngine`
3. Update `ProgressTracker` to use reporter
4. Add block-level progress to UI

**Phase 3: Application-Wide**
1. Create generic `ProgressReporter` interface
2. Add progress query API
3. Add CLI commands for progress
4. Add historical tracking
5. Add performance metrics

---

## Recommended Implementation: Approach 1 (Event Store) for MVP

### Why Approach 1?
- ✅ Minimal changes to existing code
- ✅ Leverages existing event system
- ✅ Query-able state from anywhere
- ✅ Can start with just setlist
- ✅ Scales naturally

### Trial Implementation Plan

**Step 1: Core Models** (1-2 hours)
- Create `ProgressState` and `ProgressLevel` dataclasses
- Create `ProgressEventStore` class
- Add to `src/application/services/progress_service.py`

**Step 2: Setlist Integration** (2-3 hours)
- Create `SetlistProgressEvent` with verbose data
- Update `SetlistService.process_setlist()` to emit events
- Update `SetlistService.process_song()` to emit events
- Update `SetlistService._execute_action_items()` to emit events

**Step 3: Dialog Enhancement** (2-3 hours)
- Update `SetlistProcessingDialog` to query `ProgressEventStore`
- Add detailed progress display (timing, metrics, errors)
- Add expandable sections for verbose information
- Add real-time updates from store

**Step 4: Testing** (1-2 hours)
- Test with small setlist (2-3 songs)
- Test with large setlist (10+ songs)
- Test error scenarios
- Verify UI responsiveness

**Total Estimated Time: 6-10 hours**

---

## Data Richness Examples

### What "Very Verbose" Means

**Current (Basic)**
```
Processing song 1/10: audio1.wav
→ LoadAudio → set_file_path
```

**Proposed (Verbose)**
```
Setlist: "My Setlist" (ID: abc123)
Started: 2024-01-15 10:30:00
Status: Running (3/10 songs completed, 30%)

Song 1/10: audio1.wav (ID: song-001)
  Status: Running
  Started: 2024-01-15 10:30:05
  Elapsed: 0:02:15
  Estimated Remaining: 0:15:30
  
  Action 1/5: LoadAudio → set_file_path
    Status: Completed
    Started: 2024-01-15 10:30:05
    Completed: 2024-01-15 10:30:08
    Duration: 3.2s
    Block: LoadAudio1 (ID: block-001)
    Output: audio_data (44100 Hz, 2 channels, 180.5s)
    
  Action 2/5: DetectOnsets → detect_onsets
    Status: Running
    Started: 2024-01-15 10:30:08
    Elapsed: 0:01:45
    Estimated Remaining: 0:00:30
    Block: DetectOnsets1 (ID: block-002)
    Progress: 85% (127/150 events detected)
    Metrics:
      - CPU: 45%
      - Memory: 128 MB
      - Events Found: 127
      - Processing Rate: 1.2 events/sec
```

---

## Next Steps

1. **Review and Select Approach**: Choose approach based on team feedback
2. **Create Detailed Design**: Expand selected approach with full API design
3. **Implement MVP**: Start with setlist processing
4. **Test and Iterate**: Gather feedback, refine
5. **Expand**: Apply to block execution, then application-wide

---

## Questions to Answer

1. **Storage**: Should progress state persist across restarts? (Probably not for MVP)
2. **History**: How much history to keep? (100 operations? 1000?)
3. **Performance**: How often to update? (Every event? Throttled?)
4. **UI**: How to display verbose info without overwhelming? (Expandable sections? Tabs?)
5. **Metrics**: What metrics to track? (CPU, memory, disk I/O, network?)

