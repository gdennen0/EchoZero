# Progress Tracking System - Implementation Guide

## Quick Start: MVP Implementation (Setlist Only)

This guide provides step-by-step instructions for implementing the Event-Based Progress Store approach for setlist processing.

## Step 1: Create Core Models

**File**: `src/application/services/progress_models.py`

```python
"""
Progress tracking models for centralized progress state management.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum


class ProgressStatus(str, Enum):
    """Status of a progress operation"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressLevel:
    """
    Progress at a specific level (song, action, block, subprocess, etc.)
    
    Supports hierarchical nesting for detailed progress tracking.
    """
    level_id: str
    level_type: str  # "song", "action", "block", "subprocess"
    name: str
    status: ProgressStatus = ProgressStatus.PENDING
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    children: Dict[str, 'ProgressLevel'] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        status: Optional[ProgressStatus] = None,
        **metadata
    ):
        """Update progress level"""
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message
        if status is not None:
            self.status = status
        
        # Update percentage
        if self.total > 0:
            self.percentage = min(100.0, max(0.0, (self.current / self.total) * 100.0))
        
        # Update metadata
        self.metadata.update(metadata)
        
        # Update timestamps
        if status == ProgressStatus.RUNNING and not self.started_at:
            self.started_at = datetime.now()
        elif status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
            if not self.completed_at:
                self.completed_at = datetime.now()


@dataclass
class ProgressState:
    """
    Complete progress state for a processing operation.
    
    Maintains hierarchical progress levels and provides query-able state.
    """
    operation_id: str
    operation_type: str  # "setlist_processing", "block_execution", etc.
    status: ProgressStatus = ProgressStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Hierarchical progress
    overall: Optional[ProgressLevel] = None
    levels: Dict[str, ProgressLevel] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_level(self, level_id: str) -> Optional[ProgressLevel]:
        """Get progress level by ID"""
        return self.levels.get(level_id)
    
    def get_or_create_level(
        self,
        level_id: str,
        level_type: str,
        name: str,
        parent_id: Optional[str] = None
    ) -> ProgressLevel:
        """Get or create a progress level"""
        if level_id in self.levels:
            return self.levels[level_id]
        
        level = ProgressLevel(
            level_id=level_id,
            level_type=level_type,
            name=name
        )
        self.levels[level_id] = level
        
        # Add to parent if specified
        if parent_id and parent_id in self.levels:
            self.levels[parent_id].children[level_id] = level
        
        return level
    
    def get_elapsed_seconds(self) -> Optional[float]:
        """Get elapsed time in seconds"""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
```

## Step 2: Create Progress Event Store

**File**: `src/application/services/progress_store.py`

```python
"""
Centralized progress state store.

Provides query-able progress state and publishes events for UI updates.
"""
import threading
from typing import Dict, Optional, List
from datetime import datetime

from src.application.events import EventBus, DomainEvent
from src.application.services.progress_models import (
    ProgressState,
    ProgressLevel,
    ProgressStatus
)
from src.Utils.message import Log


class ProgressEventStore:
    """
    Centralized store for progress state.
    
    Maintains current progress state for all active operations and provides
    query methods for accessing progress information.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self._event_bus = event_bus
        self._states: Dict[str, ProgressState] = {}
        self._history: List[ProgressState] = []
        self._lock = threading.Lock()
        self._max_history = 100
    
    def start_operation(
        self,
        operation_id: str,
        operation_type: str,
        metadata: Optional[Dict] = None
    ) -> ProgressState:
        """Start tracking a new operation"""
        with self._lock:
            state = ProgressState(
                operation_id=operation_id,
                operation_type=operation_type,
                status=ProgressStatus.PENDING,
                started_at=datetime.now(),
                metadata=metadata or {}
            )
            self._states[operation_id] = state
            Log.debug(f"ProgressStore: Started operation {operation_id} ({operation_type})")
            return state
    
    def get_state(self, operation_id: str) -> Optional[ProgressState]:
        """Get current progress state for an operation"""
        with self._lock:
            return self._states.get(operation_id)
    
    def get_all_active(self) -> List[ProgressState]:
        """Get all currently active operations"""
        with self._lock:
            return [
                state for state in self._states.values()
                if state.status in [ProgressStatus.PENDING, ProgressStatus.RUNNING]
            ]
    
    def update_level(
        self,
        operation_id: str,
        level_id: str,
        level_type: str,
        name: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        status: Optional[ProgressStatus] = None,
        parent_id: Optional[str] = None,
        **metadata
    ) -> Optional[ProgressLevel]:
        """Update or create a progress level"""
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                Log.warning(f"ProgressStore: Operation {operation_id} not found")
                return None
            
            # Update operation status if starting
            if status == ProgressStatus.RUNNING and state.status == ProgressStatus.PENDING:
                state.status = ProgressStatus.RUNNING
            
            # Get or create level
            level = state.get_or_create_level(level_id, level_type, name, parent_id)
            
            # Update level
            level.update(
                current=current,
                total=total,
                message=message,
                status=status,
                **metadata
            )
            
            # Update overall progress if this is a top-level operation
            if not parent_id and state.overall is None:
                state.overall = level
            
            Log.debug(
                f"ProgressStore: Updated {operation_id}/{level_id}: "
                f"{level.status.value} {level.current}/{level.total} - {level.message}"
            )
            
            return level
    
    def complete_operation(
        self,
        operation_id: str,
        status: ProgressStatus = ProgressStatus.COMPLETED,
        error: Optional[str] = None,
        error_details: Optional[Dict] = None
    ):
        """Mark an operation as complete"""
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                return
            
            state.status = status
            state.completed_at = datetime.now()
            if error:
                state.error = error
            if error_details:
                state.error_details = error_details
            
            # Move to history
            self._history.append(state)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            
            # Remove from active states
            del self._states[operation_id]
            
            Log.info(
                f"ProgressStore: Completed operation {operation_id}: {status.value}"
            )
    
    def get_history(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ProgressState]:
        """Get historical progress states"""
        with self._lock:
            history = self._history.copy()
            if operation_type:
                history = [s for s in history if s.operation_type == operation_type]
            return history[-limit:]


# Singleton instance
_store: Optional[ProgressEventStore] = None


def get_progress_store(event_bus: Optional[EventBus] = None) -> ProgressEventStore:
    """Get or create the global progress store"""
    global _store
    if _store is None:
        _store = ProgressEventStore(event_bus)
    return _store
```

## Step 3: Create Rich Progress Event

**File**: `src/application/events/events.py` (add to existing file)

```python
@dataclass
class SetlistProgressEvent(DomainEvent):
    """
    Verbose setlist processing progress event.
    
    Provides detailed progress information at all levels:
    - Setlist level (overall progress)
    - Song level (individual song progress)
    - Action level (action execution progress)
    - Block level (block processing progress)
    """
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
    started_at: Optional[str] = None  # ISO format datetime string
    elapsed_seconds: Optional[float] = None
    estimated_remaining_seconds: Optional[float] = None
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)  # CPU, memory, etc.
    
    # Error context
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
```

## Step 4: Update SetlistService

**File**: `src/application/services/setlist_service.py`

Add progress store integration:

```python
from src.application.services.progress_store import get_progress_store
from src.application.services.progress_models import ProgressStatus
from src.application.events import SetlistProgressEvent
from datetime import datetime

# In __init__ or where event_bus is available:
self._progress_store = get_progress_store(self._event_bus)

# In process_setlist():
def process_setlist(...):
    operation_id = f"setlist_{setlist_id}_{int(datetime.now().timestamp())}"
    state = self._progress_store.start_operation(
        operation_id=operation_id,
        operation_type="setlist_processing",
        metadata={"setlist_id": setlist_id, "setlist_name": setlist.name}
    )
    
    # Update overall progress
    self._progress_store.update_level(
        operation_id=operation_id,
        level_id="setlist",
        level_type="setlist",
        name=setlist.name or "Setlist",
        total=len(songs),
        status=ProgressStatus.RUNNING
    )
    
    # Emit event
    if self._event_bus:
        self._event_bus.publish(SetlistProgressEvent(
            project_id=self.current_project_id,
            data={
                "operation_id": operation_id,
                "setlist_id": setlist_id,
                "setlist_name": setlist.name or "Setlist",
                "level": "setlist",
                "level_id": "setlist",
                "current": 0,
                "total": len(songs),
                "percentage": 0.0,
                "status": "running",
                "message": f"Starting processing of {len(songs)} songs"
            }
        ))
    
    # In song loop:
    for index, song in enumerate(songs):
        # Update song level
        self._progress_store.update_level(
            operation_id=operation_id,
            level_id=song.id,
            level_type="song",
            name=Path(song.audio_path).name,
            current=index,
            total=len(songs),
            status=ProgressStatus.RUNNING,
            parent_id="setlist",
            metadata={"audio_path": song.audio_path}
        )
        
        # Emit event
        if self._event_bus:
            self._event_bus.publish(SetlistProgressEvent(
                project_id=self.current_project_id,
                data={
                    "operation_id": operation_id,
                    "setlist_id": setlist_id,
                    "setlist_name": setlist.name or "Setlist",
                    "level": "song",
                    "level_id": song.id,
                    "song_name": Path(song.audio_path).name,
                    "current": index + 1,
                    "total": len(songs),
                    "percentage": ((index + 1) / len(songs)) * 100,
                    "status": "running",
                    "message": f"Processing song {index + 1}/{len(songs)}: {Path(song.audio_path).name}"
                }
            ))
        
        # Process song...
        # Update action progress in action_progress_callback
        
    # Complete operation
    self._progress_store.complete_operation(operation_id, ProgressStatus.COMPLETED)
```

## Step 5: Update SetlistProcessingDialog

**File**: `ui/qt_gui/dialogs/setlist_processing_dialog.py`

Add query methods and verbose display:

```python
from src.application.services.progress_store import get_progress_store
from src.application.services.progress_models import ProgressStatus

class SetlistProcessingDialog(QDialog):
    def __init__(self, ...):
        # ... existing code ...
        self._progress_store = get_progress_store()
        self._operation_id: Optional[str] = None
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_from_store)
        self._update_timer.start(100)  # Update every 100ms
    
    def set_operation_id(self, operation_id: str):
        """Set the operation ID to track"""
        self._operation_id = operation_id
    
    def _update_from_store(self):
        """Update UI from progress store"""
        if not self._operation_id:
            return
        
        state = self._progress_store.get_state(self._operation_id)
        if not state:
            return
        
        # Update overall progress
        if state.overall:
            self.overall_progress.setValue(int(state.overall.percentage))
            self.overall_progress.setFormat(
                f"{state.overall.percentage:.1f}% "
                f"({state.overall.current}/{state.overall.total} songs)"
            )
        
        # Update song and action items from state
        for level_id, level in state.levels.items():
            if level.level_type == "song":
                self._update_song_from_level(level_id, level)
            elif level.level_type == "action":
                # Find parent song and update action
                # ... implementation ...
                pass
    
    def _update_song_from_level(self, song_id: str, level: ProgressLevel):
        """Update song display from progress level"""
        # Update song status
        status_map = {
            ProgressStatus.PENDING: "pending",
            ProgressStatus.RUNNING: "processing",
            ProgressStatus.COMPLETED: "completed",
            ProgressStatus.FAILED: "failed"
        }
        self.update_song_status(song_id, status_map.get(level.status, "pending"))
        
        # Update action children
        for action_id, action_level in level.children.items():
            action_index = int(action_id)  # Assuming action_id is index
            self.update_action_status(
                song_id,
                action_index,
                status_map.get(action_level.status, "pending")
            )
            
            # Add verbose information to tooltip
            tooltip = self._build_action_tooltip(action_level)
            if song_id in self.action_items_map and action_index in self.action_items_map[song_id]:
                self.action_items_map[song_id][action_index].setToolTip(0, tooltip)
    
    def _build_action_tooltip(self, level: ProgressLevel) -> str:
        """Build verbose tooltip for action"""
        lines = [
            f"Status: {level.status.value}",
            f"Progress: {level.current}/{level.total} ({level.percentage:.1f}%)",
            f"Message: {level.message}"
        ]
        
        if level.started_at:
            lines.append(f"Started: {level.started_at.strftime('%H:%M:%S')}")
        if level.completed_at:
            elapsed = (level.completed_at - level.started_at).total_seconds()
            lines.append(f"Duration: {elapsed:.1f}s")
        elif level.started_at:
            elapsed = (datetime.now() - level.started_at).total_seconds()
            lines.append(f"Elapsed: {elapsed:.1f}s")
        
        if level.metadata:
            for key, value in level.metadata.items():
                lines.append(f"{key}: {value}")
        
        if level.error:
            lines.append(f"Error: {level.error}")
        
        return "\n".join(lines)
```

## Step 6: Update SetlistView

**File**: `ui/qt_gui/views/setlist_view.py`

Pass operation_id to dialog:

```python
def _on_process_all(self):
    # ... existing code ...
    
    # Generate operation ID
    operation_id = f"setlist_{self.current_setlist_id}_{int(datetime.now().timestamp())}"
    
    # Create dialog
    dialog = SetlistProcessingDialog(...)
    dialog.set_operation_id(operation_id)
    
    # Update process_setlist to accept and use operation_id
    # Or extract from progress store after starting
```

## Testing Checklist

- [ ] Small setlist (2-3 songs) processes correctly
- [ ] Large setlist (10+ songs) processes correctly
- [ ] Progress updates in real-time
- [ ] Verbose information displays correctly
- [ ] Error scenarios show detailed error information
- [ ] UI remains responsive during processing
- [ ] Progress state query-able from dialog
- [ ] Operation completes and moves to history

## Next Steps After MVP

1. Add timing estimates (based on historical data)
2. Add performance metrics (CPU, memory)
3. Add CLI query command
4. Extend to block execution
5. Add progress export functionality

