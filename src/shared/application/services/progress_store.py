"""
Progress Event Store

Centralized state management for progress tracking.
Provides query-able progress state and publishes events for UI updates.
"""
import threading
import uuid
from typing import Dict, Optional, List, Any, Callable
from datetime import datetime

from src.shared.application.services.progress_models import (
    ProgressState,
    ProgressLevel,
    ProgressStatus
)
from src.utils.message import Log


class ProgressEventStore:
    """
    Centralized store for progress state.
    
    Maintains current progress state for all active operations and provides
    query methods for accessing progress information. Acts as the backend
    for ProgressContext (the user-facing API).
    
    Features:
    - Thread-safe state management
    - Query-able current state
    - Historical tracking (configurable limit)
    - Event callbacks for UI updates
    - Automatic cleanup of completed operations
    
    Usage:
        store = get_progress_store()
        
        # Start an operation
        state = store.start_operation("setlist_processing", name="My Setlist")
        
        # Add levels
        store.add_level(state.operation_id, "song_1", "song", "Track 1.wav")
        
        # Update progress
        store.update_level(state.operation_id, "song_1", current=1, total=5)
        
        # Query state
        current = store.get_state(state.operation_id)
        
        # Complete
        store.complete_operation(state.operation_id)
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize progress event store.
        
        Args:
            max_history: Maximum number of completed operations to keep in history
        """
        self._states: Dict[str, ProgressState] = {}
        self._history: List[ProgressState] = []
        self._lock = threading.RLock()
        self._max_history = max_history
        self._callbacks: List[Callable[[str, ProgressState], None]] = []
        
        Log.debug("ProgressEventStore: Initialized")
    
    def add_callback(self, callback: Callable[[str, ProgressState], None]) -> None:
        """
        Add a callback for progress updates.
        
        Callback signature: (event_type: str, state: ProgressState) -> None
        Event types: "started", "updated", "completed", "failed"
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, ProgressState], None]) -> None:
        """Remove a progress callback"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify(self, event_type: str, state: ProgressState) -> None:
        """Notify all callbacks of a progress update"""
        for callback in self._callbacks:
            try:
                callback(event_type, state)
            except Exception as e:
                Log.warning(f"ProgressEventStore: Callback error: {e}")
    
    def start_operation(
        self,
        operation_type: str,
        name: str = "",
        operation_id: Optional[str] = None,
        **metadata
    ) -> ProgressState:
        """
        Start tracking a new operation.
        
        Args:
            operation_type: Type of operation (e.g., "setlist_processing")
            name: Display name for the operation
            operation_id: Optional custom operation ID (auto-generated if not provided)
            **metadata: Additional metadata to store
            
        Returns:
            ProgressState for the new operation
        """
        with self._lock:
            if operation_id is None:
                operation_id = f"{operation_type}_{uuid.uuid4().hex[:8]}"
            
            state = ProgressState(
                operation_id=operation_id,
                operation_type=operation_type,
                name=name,
                metadata=metadata
            )
            state.start(name)
            
            self._states[operation_id] = state
            
            Log.info(f"ProgressStore: Started operation '{name}' ({operation_id})")
            self._notify("started", state)
            
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
    
    def add_level(
        self,
        operation_id: str,
        level_id: str,
        level_type: str,
        name: str,
        parent_id: Optional[str] = None,
        total: int = 0,
        **metadata
    ) -> Optional[ProgressLevel]:
        """
        Add a new progress level to an operation.
        
        Args:
            operation_id: Operation to add level to
            level_id: Unique identifier for this level
            level_type: Type of level (song, action, block, etc.)
            name: Display name
            parent_id: Parent level ID (None for root levels)
            total: Total items at this level
            **metadata: Additional metadata
            
        Returns:
            Created ProgressLevel or None if operation not found
        """
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                Log.warning(f"ProgressStore: Operation {operation_id} not found")
                return None
            
            level = state.add_level(
                level_id=level_id,
                level_type=level_type,
                name=name,
                parent_id=parent_id,
                total=total,
                **metadata
            )
            
            Log.debug(
                f"ProgressStore: Added level '{name}' ({level_type}) to {operation_id}"
            )
            self._notify("updated", state)
            
            return level
    
    def start_level(
        self,
        operation_id: str,
        level_id: str,
        message: Optional[str] = None
    ) -> Optional[ProgressLevel]:
        """
        Mark a level as started (running).
        
        Args:
            operation_id: Operation containing the level
            level_id: Level to start
            message: Optional status message
            
        Returns:
            Updated ProgressLevel or None if not found
        """
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                return None
            
            level = state.get_level(level_id)
            if not level:
                return None
            
            level.start(message)
            
            Log.debug(f"ProgressStore: Started level '{level.name}' in {operation_id}")
            self._notify("updated", state)
            
            return level
    
    def update_level(
        self,
        operation_id: str,
        level_id: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        increment: int = 0,
        **metadata
    ) -> Optional[ProgressLevel]:
        """
        Update progress for a level.
        
        Args:
            operation_id: Operation containing the level
            level_id: Level to update
            current: Set current progress
            total: Update total (if changed)
            message: Update status message
            increment: Increment current by this amount
            **metadata: Additional metadata to merge
            
        Returns:
            Updated ProgressLevel or None if not found
        """
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                return None
            
            level = state.get_level(level_id)
            if not level:
                return None
            
            level.update(
                current=current,
                total=total,
                message=message,
                increment=increment,
                **metadata
            )
            
            self._notify("updated", state)
            
            return level
    
    def complete_level(
        self,
        operation_id: str,
        level_id: str,
        message: Optional[str] = None
    ) -> Optional[ProgressLevel]:
        """
        Mark a level as completed.
        
        Args:
            operation_id: Operation containing the level
            level_id: Level to complete
            message: Optional completion message
            
        Returns:
            Updated ProgressLevel or None if not found
        """
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                return None
            
            level = state.get_level(level_id)
            if not level:
                return None
            
            level.complete(message)
            
            Log.debug(
                f"ProgressStore: Completed level '{level.name}' in {operation_id} "
                f"({level.get_duration_str()})"
            )
            self._notify("updated", state)
            
            return level
    
    def fail_level(
        self,
        operation_id: str,
        level_id: str,
        error: str,
        error_details: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressLevel]:
        """
        Mark a level as failed.
        
        Args:
            operation_id: Operation containing the level
            level_id: Level that failed
            error: Error message
            error_details: Optional detailed error information
            
        Returns:
            Updated ProgressLevel or None if not found
        """
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                return None
            
            level = state.get_level(level_id)
            if not level:
                return None
            
            level.fail(error, error_details)
            
            Log.warning(f"ProgressStore: Level '{level.name}' failed: {error}")
            self._notify("updated", state)
            
            return level
    
    def complete_operation(
        self,
        operation_id: str,
        error: Optional[str] = None
    ) -> Optional[ProgressState]:
        """
        Mark an operation as complete.
        
        Args:
            operation_id: Operation to complete
            error: Optional error message (marks as failed if provided)
            
        Returns:
            Completed ProgressState or None if not found
        """
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                return None
            
            state.complete(error)
            
            # Move to history
            self._history.append(state)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            
            # Remove from active states
            del self._states[operation_id]
            
            elapsed = state.get_elapsed_seconds()
            elapsed_str = f"{elapsed:.1f}s" if elapsed else "unknown"
            
            if error:
                Log.warning(
                    f"ProgressStore: Operation '{state.name}' failed after {elapsed_str}: {error}"
                )
                self._notify("failed", state)
            else:
                Log.info(
                    f"ProgressStore: Operation '{state.name}' completed in {elapsed_str}"
                )
                self._notify("completed", state)
            
            return state
    
    def cancel_operation(self, operation_id: str) -> Optional[ProgressState]:
        """
        Cancel an operation.
        
        Args:
            operation_id: Operation to cancel
            
        Returns:
            Cancelled ProgressState or None if not found
        """
        with self._lock:
            state = self._states.get(operation_id)
            if not state:
                return None
            
            state.status = ProgressStatus.CANCELLED
            state.completed_at = datetime.now()
            
            # Move to history
            self._history.append(state)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            
            # Remove from active states
            del self._states[operation_id]
            
            Log.info(f"ProgressStore: Operation '{state.name}' cancelled")
            self._notify("cancelled", state)
            
            return state
    
    def get_history(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ProgressState]:
        """
        Get historical progress states.
        
        Args:
            operation_type: Optional filter by operation type
            limit: Maximum number of results
            
        Returns:
            List of completed ProgressState objects
        """
        with self._lock:
            history = self._history.copy()
            if operation_type:
                history = [s for s in history if s.operation_type == operation_type]
            return history[-limit:]
    
    def clear_history(self) -> None:
        """Clear all historical progress states"""
        with self._lock:
            self._history.clear()
            Log.debug("ProgressStore: History cleared")


# Singleton instance
_store: Optional[ProgressEventStore] = None
_store_lock = threading.Lock()


def get_progress_store() -> ProgressEventStore:
    """
    Get or create the global progress store singleton.
    
    Returns:
        The global ProgressEventStore instance
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ProgressEventStore()
    return _store


def reset_progress_store() -> None:
    """Reset the global progress store (for testing)"""
    global _store
    with _store_lock:
        _store = None

