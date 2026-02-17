"""
Progress Context Manager

Provides a "built-in" API for progress tracking using Python context managers.
Progress is automatically tracked when entering/exiting contexts, with
automatic handling of success, failure, and nesting.

Usage:
    progress = ProgressContext()
    
    with progress.operation("setlist_processing", "My Setlist") as op:
        op.set_total(10)  # 10 songs
        
        for song in songs:
            with op.level("song", song.id, song.name) as song_ctx:
                song_ctx.set_total(5)  # 5 actions
                
                for action in actions:
                    with song_ctx.level("action", str(idx), action.name) as action_ctx:
                        action_ctx.update(message="Executing...")
                        # ... do work ...
                        # Automatically completed when exiting context
                        # Automatically failed if exception raised
"""
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Generator
import uuid

from src.shared.application.services.progress_store import (
    ProgressEventStore,
    get_progress_store
)
from src.shared.application.services.progress_models import (
    ProgressState,
    ProgressLevel,
    ProgressStatus
)
from src.utils.message import Log


class LevelContext:
    """
    Context for a progress level within an operation.
    
    Provides methods to update progress and create nested levels.
    Automatically marks level as completed when exiting context,
    or failed if an exception is raised.
    """
    
    def __init__(
        self,
        store: ProgressEventStore,
        operation_id: str,
        level_id: str,
        parent_context: Optional['LevelContext'] = None
    ):
        self._store = store
        self._operation_id = operation_id
        self._level_id = level_id
        self._parent = parent_context
        self._children: List['LevelContext'] = []
    
    @property
    def level_id(self) -> str:
        """Get the level ID"""
        return self._level_id
    
    @property
    def operation_id(self) -> str:
        """Get the operation ID"""
        return self._operation_id
    
    def set_total(self, total: int) -> None:
        """Set the total count for this level"""
        self._store.update_level(self._operation_id, self._level_id, total=total)
    
    def update(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        increment: int = 0,
        **metadata
    ) -> None:
        """
        Update progress for this level.
        
        Args:
            current: Set current progress to this value
            total: Update total (if changed)
            message: Update status message
            increment: Increment current by this amount
            **metadata: Additional metadata to merge
        """
        self._store.update_level(
            self._operation_id,
            self._level_id,
            current=current,
            total=total,
            message=message,
            increment=increment,
            **metadata
        )
    
    def increment(self, message: Optional[str] = None) -> None:
        """Increment progress by 1"""
        self._store.update_level(
            self._operation_id,
            self._level_id,
            increment=1,
            message=message
        )
    
    @contextmanager
    def level(
        self,
        level_type: str,
        level_id: str,
        name: str,
        total: int = 0,
        **metadata
    ) -> Generator['LevelContext', None, None]:
        """
        Create a nested progress level.
        
        Args:
            level_type: Type of level (e.g., "action", "block")
            level_id: Unique identifier for this level
            name: Display name
            total: Total items at this level
            **metadata: Additional metadata
            
        Yields:
            LevelContext for the nested level
        """
        # Create full level ID to ensure uniqueness
        full_level_id = f"{self._level_id}:{level_id}"
        
        # Add level to store
        self._store.add_level(
            self._operation_id,
            full_level_id,
            level_type,
            name,
            parent_id=self._level_id,
            total=total,
            **metadata
        )
        
        # Start level
        self._store.start_level(self._operation_id, full_level_id)
        
        # Create child context
        child = LevelContext(
            self._store,
            self._operation_id,
            full_level_id,
            parent_context=self
        )
        self._children.append(child)
        
        try:
            yield child
        except Exception as e:
            # Mark as failed
            self._store.fail_level(
                self._operation_id,
                full_level_id,
                str(e),
                error_details={"exception_type": type(e).__name__}
            )
            raise
        else:
            # Mark as completed
            self._store.complete_level(self._operation_id, full_level_id)
    
    # Convenience methods for common level types
    
    @contextmanager
    def song(
        self,
        song_id: str,
        name: str,
        total: int = 0,
        **metadata
    ) -> Generator['LevelContext', None, None]:
        """Create a song-level progress context"""
        with self.level("song", song_id, name, total, **metadata) as ctx:
            yield ctx
    
    @contextmanager
    def action(
        self,
        action_id: str,
        name: str,
        total: int = 0,
        **metadata
    ) -> Generator['LevelContext', None, None]:
        """Create an action-level progress context"""
        with self.level("action", action_id, name, total, **metadata) as ctx:
            yield ctx
    
    @contextmanager
    def block(
        self,
        block_id: str,
        name: str,
        block_type: str = "",
        total: int = 0,
        **metadata
    ) -> Generator['LevelContext', None, None]:
        """Create a block-level progress context"""
        if block_type:
            metadata["block_type"] = block_type
        with self.level("block", block_id, name, total, **metadata) as ctx:
            yield ctx
    
    @contextmanager
    def subprocess(
        self,
        subprocess_id: str,
        name: str,
        total: int = 0,
        **metadata
    ) -> Generator['LevelContext', None, None]:
        """Create a subprocess-level progress context"""
        with self.level("subprocess", subprocess_id, name, total, **metadata) as ctx:
            yield ctx


class OperationContext(LevelContext):
    """
    Context for a top-level operation.
    
    Extends LevelContext with operation-specific methods.
    """
    
    def __init__(
        self,
        store: ProgressEventStore,
        operation_id: str
    ):
        super().__init__(store, operation_id, "operation")
    
    def get_state(self) -> Optional[ProgressState]:
        """Get current operation state"""
        return self._store.get_state(self._operation_id)
    
    def set_total(self, total: int) -> None:
        """Set total count for the operation (number of root-level items)"""
        # For operations, we update the metadata since operations don't have
        # a traditional current/total tracking
        state = self._store.get_state(self._operation_id)
        if state:
            state.metadata["total"] = total


class ProgressContext:
    """
    Main entry point for progress tracking.
    
    Provides a "built-in" API for progress tracking using context managers.
    Use this to create operations and track progress at multiple levels.
    
    Usage:
        progress = ProgressContext()
        
        with progress.operation("setlist_processing", "My Setlist") as op:
            for song in songs:
                with op.song(song.id, song.name) as song_ctx:
                    for action in actions:
                        with song_ctx.action(str(idx), action.name) as action_ctx:
                            action_ctx.update(message="Processing...")
                            # ... do work ...
    """
    
    def __init__(self, store: Optional[ProgressEventStore] = None):
        """
        Initialize progress context.
        
        Args:
            store: Optional custom progress store (uses global singleton if not provided)
        """
        self._store = store or get_progress_store()
    
    @property
    def store(self) -> ProgressEventStore:
        """Get the progress store"""
        return self._store
    
    @contextmanager
    def operation(
        self,
        operation_type: str,
        name: str = "",
        operation_id: Optional[str] = None,
        **metadata
    ) -> Generator[OperationContext, None, None]:
        """
        Start a new progress-tracked operation.
        
        Args:
            operation_type: Type of operation (e.g., "setlist_processing")
            name: Display name for the operation
            operation_id: Optional custom operation ID
            **metadata: Additional metadata
            
        Yields:
            OperationContext for the operation
        """
        # Start operation in store
        state = self._store.start_operation(
            operation_type,
            name,
            operation_id,
            **metadata
        )
        
        # Create context
        ctx = OperationContext(self._store, state.operation_id)
        
        try:
            yield ctx
        except Exception as e:
            # Mark as failed
            self._store.complete_operation(state.operation_id, str(e))
            raise
        else:
            # Mark as completed
            self._store.complete_operation(state.operation_id)
    
    # Convenience methods for common operation types
    
    @contextmanager
    def setlist_processing(
        self,
        setlist_id: str,
        name: str = "Setlist Processing",
        **metadata
    ) -> Generator[OperationContext, None, None]:
        """
        Start a setlist processing operation.
        
        Args:
            setlist_id: Setlist identifier
            name: Display name
            **metadata: Additional metadata
            
        Yields:
            OperationContext for setlist processing
        """
        metadata["setlist_id"] = setlist_id
        with self.operation("setlist_processing", name, **metadata) as op:
            yield op
    
    @contextmanager
    def block_execution(
        self,
        block_id: str,
        block_name: str,
        block_type: str = "",
        **metadata
    ) -> Generator[OperationContext, None, None]:
        """
        Start a block execution operation.
        
        Args:
            block_id: Block identifier
            block_name: Block display name
            block_type: Block type
            **metadata: Additional metadata
            
        Yields:
            OperationContext for block execution
        """
        metadata["block_id"] = block_id
        metadata["block_type"] = block_type
        with self.operation("block_execution", block_name, **metadata) as op:
            yield op
    
    def get_active_operations(self) -> List[ProgressState]:
        """Get all currently active operations"""
        return self._store.get_all_active()
    
    def get_operation(self, operation_id: str) -> Optional[ProgressState]:
        """Get a specific operation by ID"""
        return self._store.get_state(operation_id)
    
    def get_history(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ProgressState]:
        """Get historical operations"""
        return self._store.get_history(operation_type, limit)


# Convenience function to get a progress context
def get_progress_context(store: Optional[ProgressEventStore] = None) -> ProgressContext:
    """
    Get a progress context instance.
    
    Args:
        store: Optional custom progress store
        
    Returns:
        ProgressContext instance
    """
    return ProgressContext(store)

