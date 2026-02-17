"""
Progress Helpers

Simple utilities to make progress tracking effortless in block processors.

Usage Examples:

    # Context manager for single operation with sub-steps
    with ProgressScope(progress_tracker, "Loading audio", total=3):
        # Do setup work...
        yield_progress(1, "Reading file...")
        data = load_file()
        
        yield_progress(2, "Decoding audio...")
        audio = decode(data)
        
        yield_progress(3, "Creating data item...")
        item = AudioDataItem(audio)

    # Batch processing with automatic progress
    items = [...]
    for item in track_progress(items, progress_tracker, "Processing items"):
        process_item(item)  # Progress updates automatically

    # Manual incremental progress for custom loops
    progress = IncrementalProgress(progress_tracker, "Training", total=100)
    for epoch in range(100):
        train_epoch()
        progress.step(f"Epoch {epoch+1}/100 complete")
"""
from typing import Optional, Iterator, TypeVar, Iterable, Any, Dict
from contextlib import contextmanager

from src.features.execution.application.progress_tracker import ProgressTracker
from src.utils.message import Log


# Type variable for generic iteration
T = TypeVar('T')


class IncrementalProgress:
    """
    Helper for manual incremental progress tracking.
    
    Use when you have a loop with a known number of iterations and want
    to report progress at each step.
    
    Example:
        progress = IncrementalProgress(tracker, "Processing", total=100)
        for i in range(100):
            do_work()
            progress.step(f"Completed {i+1}/100")
    """
    
    def __init__(
        self,
        progress_tracker: Optional[ProgressTracker],
        message: str,
        total: int,
        start_at: int = 0
    ):
        """
        Initialize incremental progress.
        
        Args:
            progress_tracker: ProgressTracker instance (can be None)
            message: Initial message
            total: Total number of steps
            start_at: Starting step number (default: 0)
        """
        self.progress_tracker = progress_tracker
        self.message = message
        self.total = total
        self.current = start_at
        
        if self.progress_tracker:
            self.progress_tracker.start(message, total=total, current=start_at)
    
    def step(self, message: Optional[str] = None) -> None:
        """
        Increment progress by one step.
        
        Args:
            message: Optional message to display
        """
        self.current += 1
        if self.progress_tracker:
            msg = message or f"{self.message} ({self.current}/{self.total})"
            self.progress_tracker.update(current=self.current, message=msg)
    
    def set(self, current: int, message: Optional[str] = None) -> None:
        """
        Set progress to specific value.
        
        Args:
            current: Current step number
            message: Optional message to display
        """
        self.current = current
        if self.progress_tracker:
            msg = message or f"{self.message} ({self.current}/{self.total})"
            self.progress_tracker.update(current=self.current, message=msg)
    
    def complete(self, message: Optional[str] = None) -> None:
        """
        Mark progress as complete.
        
        Args:
            message: Optional completion message
        """
        if self.progress_tracker:
            final_msg = message or f"{self.message} complete"
            self.progress_tracker.complete(final_msg)


@contextmanager
def progress_scope(
    progress_tracker: Optional[ProgressTracker],
    message: str,
    total: Optional[int] = None
):
    """
    Context manager for a scoped progress operation.
    
    Automatically calls start() on enter and complete() on exit.
    Use yield_progress() inside the context to report progress.
    
    Example:
        with progress_scope(tracker, "Loading audio", total=3):
            # Do step 1
            yield_progress(tracker, 1, "Reading file...")
            # Do step 2
            yield_progress(tracker, 2, "Decoding...")
            # Do step 3
            yield_progress(tracker, 3, "Creating item...")
    
    Args:
        progress_tracker: ProgressTracker instance (can be None)
        message: Progress message
        total: Total number of steps (None for indeterminate)
    """
    if progress_tracker:
        progress_tracker.start(message, total=total)
    try:
        yield
    finally:
        if progress_tracker:
            progress_tracker.complete(f"{message} complete")


def yield_progress(
    progress_tracker: Optional[ProgressTracker],
    current: int,
    message: Optional[str] = None
) -> None:
    """
    Report progress at a specific point.
    
    Use inside progress_scope() or standalone.
    
    Args:
        progress_tracker: ProgressTracker instance (can be None)
        current: Current progress value
        message: Optional message
    """
    if progress_tracker:
        progress_tracker.update(current=current, message=message)


def track_progress(
    items: Iterable[T],
    progress_tracker: Optional[ProgressTracker],
    message: str,
    total: Optional[int] = None
) -> Iterator[T]:
    """
    Wrap an iterable to automatically report progress for each item.
    
    Perfect for processing lists/batches where each iteration is one step.
    
    Example:
        audio_files = ["song1.wav", "song2.wav", "song3.wav"]
        for filepath in track_progress(audio_files, tracker, "Loading audio"):
            load_audio(filepath)  # Progress updates automatically!
    
    Args:
        items: Iterable to process
        progress_tracker: ProgressTracker instance (can be None)
        message: Progress message template
        total: Total items (will try to determine from items if None)
    
    Yields:
        Each item from the iterable, with progress automatically reported
    """
    # Try to determine total if not provided
    if total is None:
        try:
            total = len(items)  # type: ignore
        except TypeError:
            # Can't determine length, use indeterminate progress
            total = None
    
    # Start progress
    if progress_tracker:
        progress_tracker.start(message, total=total)
    
    # Iterate and report progress
    current = 0
    try:
        for item in items:
            current += 1
            if progress_tracker:
                if total:
                    msg = f"{message} ({current}/{total})"
                else:
                    msg = f"{message} ({current})"
                progress_tracker.update(current=current, message=msg)
            
            yield item
    finally:
        # Complete progress even if interrupted
        if progress_tracker:
            progress_tracker.complete(f"{message} complete")


def get_progress_tracker(metadata: Optional[Dict[str, Any]]) -> Optional[ProgressTracker]:
    """
    Safely extract progress tracker from metadata.
    
    Helper function to reduce boilerplate in block processors.
    
    Args:
        metadata: Block metadata dictionary
        
    Returns:
        ProgressTracker if available, None otherwise
    """
    if not metadata:
        return None
    return metadata.get("progress_tracker")


