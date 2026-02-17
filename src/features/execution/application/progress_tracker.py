"""
Progress Tracker

Standardized progress tracking for block processors.
Provides a simple, consistent API for reporting progress during block execution.

When execution runs in a background thread (RunBlockThread), progress events
are queued to the main thread. Publishing too often causes GIL contention and
can slow the worker. Updates are throttled to at most one publish per
PROGRESS_THROTTLE_SECONDS so the UI stays responsive without hurting throughput.
"""
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.application.events import EventBus, SubprocessProgress
from src.features.blocks.domain import Block
from src.utils.message import Log

# Minimum seconds between progress publishes when running from a background thread.
# Reduces main-thread wakeups and GIL contention; start/complete/0%/100% always publish.
PROGRESS_THROTTLE_SECONDS = 0.25


@dataclass
class ProgressTrackerContext:
    """Context information for progress tracking"""
    block: Block
    project_id: Optional[str] = None
    event_bus: Optional[EventBus] = None


class ProgressTracker:
    """
    Standardized progress tracker for block processors.
    
    Provides a simple API for reporting progress during block execution.
    Automatically publishes SubprocessProgress events to the event bus.
    
    Usage:
        progress = metadata.get('progress_tracker')
        if progress:
            progress.start("Processing items...", total=100)
            for i in range(100):
                # ... do work ...
                progress.update(current=i+1, message=f"Processed {i+1} items")
            progress.complete("Processing complete")
    
    The tracker handles:
    - Publishing SubprocessProgress events
    - Calculating percentages
    - Formatting progress messages
    - Handling missing event_bus gracefully (no-op if not available)
    """
    
    def __init__(self, context: ProgressTrackerContext):
        """
        Initialize progress tracker.
        
        Args:
            context: ProgressContext with block, project_id, and event_bus
        """
        self.context = context
        self._total: Optional[int] = None
        self._current: int = 0
        self._current_message: str = ""
        self._started: bool = False
        self._last_publish_time: float = 0.0
    
    def start(
        self,
        message: str = "",
        total: Optional[int] = None,
        current: int = 0
    ) -> None:
        """
        Start progress tracking.
        
        Args:
            message: Initial progress message
            total: Total number of items/steps (None for indeterminate)
            current: Current progress (default: 0)
        """
        self._total = total
        self._current = current
        self._current_message = message or f"Processing {self.context.block.name}..."
        self._started = True
        
        self._publish(0, message=self._current_message)
        
        if self.context.event_bus:
            Log.debug(
                f"ProgressTracker: Started progress for block '{self.context.block.name}': "
                f"{message} ({current}/{total if total else '?'})"
            )
    
    def update(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        increment: int = 0
    ) -> None:
        """
        Update progress.
        
        Args:
            current: Set current progress to this value (overrides increment)
            total: Update total (if changed)
            message: Update progress message
            increment: Increment current progress by this amount (ignored if current is set)
        """
        if not self._started:
            # Auto-start if not explicitly started
            self.start(message or self._current_message, total=total)
            return
        
        # Update current
        if current is not None:
            self._current = current
        elif increment != 0:
            self._current += increment
        
        # Update total if provided
        if total is not None:
            self._total = total
        
        # Update message if provided
        if message:
            self._current_message = message
        
        # Calculate percentage
        if self._total and self._total > 0:
            percentage = int((self._current / self._total) * 100)
            # Clamp to 0-100
            percentage = max(0, min(100, percentage))
        else:
            # Indeterminate progress - show as 0% or don't update
            percentage = 0
        
        self._publish(percentage, message=self._current_message)
    
    def complete(self, message: Optional[str] = None) -> None:
        """
        Mark progress as complete.
        
        Args:
            message: Optional completion message
        """
        if not self._started:
            return
        
        final_message = message or f"Completed {self.context.block.name}"
        self._current_message = final_message
        
        # Set to 100% if we have a total
        if self._total and self._total > 0:
            self._current = self._total
            self._publish(100, message=final_message)
        else:
            # For indeterminate progress, just publish final message
            self._publish(100, message=final_message)
        
        if self.context.event_bus:
            Log.debug(
                f"ProgressTracker: Completed progress for block '{self.context.block.name}': {final_message}"
            )
    
    def _publish(self, percentage: int, message: str) -> None:
        """
        Publish SubprocessProgress event. Throttled so we do not publish more
        than once per PROGRESS_THROTTLE_SECONDS (except 0%, 100%, or complete).
        """
        if not self.context.event_bus:
            Log.warning(f"ProgressTracker: No event_bus available, skipping publish for {self.context.block.name}")
            return

        now = time.monotonic()
        # Always publish start (0), complete (100), or first update
        if percentage == 0 or percentage >= 100:
            pass  # always send
        elif now - self._last_publish_time < PROGRESS_THROTTLE_SECONDS:
            return  # skip to reduce main-thread wakeups and GIL contention
        self._last_publish_time = now

        try:
            Log.debug(f"ProgressTracker: Publishing SubprocessProgress - {self.context.block.name}: {percentage}% - {message}")
            self.context.event_bus.publish(SubprocessProgress(
                project_id=self.context.project_id,
                data={
                    "block_id": self.context.block.id,
                    "block_name": self.context.block.name,
                    "subprocess": "block_processing",
                    "percentage": percentage,
                    "current": self._current,
                    "total": self._total,
                    "message": message
                }
            ))
        except Exception as e:
            # Don't let progress tracking errors break execution
            Log.warning(f"ProgressTracker: Failed to publish progress event: {e}")
    
    def is_available(self) -> bool:
        """
        Check if progress tracking is available (event_bus configured).
        
        Returns:
            True if event_bus is available, False otherwise
        """
        return self.context.event_bus is not None


def create_progress_tracker(
    block: Block,
    project_id: Optional[str] = None,
    event_bus: Optional[EventBus] = None
) -> ProgressTracker:
    """
    Create a ProgressTracker instance.
    
    Args:
        block: Block being processed
        project_id: Project identifier
        event_bus: Event bus for publishing progress events
        
    Returns:
        ProgressTracker instance
    """
    context = ProgressTrackerContext(
        block=block,
        project_id=project_id,
        event_bus=event_bus
    )
    return ProgressTracker(context)







