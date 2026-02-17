"""
Status Publisher Pattern

Provides a unified pattern for publishing and subscribing to status updates.
Works with both Qt signals and EventBus for maximum flexibility.

Usage:
    # Create a status publisher
    class MyService(StatusPublisher):
        def __init__(self, event_bus: Optional[EventBus] = None):
            super().__init__(event_bus=event_bus, source_name="MyService")
        
        def do_work(self):
            self.set_status(StatusLevel.PROCESSING, "Working...")
            # ... work ...
            self.set_status(StatusLevel.SUCCESS, "Done!")
    
    # Subscribe to status changes
    service = MyService(event_bus)
    service.subscribe(lambda status: print(f"Status: {status.message}"))
    
    # Or use the StatusSubscriber for cleaner subscription management
    subscriber = StatusSubscriber()
    subscriber.subscribe_to(service, my_handler)
    subscriber.cleanup()  # Unsubscribes from all

Features:
- Unified interface for status publishing
- Automatic change detection (only notifies on actual changes)
- Works with or without Qt
- Works with or without EventBus
- Thread-safe
- Status history tracking (optional)
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Callable, List, Dict, Any, Set
from threading import Lock
import weakref

from src.utils.message import Log

# Try to import EventBus - not required
try:
    from src.application.events.event_bus import EventBus
    from src.application.events.events import DomainEvent
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EventBus = None
    DomainEvent = None
    EVENT_BUS_AVAILABLE = False

# Try to import Qt - not required
try:
    from PyQt6.QtCore import QObject, pyqtSignal
    QT_AVAILABLE = True
except ImportError:
    QObject = object
    pyqtSignal = None
    QT_AVAILABLE = False


# =============================================================================
# Status Level Enum
# =============================================================================

class StatusLevel(Enum):
    """
    Standard status levels for components.
    
    Levels are ordered by severity (lowest to highest).
    """
    UNKNOWN = auto()      # Status not yet determined
    IDLE = auto()         # Inactive, waiting for input
    PENDING = auto()      # Has requirements to fulfill
    PROCESSING = auto()   # Currently working
    WARNING = auto()      # Completed with warnings
    SUCCESS = auto()      # Completed successfully
    ERROR = auto()        # Failed with error
    
    @property
    def color(self) -> str:
        """Get standard color for this status level."""
        colors = {
            StatusLevel.UNKNOWN: "#808080",    # Gray
            StatusLevel.IDLE: "#606060",       # Dark gray
            StatusLevel.PENDING: "#FFA500",    # Orange
            StatusLevel.PROCESSING: "#2196F3", # Blue
            StatusLevel.WARNING: "#FF9800",    # Amber
            StatusLevel.SUCCESS: "#4CAF50",    # Green
            StatusLevel.ERROR: "#F44336",      # Red
        }
        return colors.get(self, "#808080")
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal status (no more work expected)."""
        return self in (StatusLevel.SUCCESS, StatusLevel.ERROR)
    
    @property
    def is_active(self) -> bool:
        """Check if this indicates active work."""
        return self == StatusLevel.PROCESSING


# =============================================================================
# Status Data Class
# =============================================================================

@dataclass
class Status:
    """
    Represents the status of a component.
    
    Attributes:
        level: Status severity level
        message: Human-readable status message
        details: Optional additional details
        source: Source component name
        timestamp: When this status was set
        metadata: Optional additional metadata
    """
    level: StatusLevel = StatusLevel.UNKNOWN
    message: str = ""
    details: Optional[str] = None
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __eq__(self, other) -> bool:
        """Check equality (ignores timestamp for change detection)."""
        if not isinstance(other, Status):
            return False
        return (
            self.level == other.level and
            self.message == other.message and
            self.details == other.details and
            self.source == other.source
        )
    
    def __hash__(self) -> int:
        return hash((self.level, self.message, self.details, self.source))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.name,
            "message": self.message,
            "details": self.details,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Status":
        """Create from dictionary."""
        return cls(
            level=StatusLevel[data.get("level", "UNKNOWN")],
            message=data.get("message", ""),
            details=data.get("details"),
            source=data.get("source", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Status Changed Event (for EventBus integration)
# =============================================================================

if EVENT_BUS_AVAILABLE:
    @dataclass
    class ComponentStatusChanged(DomainEvent):
        """
        Event published when a component's status changes.
        
        This extends the existing StatusChanged event for general component use.
        """
        name = "ComponentStatusChanged"


# =============================================================================
# Status Handler Type
# =============================================================================

StatusHandler = Callable[[Status], None]


# =============================================================================
# Status Publisher Base Class
# =============================================================================

class StatusPublisher:
    """
    Base class for components that publish status updates.
    
    Provides:
    - Unified interface for setting and getting status
    - Automatic change detection
    - Multiple subscription options (direct callbacks, EventBus)
    - Thread-safe operations
    - Optional status history tracking
    
    Usage:
        class MyProcessor(StatusPublisher):
            def __init__(self):
                super().__init__(source_name="MyProcessor")
            
            def process(self, data):
                self.set_status(StatusLevel.PROCESSING, "Processing data...")
                try:
                    result = do_work(data)
                    self.set_status(StatusLevel.SUCCESS, f"Processed {len(result)} items")
                    return result
                except Exception as e:
                    self.set_status(StatusLevel.ERROR, f"Failed: {e}")
                    raise
    
    Attributes:
        source_name: Name of this component (for status source field)
        event_bus: Optional EventBus for cross-component notification
        track_history: Whether to keep status history
        max_history: Maximum history entries to keep
    """
    
    def __init__(
        self,
        source_name: str = "",
        event_bus: Optional["EventBus"] = None,
        track_history: bool = False,
        max_history: int = 100,
    ):
        """
        Initialize status publisher.
        
        Args:
            source_name: Name for status source field
            event_bus: Optional EventBus for publishing events
            track_history: Whether to keep status history
            max_history: Maximum history entries
        """
        self._source_name = source_name or self.__class__.__name__
        self._event_bus = event_bus
        self._track_history = track_history
        self._max_history = max_history
        
        self._current_status: Status = Status(source=self._source_name)
        self._previous_status: Optional[Status] = None
        self._history: List[Status] = []
        self._handlers: List[StatusHandler] = []
        self._lock = Lock()
    
    @property
    def status(self) -> Status:
        """Get current status."""
        with self._lock:
            return self._current_status
    
    @property
    def previous_status(self) -> Optional[Status]:
        """Get previous status (before last change)."""
        with self._lock:
            return self._previous_status
    
    @property
    def source_name(self) -> str:
        """Get source name."""
        return self._source_name
    
    @property
    def history(self) -> List[Status]:
        """Get status history (if tracking enabled)."""
        with self._lock:
            return self._history.copy()
    
    def set_status(
        self,
        level: StatusLevel,
        message: str = "",
        details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_notify: bool = False,
    ) -> None:
        """
        Set the current status.
        
        Only notifies subscribers if status actually changed (or force_notify=True).
        
        Args:
            level: Status level
            message: Status message
            details: Optional additional details
            metadata: Optional metadata
            force_notify: Force notification even if status unchanged
        """
        new_status = Status(
            level=level,
            message=message,
            details=details,
            source=self._source_name,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        
        with self._lock:
            status_changed = self._current_status != new_status
            
            if status_changed:
                self._previous_status = self._current_status
                self._current_status = new_status
                
                # Track history
                if self._track_history:
                    self._history.append(new_status)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:]
            
            # Get handlers to notify
            handlers_to_notify = self._handlers.copy() if (status_changed or force_notify) else []
        
        # Notify outside lock
        if handlers_to_notify:
            for handler in handlers_to_notify:
                try:
                    handler(new_status)
                except Exception as e:
                    Log.error(f"StatusPublisher: Error in handler for '{self._source_name}': {e}")
            
            # Also publish to EventBus if available
            if self._event_bus and EVENT_BUS_AVAILABLE:
                try:
                    event = ComponentStatusChanged(
                        project_id=None,
                        data={
                            "source": self._source_name,
                            "status": new_status.to_dict(),
                            "previous_status": self._previous_status.to_dict() if self._previous_status else None,
                        }
                    )
                    self._event_bus.publish(event)
                except Exception as e:
                    Log.error(f"StatusPublisher: Error publishing to EventBus: {e}")
    
    def subscribe(self, handler: StatusHandler) -> None:
        """
        Subscribe to status changes.
        
        Args:
            handler: Function to call when status changes
        """
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)
    
    def unsubscribe(self, handler: StatusHandler) -> None:
        """
        Unsubscribe from status changes.
        
        Args:
            handler: Handler to remove
        """
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)
    
    def clear_subscriptions(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._handlers.clear()
    
    def reset_status(self) -> None:
        """Reset to initial unknown status."""
        self.set_status(StatusLevel.UNKNOWN, "")
    
    # Convenience methods for common status transitions
    
    def set_idle(self, message: str = "Idle") -> None:
        """Set status to IDLE."""
        self.set_status(StatusLevel.IDLE, message)
    
    def set_pending(self, message: str = "Pending") -> None:
        """Set status to PENDING."""
        self.set_status(StatusLevel.PENDING, message)
    
    def set_processing(self, message: str = "Processing...") -> None:
        """Set status to PROCESSING."""
        self.set_status(StatusLevel.PROCESSING, message)
    
    def set_success(self, message: str = "Success") -> None:
        """Set status to SUCCESS."""
        self.set_status(StatusLevel.SUCCESS, message)
    
    def set_warning(self, message: str, details: Optional[str] = None) -> None:
        """Set status to WARNING."""
        self.set_status(StatusLevel.WARNING, message, details=details)
    
    def set_error(self, message: str, details: Optional[str] = None) -> None:
        """Set status to ERROR."""
        self.set_status(StatusLevel.ERROR, message, details=details)


# =============================================================================
# Status Subscriber Helper
# =============================================================================

class StatusSubscriber:
    """
    Helper class for managing subscriptions to multiple StatusPublishers.
    
    Provides clean subscription management with automatic cleanup.
    
    Usage:
        subscriber = StatusSubscriber()
        subscriber.subscribe_to(service1, my_handler)
        subscriber.subscribe_to(service2, my_handler)
        
        # Later, clean up all subscriptions
        subscriber.cleanup()
    
    Can also be used as a context manager:
        with StatusSubscriber() as sub:
            sub.subscribe_to(service, handler)
            # ... use service ...
        # Automatically cleaned up
    """
    
    def __init__(self):
        self._subscriptions: List[tuple] = []  # List of (publisher, handler)
    
    def subscribe_to(self, publisher: StatusPublisher, handler: StatusHandler) -> None:
        """
        Subscribe to a StatusPublisher.
        
        Args:
            publisher: StatusPublisher to subscribe to
            handler: Handler function
        """
        publisher.subscribe(handler)
        self._subscriptions.append((publisher, handler))
    
    def unsubscribe_from(self, publisher: StatusPublisher, handler: StatusHandler) -> None:
        """
        Unsubscribe from a specific publisher.
        
        Args:
            publisher: StatusPublisher to unsubscribe from
            handler: Handler function to remove
        """
        publisher.unsubscribe(handler)
        self._subscriptions = [
            (p, h) for p, h in self._subscriptions
            if not (p is publisher and h is handler)
        ]
    
    def cleanup(self) -> None:
        """Unsubscribe from all publishers."""
        for publisher, handler in self._subscriptions:
            try:
                publisher.unsubscribe(handler)
            except Exception:
                pass  # Publisher may have been deleted
        self._subscriptions.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# =============================================================================
# Qt-Compatible Status Publisher
# =============================================================================

if QT_AVAILABLE:
    class QtStatusPublisher(QObject, StatusPublisher):
        """
        StatusPublisher with Qt signal support.
        
        Emits a Qt signal in addition to calling handlers, useful for
        connecting to Qt widgets directly.
        
        Usage:
            class MyWidget(QWidget):
                def __init__(self, service: QtStatusPublisher):
                    super().__init__()
                    service.status_changed.connect(self._on_status)
                
                def _on_status(self, status_dict: dict):
                    status = Status.from_dict(status_dict)
                    self.label.setText(status.message)
        """
        
        # Signal emitted when status changes (passes status as dict for Qt compatibility)
        status_changed = pyqtSignal(dict)
        
        def __init__(
            self,
            source_name: str = "",
            event_bus: Optional["EventBus"] = None,
            track_history: bool = False,
            max_history: int = 100,
            parent: Optional[QObject] = None,
        ):
            QObject.__init__(self, parent)
            StatusPublisher.__init__(
                self,
                source_name=source_name,
                event_bus=event_bus,
                track_history=track_history,
                max_history=max_history,
            )
        
        def set_status(
            self,
            level: StatusLevel,
            message: str = "",
            details: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            force_notify: bool = False,
        ) -> None:
            """Set status and emit Qt signal if changed."""
            old_status = self._current_status
            
            # Call parent implementation
            super().set_status(level, message, details, metadata, force_notify)
            
            # Emit Qt signal if status changed
            if self._current_status != old_status or force_notify:
                self.status_changed.emit(self._current_status.to_dict())
else:
    # Fallback when Qt not available
    QtStatusPublisher = StatusPublisher
