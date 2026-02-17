"""
Event Bus System

Provides publish/subscribe pattern for domain events.
Allows UI and other components to react to domain state changes.

Thread-safe and Qt-aware: When events are published from background threads,
handlers are automatically dispatched to the main thread for Qt widget updates.
"""
from typing import Dict, List, Callable, Any, Union, Type
from threading import Lock, current_thread, main_thread
import sys

from src.application.events.events import DomainEvent
from src.utils.message import Log

# Try to import Qt - if available, we'll use it for thread-safe dispatching
try:
    from PyQt6.QtCore import QTimer, QObject, pyqtSignal, QThread, QCoreApplication, QEvent, QMetaObject, Qt
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    QTimer = None
    QObject = None
    QThread = None
    QCoreApplication = None
    QEvent = None
    QMetaObject = None
    Qt = None


# Custom QEvent type for dispatching handler calls to main thread
if QT_AVAILABLE:
    class HandlerCallEvent(QEvent):
        """Custom event to carry handler calls across threads."""
        EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
        
        def __init__(self, handler, event):
            super().__init__(self.EVENT_TYPE)
            self.handler = handler
            self.event = event
    
    class EventDispatcher(QObject):
        """QObject that lives on main thread and processes handler calls."""
        def __init__(self):
            super().__init__()
        
        def event(self, e):
            if isinstance(e, HandlerCallEvent):
                event_name = e.event.name if hasattr(e.event, 'name') else type(e.event).__name__
                try:
                    e.handler(e.event)
                except Exception as ex:
                    Log.error(f"EventBus: Error in handler for '{event_name}': {ex}")
                return True
            return super().event(e)
    
    # Singleton dispatcher instance (must be created on main thread)
    _event_dispatcher = None
    
    def init_event_dispatcher():
        """Initialize the event dispatcher on the main thread. Call during app startup."""
        global _event_dispatcher
        if _event_dispatcher is None:
            _event_dispatcher = EventDispatcher()
            Log.info(f"EventBus: EventDispatcher initialized on {'main' if current_thread() is main_thread() else 'background'} thread")
        return _event_dispatcher
    
    def get_event_dispatcher():
        """Get the event dispatcher. Must call init_event_dispatcher() first from main thread."""
        global _event_dispatcher
        if _event_dispatcher is None:
            # Fallback: create on current thread (may not work if called from background)
            Log.warning("EventBus: EventDispatcher accessed before initialization - creating now (may be wrong thread)")
            _event_dispatcher = EventDispatcher()
        return _event_dispatcher


class EventBus:
    """
    Event bus for publishing and subscribing to domain events.
    
    Thread-safe event bus that allows components to subscribe to events
    and receive notifications when events are published.
    
    Usage:
        bus = EventBus()
        bus.subscribe("BlockAdded", handle_block_added)
        # Or with class:
        bus.subscribe(BlockAdded, handle_block_added)
        bus.publish(BlockAdded(project_id="...", data={...}))
    """
    
    def __init__(self):
        """Initialize event bus"""
        self._subscribers: Dict[str, List[Callable[[DomainEvent], None]]] = {}
        self._lock = Lock()  # Thread safety for multi-threaded environments
        Log.info("EventBus: Initialized")
    
    def _normalize_event_name(self, event_name_or_class: Union[str, Type[DomainEvent]]) -> str:
        """Convert event class or string to normalized string name."""
        if isinstance(event_name_or_class, str):
            return event_name_or_class
        elif hasattr(event_name_or_class, 'name'):
            return event_name_or_class.name
        elif hasattr(event_name_or_class, '__name__'):
            return event_name_or_class.__name__
        else:
            return str(event_name_or_class)
    
    def subscribe(self, event_name: Union[str, Type[DomainEvent]], handler: Callable[[DomainEvent], None]) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_name: Name of the event type (e.g., "BlockAdded") or event class
            handler: Function to call when event is published
                Must accept DomainEvent as parameter
        """
        event_name = self._normalize_event_name(event_name)
        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            
            if handler not in self._subscribers[event_name]:
                self._subscribers[event_name].append(handler)
    
    def unsubscribe(self, event_name: Union[str, Type[DomainEvent]], handler: Callable[[DomainEvent], None]) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_name: Name of the event type or event class
            handler: Handler function to remove
        """
        event_name = self._normalize_event_name(event_name)
        with self._lock:
            if event_name in self._subscribers:
                if handler in self._subscribers[event_name]:
                    self._subscribers[event_name].remove(handler)
                    
                    # Clean up empty lists
                    if not self._subscribers[event_name]:
                        del self._subscribers[event_name]
    
    def publish(self, event: DomainEvent) -> None:
        """
        Publish a domain event to all subscribers.
        
        Thread-safe: If Qt is available and we're not on the main thread,
        handlers are queued to the main thread to prevent Qt widget access violations.
        
        Args:
            event: DomainEvent instance to publish
        """
        event_name = event.name if hasattr(event, 'name') else type(event).__name__
        
        # Get subscribers (make a copy to avoid lock contention during handler execution)
        with self._lock:
            handlers = self._subscribers.get(event_name, []).copy()
        
        if not handlers:
            return
        
        # Only log important events at INFO level
        important_events = {'ProjectCreated', 'project.loaded', 'project.created', 'ExecutionStarted', 'ExecutionCompleted', 'ExecutionFailed'}
        if event_name in important_events:
            Log.info(f"EventBus: Publishing '{event_name}' to {len(handlers)} subscribers")
        
        # Check if we're on the main thread
        is_main_thread = current_thread() is main_thread()
        
        # If Qt is available and we're NOT on the main thread, queue handlers to main thread
        # This prevents Qt widget access violations from background threads
        if QT_AVAILABLE and not is_main_thread:
            # Check if QApplication exists (Qt app must be running)
            app = QCoreApplication.instance()
            if app is not None:
                # Queue each handler to the main thread using QCoreApplication.postEvent
                # This is the proper way to dispatch from background threads to main thread
                dispatcher = get_event_dispatcher()
                for handler in handlers:
                    try:
                        # Post a custom event to the dispatcher (which lives on main thread)
                        QCoreApplication.postEvent(dispatcher, HandlerCallEvent(handler, event))
                    except Exception as e:
                        Log.error(f"EventBus: Error queuing handler for '{event_name}' to main thread: {e}")
            else:
                # Qt not initialized yet - call handlers directly (shouldn't happen in normal flow)
                Log.warning(f"EventBus: Qt not initialized, calling handlers directly from background thread")
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        Log.error(f"EventBus: Error in handler for '{event_name}': {e}")
        else:
            # Call handlers directly (we're on main thread or Qt not available)
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    Log.error(f"EventBus: Error in handler for '{event_name}': {e}")
    
    def _safe_call_handler(self, handler: Callable[[DomainEvent], None], event: DomainEvent) -> None:
        """
        Safely call a handler (used when dispatching from background thread to main thread).
        
        Args:
            handler: Handler function to call
            event: Event to pass to handler
        """
        event_name = event.name if hasattr(event, 'name') else type(event).__name__
        try:
            handler(event)
        except Exception as e:
            Log.error(f"EventBus: Error in handler for '{event_name}': {e}")
    
    def publish_all(self, events: List[DomainEvent]) -> None:
        """
        Publish multiple events in order.
        
        Args:
            events: List of DomainEvent instances
        """
        for event in events:
            self.publish(event)
    
    def get_subscriber_count(self, event_name: str) -> int:
        """
        Get number of subscribers for an event type.
        
        Args:
            event_name: Name of the event type
            
        Returns:
            Number of subscribers
        """
        with self._lock:
            return len(self._subscribers.get(event_name, []))
    
    def clear(self) -> None:
        """Clear all subscribers"""
        with self._lock:
            self._subscribers.clear()
            Log.info("EventBus: Cleared all subscribers")

