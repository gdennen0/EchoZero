"""
Sync Subscription Service

Provides PyQt signal-based subscriptions for layer/event changes.
Allows ShowManager to subscribe to Editor and MA3 changes for automatic sync.

The service acts as a central hub for change notifications:
- Editor layers/events -> signals -> ShowManager (for sync to MA3)
- MA3 tracks/events -> signals -> ShowManager (for sync to Editor)

Thread-safe design with proper signal/slot connections.
"""
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.message import Log


class ChangeType(Enum):
    """Type of change that occurred."""
    ADDED = auto()      # New item created
    MODIFIED = auto()   # Existing item changed
    DELETED = auto()    # Item removed
    MOVED = auto()      # Item time/position changed
    BATCH = auto()      # Multiple changes at once


class SourceType(Enum):
    """Source of the change."""
    EDITOR = "editor"   # Change came from Editor
    MA3 = "ma3"         # Change came from MA3


@dataclass
class LayerChangeEvent:
    """
    Event data for layer changes.
    
    Attributes:
        source: Where the change came from
        change_type: Type of change
        layer_id: Layer identifier (Editor layer name or MA3 coord)
        block_id: Block identifier (Editor block ID or ShowManager block ID)
        data: Additional data (layer properties, event count, etc.)
    """
    source: SourceType
    change_type: ChangeType
    layer_id: str
    block_id: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventChangeEvent:
    """
    Event data for event/marker changes within a layer.
    
    Attributes:
        source: Where the change came from
        change_type: Type of change
        layer_id: Parent layer identifier
        block_id: Block identifier
        event_ids: List of affected event IDs
        events: List of event data dicts
    """
    source: SourceType
    change_type: ChangeType
    layer_id: str
    block_id: str
    event_ids: List[str] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Subscription:
    """
    Subscription to layer/event changes.
    
    Attributes:
        subscriber_id: Unique ID for the subscriber (e.g., ShowManager block ID)
        source: Source to subscribe to (EDITOR or MA3)
        layer_id: Optional specific layer to watch (None = all layers)
        block_id: Optional specific block to watch (None = all blocks)
        callback: Optional callback function (alternative to signals)
    """
    subscriber_id: str
    source: SourceType
    layer_id: Optional[str] = None
    block_id: Optional[str] = None
    callback: Optional[Callable[[Any], None]] = None


class SyncSubscriptionService(QObject):
    """
    Service for managing subscriptions to layer/event changes.
    
    Provides PyQt signals for change notifications and manages
    subscriptions from ShowManager blocks.
    
    Signals:
        layer_changed: Emitted when a layer is added/modified/deleted
        events_changed: Emitted when events within a layer change
        sync_requested: Emitted when a sync operation is requested
        connection_status_changed: Emitted when MA3 connection status changes
    
    Usage:
        # Create service
        service = SyncSubscriptionService()
        
        # Subscribe to Editor changes
        service.subscribe(
            subscriber_id="show_manager_1",
            source=SourceType.EDITOR,
            layer_id="101_kicks"
        )
        
        # Connect to signals
        service.layer_changed.connect(on_layer_changed)
        service.events_changed.connect(on_events_changed)
        
        # Emit changes (called by Editor or MA3 integration)
        service.emit_layer_change(LayerChangeEvent(...))
    """
    
    # Signals for layer/event changes
    layer_changed = pyqtSignal(object)  # LayerChangeEvent
    events_changed = pyqtSignal(object)  # EventChangeEvent
    
    # Sync operation signals
    sync_requested = pyqtSignal(str, str, str)  # subscriber_id, source_type, layer_id
    sync_completed = pyqtSignal(str, str, str, bool)  # subscriber_id, source_type, layer_id, success
    
    # Status signals
    connection_status_changed = pyqtSignal(str, bool)  # block_id, connected
    
    def __init__(self, parent: Optional[QObject] = None):
        """
        Initialize sync subscription service.
        
        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)
        
        # Subscriptions by subscriber_id
        self._subscriptions: Dict[str, List[Subscription]] = {}
        
        # Active watchers: (source, layer_id, block_id) -> set of subscriber_ids
        self._watchers: Dict[tuple, Set[str]] = {}
        
        Log.info("SyncSubscriptionService: Initialized")
    
    # =========================================================================
    # Subscription Management
    # =========================================================================
    
    def subscribe(
        self,
        subscriber_id: str,
        source: SourceType,
        layer_id: Optional[str] = None,
        block_id: Optional[str] = None,
        callback: Optional[Callable[[Any], None]] = None
    ) -> Subscription:
        """
        Subscribe to layer/event changes.
        
        Args:
            subscriber_id: Unique ID for the subscriber
            source: Source to watch (EDITOR or MA3)
            layer_id: Optional specific layer to watch
            block_id: Optional specific block to watch
            callback: Optional callback function for changes
            
        Returns:
            Created Subscription object
        """
        subscription = Subscription(
            subscriber_id=subscriber_id,
            source=source,
            layer_id=layer_id,
            block_id=block_id,
            callback=callback
        )
        
        # Store subscription
        if subscriber_id not in self._subscriptions:
            self._subscriptions[subscriber_id] = []
        self._subscriptions[subscriber_id].append(subscription)
        
        # Register watcher
        watcher_key = (source.value, layer_id, block_id)
        if watcher_key not in self._watchers:
            self._watchers[watcher_key] = set()
        self._watchers[watcher_key].add(subscriber_id)
        
        Log.debug(f"SyncSubscriptionService: {subscriber_id} subscribed to {source.value} "
                  f"layer={layer_id} block={block_id}")
        
        return subscription
    
    def unsubscribe(
        self,
        subscriber_id: str,
        source: Optional[SourceType] = None,
        layer_id: Optional[str] = None
    ) -> int:
        """
        Unsubscribe from layer/event changes.
        
        Args:
            subscriber_id: Subscriber ID to unsubscribe
            source: Optional - only unsubscribe from this source
            layer_id: Optional - only unsubscribe from this layer
            
        Returns:
            Number of subscriptions removed
        """
        if subscriber_id not in self._subscriptions:
            return 0
        
        subscriptions = self._subscriptions[subscriber_id]
        removed = 0
        remaining = []
        
        for sub in subscriptions:
            should_remove = True
            if source is not None and sub.source != source:
                should_remove = False
            if layer_id is not None and sub.layer_id != layer_id:
                should_remove = False
            
            if should_remove:
                # Remove from watchers
                watcher_key = (sub.source.value, sub.layer_id, sub.block_id)
                if watcher_key in self._watchers:
                    self._watchers[watcher_key].discard(subscriber_id)
                    if not self._watchers[watcher_key]:
                        del self._watchers[watcher_key]
                removed += 1
            else:
                remaining.append(sub)
        
        if remaining:
            self._subscriptions[subscriber_id] = remaining
        else:
            del self._subscriptions[subscriber_id]
        
        Log.debug(f"SyncSubscriptionService: {subscriber_id} unsubscribed {removed} subscription(s)")
        return removed
    
    def unsubscribe_all(self, subscriber_id: str) -> int:
        """
        Remove all subscriptions for a subscriber.
        
        Args:
            subscriber_id: Subscriber ID to unsubscribe completely
            
        Returns:
            Number of subscriptions removed
        """
        return self.unsubscribe(subscriber_id)
    
    def get_subscriptions(self, subscriber_id: str) -> List[Subscription]:
        """
        Get all subscriptions for a subscriber.
        
        Args:
            subscriber_id: Subscriber ID
            
        Returns:
            List of Subscription objects
        """
        return self._subscriptions.get(subscriber_id, []).copy()
    
    def get_subscribers_for_layer(
        self,
        source: SourceType,
        layer_id: str,
        block_id: Optional[str] = None
    ) -> Set[str]:
        """
        Get all subscriber IDs watching a specific layer.
        
        Args:
            source: Source type
            layer_id: Layer identifier
            block_id: Optional block identifier
            
        Returns:
            Set of subscriber IDs
        """
        subscribers = set()
        
        # Check exact matches
        watcher_key = (source.value, layer_id, block_id)
        if watcher_key in self._watchers:
            subscribers.update(self._watchers[watcher_key])
        
        # Check wildcards (layer_id=None means "all layers")
        wildcard_key = (source.value, None, block_id)
        if wildcard_key in self._watchers:
            subscribers.update(self._watchers[wildcard_key])
        
        # Check block wildcards
        wildcard_key = (source.value, layer_id, None)
        if wildcard_key in self._watchers:
            subscribers.update(self._watchers[wildcard_key])
        
        # Check all wildcards
        wildcard_key = (source.value, None, None)
        if wildcard_key in self._watchers:
            subscribers.update(self._watchers[wildcard_key])
        
        return subscribers
    
    # =========================================================================
    # Change Notification
    # =========================================================================
    
    def emit_layer_change(self, event: LayerChangeEvent) -> None:
        """
        Emit a layer change notification.
        
        Args:
            event: LayerChangeEvent with change details
        """
        # Get subscribers for this change
        subscribers = self.get_subscribers_for_layer(
            source=event.source,
            layer_id=event.layer_id,
            block_id=event.block_id
        )
        
        # Emit signal
        self.layer_changed.emit(event)
        
        # Call subscriber callbacks
        for subscriber_id in subscribers:
            subs = self._subscriptions.get(subscriber_id, [])
            for sub in subs:
                if sub.callback:
                    try:
                        sub.callback(event)
                    except Exception as e:
                        Log.warning(f"SyncSubscriptionService: Error in callback for "
                                    f"{subscriber_id}: {e}")
        
        Log.debug(f"SyncSubscriptionService: Emitted layer_changed for {event.layer_id} "
                  f"({event.change_type.name}) to {len(subscribers)} subscriber(s)")
    
    def emit_events_change(self, event: EventChangeEvent) -> None:
        """
        Emit an events change notification.
        
        Args:
            event: EventChangeEvent with change details
        """
        # Get subscribers for this change
        subscribers = self.get_subscribers_for_layer(
            source=event.source,
            layer_id=event.layer_id,
            block_id=event.block_id
        )
        
        # Emit signal
        self.events_changed.emit(event)
        
        # Call subscriber callbacks
        for subscriber_id in subscribers:
            subs = self._subscriptions.get(subscriber_id, [])
            for sub in subs:
                if sub.callback:
                    try:
                        sub.callback(event)
                    except Exception as e:
                        Log.warning(f"SyncSubscriptionService: Error in callback for "
                                    f"{subscriber_id}: {e}")
        
        Log.debug(f"SyncSubscriptionService: Emitted events_changed for {event.layer_id} "
                  f"({len(event.event_ids)} events) to {len(subscribers)} subscriber(s)")
    
    # =========================================================================
    # Sync Operations
    # =========================================================================
    
    def request_sync(
        self,
        subscriber_id: str,
        source: SourceType,
        layer_id: str
    ) -> None:
        """
        Request a sync operation for a layer.
        
        Args:
            subscriber_id: Requesting subscriber
            source: Source of the sync (EDITOR or MA3)
            layer_id: Layer to sync
        """
        self.sync_requested.emit(subscriber_id, source.value, layer_id)
        Log.debug(f"SyncSubscriptionService: Sync requested by {subscriber_id} for "
                  f"{source.value}:{layer_id}")
    
    def complete_sync(
        self,
        subscriber_id: str,
        source: SourceType,
        layer_id: str,
        success: bool
    ) -> None:
        """
        Mark a sync operation as complete.
        
        Args:
            subscriber_id: Subscriber that requested the sync
            source: Source of the sync
            layer_id: Layer that was synced
            success: Whether sync was successful
        """
        self.sync_completed.emit(subscriber_id, source.value, layer_id, success)
        status = "succeeded" if success else "failed"
        Log.debug(f"SyncSubscriptionService: Sync {status} for {subscriber_id} "
                  f"{source.value}:{layer_id}")
    
    # =========================================================================
    # Connection Status
    # =========================================================================
    
    def update_connection_status(self, block_id: str, connected: bool) -> None:
        """
        Update MA3 connection status for a block.
        
        Args:
            block_id: ShowManager block ID
            connected: Whether connected to MA3
        """
        self.connection_status_changed.emit(block_id, connected)
        status = "connected" if connected else "disconnected"
        Log.debug(f"SyncSubscriptionService: Block {block_id} MA3 {status}")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def cleanup(self) -> None:
        """Clean up all subscriptions."""
        subscriber_ids = list(self._subscriptions.keys())
        for subscriber_id in subscriber_ids:
            self.unsubscribe_all(subscriber_id)
        
        self._watchers.clear()
        Log.info("SyncSubscriptionService: Cleaned up all subscriptions")


# Convenience functions for creating events
def editor_layer_added(layer_id: str, block_id: str, **data) -> LayerChangeEvent:
    """Create event for Editor layer added."""
    return LayerChangeEvent(
        source=SourceType.EDITOR,
        change_type=ChangeType.ADDED,
        layer_id=layer_id,
        block_id=block_id,
        data=data
    )


def editor_layer_modified(layer_id: str, block_id: str, **data) -> LayerChangeEvent:
    """Create event for Editor layer modified."""
    return LayerChangeEvent(
        source=SourceType.EDITOR,
        change_type=ChangeType.MODIFIED,
        layer_id=layer_id,
        block_id=block_id,
        data=data
    )


def editor_layer_deleted(layer_id: str, block_id: str) -> LayerChangeEvent:
    """Create event for Editor layer deleted."""
    return LayerChangeEvent(
        source=SourceType.EDITOR,
        change_type=ChangeType.DELETED,
        layer_id=layer_id,
        block_id=block_id
    )


def editor_events_added(
    layer_id: str,
    block_id: str,
    event_ids: List[str],
    events: List[Dict[str, Any]]
) -> EventChangeEvent:
    """Create event for Editor events added."""
    return EventChangeEvent(
        source=SourceType.EDITOR,
        change_type=ChangeType.ADDED,
        layer_id=layer_id,
        block_id=block_id,
        event_ids=event_ids,
        events=events
    )


def editor_events_modified(
    layer_id: str,
    block_id: str,
    event_ids: List[str],
    events: List[Dict[str, Any]]
) -> EventChangeEvent:
    """Create event for Editor events modified."""
    return EventChangeEvent(
        source=SourceType.EDITOR,
        change_type=ChangeType.MODIFIED,
        layer_id=layer_id,
        block_id=block_id,
        event_ids=event_ids,
        events=events
    )


def editor_events_deleted(
    layer_id: str,
    block_id: str,
    event_ids: List[str]
) -> EventChangeEvent:
    """Create event for Editor events deleted."""
    return EventChangeEvent(
        source=SourceType.EDITOR,
        change_type=ChangeType.DELETED,
        layer_id=layer_id,
        block_id=block_id,
        event_ids=event_ids
    )


def ma3_track_changed(coord: str, block_id: str, **data) -> LayerChangeEvent:
    """Create event for MA3 track changed."""
    return LayerChangeEvent(
        source=SourceType.MA3,
        change_type=ChangeType.MODIFIED,
        layer_id=coord,
        block_id=block_id,
        data=data
    )


def ma3_events_changed(
    coord: str,
    block_id: str,
    event_ids: List[str],
    events: List[Dict[str, Any]]
) -> EventChangeEvent:
    """Create event for MA3 events changed."""
    return EventChangeEvent(
        source=SourceType.MA3,
        change_type=ChangeType.MODIFIED,
        layer_id=coord,
        block_id=block_id,
        event_ids=event_ids,
        events=events
    )
