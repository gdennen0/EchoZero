"""
Application layer for ShowManager feature.

Contains:
- SyncSystemManager: Single orchestration point for all sync operations
- ShowManagerListenerService: Manages OSC listeners
- ShowManagerStateService: Manages connection state
- SyncSubscriptionService: Signal-based subscriptions for layer/event changes
- MA3EventHandler: Handles incoming event changes from MA3
- SyncLayerManager: Compares and merges events using fingerprints

Usage:
    from src.features.show_manager.application.sync_system_manager import SyncSystemManager
    from src.features.show_manager.application import SyncSubscriptionService
    from src.features.show_manager.application import MA3EventHandler
"""
# Note: SyncSystemManager is NOT imported here to avoid circular imports
# Import it directly from: src.features.show_manager.application.sync_system_manager

from src.features.show_manager.application.show_manager_listener_service import ShowManagerListenerService
from src.features.show_manager.application.show_manager_state_service import ShowManagerStateService
from src.features.show_manager.application.sync_subscription_service import (
    SyncSubscriptionService,
    ChangeType,
    SourceType,
    LayerChangeEvent,
    EventChangeEvent,
    Subscription,
    # Convenience functions
    editor_layer_added,
    editor_layer_modified,
    editor_layer_deleted,
    editor_events_added,
    editor_events_modified,
    editor_events_deleted,
    ma3_track_changed,
    ma3_events_changed,
)
from src.features.show_manager.application.ma3_event_handler import (
    MA3EventHandler,
    MA3EventChange,
    MA3ChangeType,
)
from src.features.show_manager.application.sync_layer_manager import (
    SyncLayerManager,
    SyncLayerComparison,
)
from src.features.show_manager.application.sync_safety import (
    SafeSyncService,
    SyncBackupManager,
    SyncValidator,
    SyncAction,
    SyncResult,
    ValidationResult,
    EventSnapshot,
    LayerSnapshot,
)

__all__ = [
    # New architecture - import directly from sync_system_manager module
    # 'SyncSystemManager',  # Not exported to avoid circular imports
    'ShowManagerListenerService',
    'ShowManagerStateService',
    'SyncSubscriptionService',
    'ChangeType',
    'SourceType',
    'LayerChangeEvent',
    'EventChangeEvent',
    'Subscription',
    # MA3 Event Handling
    'MA3EventHandler',
    'MA3EventChange',
    'MA3ChangeType',
    # Sync Layer Manager
    'SyncLayerManager',
    'SyncLayerComparison',
    # Sync Safety
    'SafeSyncService',
    'SyncBackupManager',
    'SyncValidator',
    'SyncAction',
    'SyncResult',
    'ValidationResult',
    'EventSnapshot',
    'LayerSnapshot',
    # Convenience functions
    'editor_layer_added',
    'editor_layer_modified',
    'editor_layer_deleted',
    'editor_events_added',
    'editor_events_modified',
    'editor_events_deleted',
    'ma3_track_changed',
    'ma3_events_changed',
]
