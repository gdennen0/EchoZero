"""
MA3 (grandMA3) feature module.

Handles grandMA3 integration including:
- OSC communication
- Timecode synchronization
- Event routing

Layers:
    domain/
        MA3Event           - MA3 event entity
        MA3SyncState       - Sync state tracking
    
    application/
        MA3CommunicationService  - OSC communication
        MA3SyncService           - Synchronization
        MA3LayerMappingService   - Layer mapping
        ShowManagerStateService  - State management

Note: Layer sync is now handled by SyncSystemManager and SyncLayerEntity.
Import from: src.features.show_manager.domain import SyncLayerEntity
Import from: src.features.show_manager.application.sync_system_manager import SyncSystemManager

Usage:
    from src.features.ma3 import MA3CommunicationService, MA3SyncState
"""
from src.features.ma3.domain import (
    MA3Event,
    MA3SyncState,
)
from src.features.ma3.application import (
    MA3CommunicationService,
    MA3SyncService,
    MA3LayerMappingService,
    MA3RoutingService,
    ShowManagerListenerService,
    ShowManagerStateService,
)

__all__ = [
    # Domain
    'MA3Event',
    'MA3SyncState',
    # Application
    'MA3CommunicationService',
    'MA3SyncService',
    'MA3LayerMappingService',
    'MA3RoutingService',
    'ShowManagerListenerService',
    'ShowManagerStateService',
]
