"""
Show Manager feature module.

Handles show management including:
- Show manager block state and connection management
- Layer synchronization between Editor and MA3
- OSC listener services for real-time updates
- Sync subscription service for change propagation

Usage:
    from src.features.show_manager.domain import (
        SyncLayerEntity, SyncLayerSettings,
        SyncSource, SyncStatus, ConflictStrategy
    )
    from src.features.show_manager.application.sync_system_manager import SyncSystemManager
    from src.features.show_manager.application import (
        ShowManagerListenerService,
        ShowManagerStateService,
    )
"""
