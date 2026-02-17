"""
Domain layer for ShowManager feature.

Contains:
- SyncLayerEntity - unified entity for sync layers
- Layer sync types (enums)
- Sync state types - fingerprint matching, conflicts, state tracking

Usage:
    from src.features.show_manager.domain import (
        SyncLayerEntity, SyncLayerSettings,
        SyncSource, SyncStatus, ConflictStrategy,
    )
"""
from src.features.show_manager.domain.layer_sync_types import (
    SyncDirection,
    ConflictResolution,
    LayerSyncStatus,
    SyncType,
)
from src.features.show_manager.domain.sync_state import (
    SyncState,
    ChangeType,
    ConflictType,
    Resolution,
    EventFingerprint,
    MA3EventData,
    SyncChange,
    Conflict,
    ComparisonResult,
    TrackedLayer,
    compute_fingerprint,
    events_match,
)
from src.features.show_manager.domain.sync_layer_entity import (
    SyncLayerEntity,
    SyncLayerSettings,
    SyncSource,
    SyncStatus,
    SyncDirection as NewSyncDirection,
    ConflictStrategy,
)

__all__ = [
    # Unified entity
    'SyncLayerEntity',
    'SyncLayerSettings',
    'SyncSource',
    'SyncStatus',
    'ConflictStrategy',
    # Layer sync types
    'SyncDirection',
    'ConflictResolution',
    'LayerSyncStatus',
    'SyncType',
    # Sync state types
    'SyncState',
    'ChangeType',
    'ConflictType',
    'Resolution',
    'EventFingerprint',
    'MA3EventData',
    'SyncChange',
    'Conflict',
    'ComparisonResult',
    'TrackedLayer',
    'compute_fingerprint',
    'events_match',
]
