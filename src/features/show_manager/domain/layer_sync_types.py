"""
Layer Sync Types

Shared enums and types for layer synchronization between Editor and MA3.
"""
from enum import Enum, auto


class SyncDirection(Enum):
    """Direction of synchronization."""
    MA3_TO_EZ = auto()  # MA3 is source, EZ is target
    EZ_TO_MA3 = auto()  # EZ is source, MA3 is target
    BIDIRECTIONAL = auto()  # Both directions active


class ConflictResolution(Enum):
    """How to resolve conflicts."""
    USE_MA3 = auto()  # MA3 version wins
    USE_EZ = auto()  # EchoZero version wins
    MERGE = auto()  # Attempt to merge changes
    SKIP = auto()  # Skip this event, leave as-is
    PROMPT_USER = auto()  # Ask user to decide


class LayerSyncStatus(Enum):
    """Status of layer synchronization."""
    SYNCED = "synced"
    OUT_OF_SYNC = "out_of_sync"
    CONFLICT = "conflict"
    UNMAPPED = "unmapped"
    EXCLUDED = "excluded"


class SyncType(Enum):
    """Type of sync behavior for a layer."""
    SHOWMANAGER_LAYER = "showmanager_layer"
    EZ_LAYER = "ez_layer"


__all__ = [
    'SyncDirection',
    'ConflictResolution',
    'LayerSyncStatus',
    'SyncType',
]
