"""
Application layer for setlists feature.

Contains:
- SetlistService - coordinator for setlist operations
- SetlistProcessingService - song processing, action execution, pre/post hooks
- SetlistSnapshotService - song switching via snapshot save/restore
"""
from src.features.setlists.application.setlist_service import SetlistService
from src.features.setlists.application.setlist_processing_service import SetlistProcessingService
from src.features.setlists.application.setlist_snapshot_service import SetlistSnapshotService

__all__ = [
    'SetlistService',
    'SetlistProcessingService',
    'SetlistSnapshotService',
]
