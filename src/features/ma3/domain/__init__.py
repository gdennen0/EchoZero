"""
Domain layer for MA3 (grandMA3) feature.

Contains:
- MA3Event entity
- MA3SyncState entity
- OSC message types

Usage:
    from src.features.ma3.domain import MA3Event, MA3SyncState
    
    # For sync layer entities, use:
    # from src.features.show_manager.domain import SyncLayerEntity
"""
from src.features.ma3.domain.ma3_event import MA3Event
from src.features.ma3.domain.ma3_sync_state import MA3SyncState
from src.features.ma3.domain.osc_message import (
    OSCMessage,
    MessageType,
    ChangeType,
    TrackGroupData,
    TrackData,
    EventData,
)

__all__ = [
    'MA3Event',
    'MA3SyncState',
    # OSC Message types
    'OSCMessage',
    'MessageType',
    'ChangeType',
    'TrackGroupData',
    'TrackData',
    'EventData',
]
