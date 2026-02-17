"""
Data Item Entities - Re-export

DEPRECATED: Import from src.shared.domain.entities instead.

This module re-exports DataItem classes from their new location
for backwards compatibility.
"""
# Re-export from new location
from src.shared.domain.entities.data_item import DataItem
from src.shared.domain.entities.audio_data_item import AudioDataItem
from src.shared.domain.entities.event_data_item import EventDataItem

__all__ = ['DataItem', 'AudioDataItem', 'EventDataItem']
