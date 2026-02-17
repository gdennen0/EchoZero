"""
Shared domain entities used across multiple features.

Contains:
- DataItem - base class for all data items
- AudioDataItem - audio-specific data
- EventDataItem - event-based data
- DataItemSummary - lightweight metadata
- ReadOnlyDataItem - immutable wrapper
- EventLayer - event layer for timeline
- DataStateSnapshot - snapshot of data state
- BlockSummary - lightweight block metadata
"""
from src.shared.domain.entities.data_item import DataItem
from src.shared.domain.entities.audio_data_item import AudioDataItem
from src.shared.domain.entities.event_data_item import EventDataItem, Event
from src.shared.domain.entities.data_item_summary import DataItemSummary
from src.shared.domain.entities.read_only_data_item import ReadOnlyDataItem
from src.shared.domain.entities.event_layer import EventLayer
from src.shared.domain.entities.data_state_snapshot import DataStateSnapshot
from src.shared.domain.entities.block_summary import BlockSummary
from src.shared.domain.entities.layer_order import LayerOrder, LayerKey

__all__ = [
    'DataItem',
    'AudioDataItem',
    'EventDataItem',
    'Event',
    'DataItemSummary',
    'ReadOnlyDataItem',
    'EventLayer',
    'DataStateSnapshot',
    'BlockSummary',
    'LayerOrder',
    'LayerKey',
]
