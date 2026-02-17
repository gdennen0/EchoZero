"""
Domain entities - Core business objects

Note: Most entities have moved to src.shared.domain.entities or src.features.*.domain
This file provides backwards compatibility.
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'ReadOnlyDataItem':
        from src.shared.domain.entities import ReadOnlyDataItem
        return ReadOnlyDataItem
    if name in ('DataItem', 'AudioDataItem', 'EventDataItem', 'DataItemSummary'):
        from src.shared.domain.entities import DataItem, AudioDataItem, EventDataItem, DataItemSummary
        return {'DataItem': DataItem, 'AudioDataItem': AudioDataItem, 
                'EventDataItem': EventDataItem, 'DataItemSummary': DataItemSummary}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['ReadOnlyDataItem', 'DataItem', 'AudioDataItem', 'EventDataItem', 'DataItemSummary']
