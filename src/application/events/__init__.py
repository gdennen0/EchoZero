"""Event system for application layer"""

from src.application.events.events import (
    DomainEvent,
    # Project events
    ProjectInitialized,
    ProjectCreated,
    ProjectLoaded,
    ProjectUpdated,
    ProjectDeleted,
    # Block events
    BlockAdded,
    BlockUpdated,
    BlockRemoved,
    BlockChanged,
    StatusChanged,
    # Connection events
    ConnectionsChanged,
    ConnectionCreated,
    ConnectionRemoved,
    # Execution events
    ExecutionStarted,
    ExecutionProgress,
    BlockExecuted,
    BlockExecutionFailed,
    ExecutionCompleted,
    SubprocessProgress,
    # Run events
    RunProgress,
    RunCompleted,
    # Error events
    ErrorOccurred,
    # UI events
    GraphPositionsUpdated,
    UIStateChanged,
    # MA3 communication events
    MA3MessageReceived,
    # Setlist events
    SetlistProcessingStarted,
    SetlistSongProcessing,
    SetlistSongCompleted,
    SetlistProcessingCompleted,
)
from src.application.events.event_bus import EventBus, init_event_dispatcher

__all__ = [
    'DomainEvent',
    'EventBus',
    'init_event_dispatcher',
    # Project events
    'ProjectInitialized',
    'ProjectCreated',
    'ProjectLoaded',
    'ProjectUpdated',
    'ProjectDeleted',
    # Block events
    'BlockAdded',
    'BlockUpdated',
    'BlockRemoved',
    'BlockChanged',
    'StatusChanged',
    # Connection events
    'ConnectionsChanged',
    'ConnectionCreated',
    'ConnectionRemoved',
    # Execution events
    'ExecutionStarted',
    'ExecutionProgress',
    'BlockExecuted',
    'BlockExecutionFailed',
    'ExecutionCompleted',
    'SubprocessProgress',
    # Run events
    'RunProgress',
    'RunCompleted',
    # Error events
    'ErrorOccurred',
    # UI events
    'GraphPositionsUpdated',
    'UIStateChanged',
    # MA3 communication events
    'MA3MessageReceived',
    # Setlist events
    'SetlistProcessingStarted',
    'SetlistSongProcessing',
    'SetlistSongCompleted',
    'SetlistProcessingCompleted',
]
