"""
Shared events module.

Provides the event bus system for cross-component communication.
Re-exports from src/application/events for backwards compatibility.

Usage:
    from src.shared.application.events import EventBus, DomainEvent
    
    bus = EventBus()
    bus.subscribe("BlockAdded", handler)
    bus.publish(BlockAdded(project_id="...", data={}))
"""
# Re-export from existing location for backwards compatibility
from src.application.events.event_bus import (
    EventBus,
    init_event_dispatcher,
    get_event_dispatcher,
)
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
    # Other events
    RunProgress,
    RunCompleted,
    ErrorOccurred,
    GraphPositionsUpdated,
    UIStateChanged,
    MA3MessageReceived,
    # Setlist events
    SetlistProcessingStarted,
    SetlistSongProcessing,
    SetlistSongCompleted,
    SetlistProcessingCompleted,
)

__all__ = [
    'EventBus',
    'init_event_dispatcher',
    'get_event_dispatcher',
    'DomainEvent',
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
    # Other events
    'RunProgress',
    'RunCompleted',
    'ErrorOccurred',
    'GraphPositionsUpdated',
    'UIStateChanged',
    'MA3MessageReceived',
    # Setlist events
    'SetlistProcessingStarted',
    'SetlistSongProcessing',
    'SetlistSongCompleted',
    'SetlistProcessingCompleted',
]
