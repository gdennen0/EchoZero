"""
Domain package: Pure data types and aggregate roots for EchoZero's pipeline model.
Exists because the domain layer must be pure data — no Qt, no database, no side effects (FP1, FP7).
All other layers depend on these types; nothing here depends on infrastructure or UI.

Modules:
    enums    — PortType, Direction, BlockState, BlockCategory
    types    — Port, Connection, Event, Layer, EventData, AudioData, BlockSettings, Block
    graph    — Graph aggregate root with invariant enforcement
    events   — Domain event types for the EventBus (BlockAdded, SettingsChanged, etc.)
    project  — ProjectRecord, SongRecord, Setlist entities (added when needed)
"""

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.events import (
    BlockAddedEvent,
    BlockRemovedEvent,
    BlockStateChangedEvent,
    ConnectionAddedEvent,
    ConnectionRemovedEvent,
    DomainEvent,
    ProjectLoadedEvent,
    ProjectSavedEvent,
    SettingsChangedEvent,
    create_event_id,
)
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData,
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
    WaveformData,
    WaveformPeaks,
)

__all__ = [
    # Enums
    "PortType",
    "Direction",
    "BlockState",
    "BlockCategory",
    # Domain events
    "DomainEvent",
    "BlockAddedEvent",
    "BlockRemovedEvent",
    "ConnectionAddedEvent",
    "ConnectionRemovedEvent",
    "BlockStateChangedEvent",
    "SettingsChangedEvent",
    "ProjectLoadedEvent",
    "ProjectSavedEvent",
    "create_event_id",
    # Types
    "Port",
    "Connection",
    "Event",
    "Layer",
    "EventData",
    "AudioData",
    "BlockSettings",
    "Block",
    "WaveformData",
    "WaveformPeaks",
    # Aggregate
    "Graph",
]
