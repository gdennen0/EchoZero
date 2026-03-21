"""
Domain types: Frozen dataclasses for EchoZero's core entities.
Exists because every concept in the pipeline needs a typed, immutable representation (FP2).
All types are value objects or entities — no behavior, no side effects, no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType


@dataclass(frozen=True)
class Port:
    """A typed input or output slot on a block."""

    name: str
    port_type: PortType
    direction: Direction


@dataclass(frozen=True)
class Connection:
    """A typed link between an output port and an input port on two blocks."""

    source_block_id: str
    source_output_name: str
    target_block_id: str
    target_input_name: str


@dataclass(frozen=True)
class Event:
    """A time-stamped marker or region on the timeline."""

    id: str
    time: float
    duration: float
    classifications: dict[str, Any]
    metadata: dict[str, Any]
    origin: str


@dataclass(frozen=True)
class Layer:
    """A named group of events on the timeline."""

    id: str
    name: str
    events: tuple[Event, ...]


@dataclass(frozen=True)
class EventData:
    """Immutable collection of layers produced by a pipeline execution."""

    layers: tuple[Layer, ...]


@dataclass(frozen=True)
class AudioData:
    """Reference to an audio file on disk with format metadata."""

    sample_rate: int
    duration: float
    file_path: str
    channel_count: int = 1


@dataclass(frozen=True)
class BlockSettings:
    """Configuration entries for a block instance."""

    entries: dict[str, Any]


@dataclass(frozen=True)
class Block:
    """A processing unit in the pipeline graph."""

    id: str
    name: str
    block_type: str
    category: BlockCategory
    input_ports: tuple[Port, ...]
    output_ports: tuple[Port, ...]
    control_ports: tuple[Port, ...] = ()
    settings: BlockSettings = field(default_factory=lambda: BlockSettings(entries={}))
    state: BlockState = BlockState.FRESH
