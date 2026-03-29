"""
Domain types: Frozen dataclasses for EchoZero's core entities.
Exists because every concept in the pipeline needs a typed, immutable representation (FP2).
All types are value objects or entities — no behavior, no side effects, no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
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


class BlockSettings:
    """Immutable settings dict for a block.

    Usage:
        settings = BlockSettings({"threshold": 0.3})
        settings["threshold"]       # 0.3
        settings.get("threshold")   # 0.3
        settings["threshold"] = 1   # TypeError
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(data or {})))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("BlockSettings is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("BlockSettings is immutable")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BlockSettings is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BlockSettings is immutable")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockSettings):
            return self._data == other._data
        if isinstance(other, dict):
            return dict(self._data) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._data.items())))

    def __repr__(self) -> str:
        return f"BlockSettings({dict(self._data)})"


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
    settings: BlockSettings = field(default_factory=lambda: BlockSettings({}))
    state: BlockState = BlockState.FRESH
