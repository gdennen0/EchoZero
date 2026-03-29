"""
Domain enums: Classification types for ports, blocks, and execution state.
Exists because stringly-typed interfaces are banned (STYLE.md) — use enums everywhere.
Referenced by all domain types and the execution engine.
"""

from enum import Enum, auto


class PortType(Enum):
    """Classifies what kind of data flows through a port."""

    AUDIO = auto()
    EVENT = auto()
    OSC = auto()
    CONTROL = auto()
    WAVEFORM = auto()


class Direction(Enum):
    """Specifies whether a port accepts or produces data."""

    INPUT = auto()
    OUTPUT = auto()
    CONTROL = auto()


class BlockState(Enum):
    """Tracks whether a block's outputs are current or need re-execution."""

    FRESH = auto()
    STALE = auto()
    UPSTREAM_ERROR = auto()
    ERROR = auto()


class BlockCategory(Enum):
    """Classifies a block's execution semantics: pure transform, manual, or ephemeral."""

    PROCESSOR = auto()
    WORKSPACE = auto()
    PLAYBACK = auto()
