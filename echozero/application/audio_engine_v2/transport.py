"""
Explicit transport command and state models for audio engine v2.
Exists because playback control must be data-driven instead of direct engine mutation.
Connects UI/application intents to future sample-boundary RT command application.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class TransportPlayState(Enum):
    """Coarse transport play state."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class TransportCommandKind(Enum):
    """Explicit transport command kinds accepted by the v2 command stream."""

    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    SEEK = "seek"
    SET_LOOP = "set_loop"


@dataclass(frozen=True, slots=True)
class LoopRegion:
    """One enabled transport loop region in seconds."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0.0:
            raise ValueError("Loop start must be non-negative.")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Loop end must be after loop start.")


@dataclass(frozen=True, slots=True)
class TransportState:
    """Immutable transport state owned by a playback snapshot generation."""

    play_state: TransportPlayState = TransportPlayState.STOPPED
    position_seconds: float = 0.0
    loop_region: LoopRegion | None = None
    loop_enabled: bool = False
    command_sequence: int = 0


@dataclass(frozen=True, slots=True)
class TransportCommand:
    """One explicit transport command payload."""

    kind: TransportCommandKind
    sequence: int
    position_seconds: float | None = None
    loop_region: LoopRegion | None = None
    loop_enabled: bool | None = None

    @classmethod
    def play(cls, sequence: int) -> TransportCommand:
        """Build a play command."""

        return cls(kind=TransportCommandKind.PLAY, sequence=sequence)

    @classmethod
    def pause(cls, sequence: int) -> TransportCommand:
        """Build a pause command."""

        return cls(kind=TransportCommandKind.PAUSE, sequence=sequence)

    @classmethod
    def stop(cls, sequence: int) -> TransportCommand:
        """Build a stop command."""

        return cls(kind=TransportCommandKind.STOP, sequence=sequence)

    @classmethod
    def seek(cls, sequence: int, position_seconds: float) -> TransportCommand:
        """Build a seek command."""

        return cls(
            kind=TransportCommandKind.SEEK,
            sequence=sequence,
            position_seconds=max(0.0, float(position_seconds)),
        )

    @classmethod
    def set_loop(
        cls,
        sequence: int,
        *,
        loop_region: LoopRegion | None,
        enabled: bool,
    ) -> TransportCommand:
        """Build a loop-region command."""

        return cls(
            kind=TransportCommandKind.SET_LOOP,
            sequence=sequence,
            loop_region=loop_region,
            loop_enabled=enabled,
        )


def apply_transport_command(
    state: TransportState,
    command: TransportCommand,
) -> TransportState:
    """Return the next transport state after applying one explicit command."""

    if command.kind is TransportCommandKind.PLAY:
        return replace(
            state,
            play_state=TransportPlayState.PLAYING,
            command_sequence=command.sequence,
        )
    if command.kind is TransportCommandKind.PAUSE:
        return replace(
            state,
            play_state=TransportPlayState.PAUSED,
            command_sequence=command.sequence,
        )
    if command.kind is TransportCommandKind.STOP:
        return replace(
            state,
            play_state=TransportPlayState.STOPPED,
            position_seconds=0.0,
            command_sequence=command.sequence,
        )
    if command.kind is TransportCommandKind.SEEK:
        return replace(
            state,
            position_seconds=max(0.0, float(command.position_seconds or 0.0)),
            command_sequence=command.sequence,
        )
    if command.kind is TransportCommandKind.SET_LOOP:
        return replace(
            state,
            loop_region=command.loop_region,
            loop_enabled=bool(command.loop_enabled and command.loop_region is not None),
            command_sequence=command.sequence,
        )
    raise ValueError(f"Unsupported transport command: {command.kind}")
