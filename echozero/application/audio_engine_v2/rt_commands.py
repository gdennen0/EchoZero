"""
RT command batch concepts for audio engine v2.
Exists because callback-visible changes must apply at block or sample boundaries.
Connects immutable graph/transport values to the non-live offline renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from echozero.application.audio_engine_v2.graph import MixParameters, PreparedGraph
from echozero.application.audio_engine_v2.rt_graph import RtGraph, prepare_rt_graph
from echozero.application.audio_engine_v2.transport import (
    TransportCommand,
    TransportState,
    apply_transport_command,
)


class RtCommandKind(Enum):
    """Command kinds accepted by the prototype RT command lane."""

    COMMIT_GRAPH = "commit_graph"
    TRANSPORT = "transport"
    TRACK_MIX = "track_mix"


@dataclass(frozen=True, slots=True)
class RtCommand:
    """One callback-boundary command with a monotonic sequence."""

    kind: RtCommandKind
    sequence: int
    graph: PreparedGraph | None = None
    transport_command: TransportCommand | None = None
    track_id: str | None = None
    mix: MixParameters | None = None

    @classmethod
    def commit_graph(cls, sequence: int, graph: PreparedGraph) -> RtCommand:
        """Build a graph commit command."""

        return cls(kind=RtCommandKind.COMMIT_GRAPH, sequence=sequence, graph=graph)

    @classmethod
    def transport(cls, command: TransportCommand) -> RtCommand:
        """Build a transport command wrapper."""

        return cls(
            kind=RtCommandKind.TRANSPORT,
            sequence=command.sequence,
            transport_command=command,
        )

    @classmethod
    def track_mix(
        cls,
        sequence: int,
        *,
        track_id: str,
        mix: MixParameters,
    ) -> RtCommand:
        """Build a track mix edit command."""

        return cls(
            kind=RtCommandKind.TRACK_MIX,
            sequence=sequence,
            track_id=track_id,
            mix=mix,
        )


@dataclass(frozen=True, slots=True)
class RtCommandBatch:
    """A bounded tuple of commands applied at the next render boundary."""

    commands: tuple[RtCommand, ...]

    def __post_init__(self) -> None:
        if len(self.commands) > 64:
            raise ValueError("RT command batches are bounded to 64 commands.")


@dataclass(frozen=True, slots=True)
class RtRuntimeState:
    """Callback-visible immutable runtime state for the offline prototype."""

    graph: RtGraph
    transport: TransportState
    command_sequence: int = 0


@dataclass(frozen=True, slots=True)
class RtCommandResult:
    """Result of reducing a command batch at a render boundary."""

    state: RtRuntimeState
    applied_sequences: tuple[int, ...]
    stale_sequences: tuple[int, ...]


def apply_rt_command_batch(
    state: RtRuntimeState,
    batch: RtCommandBatch,
) -> RtCommandResult:
    """Apply fresh RT commands in sequence order and report stale commands."""

    next_state = state
    applied: list[int] = []
    stale: list[int] = []
    for command in sorted(batch.commands, key=lambda item: item.sequence):
        if command.sequence <= next_state.command_sequence:
            stale.append(command.sequence)
            continue
        if _is_stale_transport_payload(next_state, command):
            next_state = replace(next_state, command_sequence=command.sequence)
            stale.append(command.sequence)
            continue
        next_state = _apply_fresh_command(next_state, command)
        applied.append(command.sequence)
    return RtCommandResult(
        state=next_state,
        applied_sequences=tuple(applied),
        stale_sequences=tuple(stale),
    )


def _is_stale_transport_payload(state: RtRuntimeState, command: RtCommand) -> bool:
    if command.kind is not RtCommandKind.TRANSPORT:
        return False
    if command.transport_command is None:
        raise ValueError("Transport command is missing a transport payload.")
    command_sequence = int(command.transport_command.sequence)
    transport_sequence = int(state.transport.command_sequence)
    return command_sequence <= transport_sequence


def _apply_fresh_command(state: RtRuntimeState, command: RtCommand) -> RtRuntimeState:
    if command.kind is RtCommandKind.COMMIT_GRAPH:
        if command.graph is None:
            raise ValueError("Graph commit command is missing a PreparedGraph.")
        return replace(
            state,
            graph=prepare_rt_graph(command.graph),
            command_sequence=command.sequence,
        )
    if command.kind is RtCommandKind.TRANSPORT:
        if command.transport_command is None:
            raise ValueError("Transport command is missing a transport payload.")
        return replace(
            state,
            transport=apply_transport_command(state.transport, command.transport_command),
            command_sequence=command.sequence,
        )
    if command.kind is RtCommandKind.TRACK_MIX:
        return replace(
            state,
            graph=_replace_rt_track_mix(state.graph, command),
            command_sequence=command.sequence,
        )
    raise ValueError(f"Unsupported RT command kind: {command.kind}")


def _replace_rt_track_mix(graph: RtGraph, command: RtCommand) -> RtGraph:
    if command.track_id is None or command.mix is None:
        raise ValueError("Track mix command requires track id and mix parameters.")
    replaced = []
    matched = False
    for track in graph.tracks:
        if track.track_id == command.track_id:
            matched = True
            replaced.append(replace(track, mix=command.mix))
        else:
            replaced.append(track)
    if not matched:
        raise ValueError(f"RT track not found: {command.track_id}")
    return replace(graph, tracks=tuple(replaced))
