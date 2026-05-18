"""
Offline block renderer for the audio engine v2 RT graph prototype.
Exists to prove callback-shaped routing, command, and declick semantics safely.
Connects PreparedGraph parity work to deterministic non-live render tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pow
from typing import cast, TypeAlias

import numpy as np
from numpy.typing import NDArray

from echozero.application.audio_engine_v2.graph import MixParameters
from echozero.application.audio_engine_v2.rt_commands import (
    RtCommandBatch,
    RtRuntimeState,
    apply_rt_command_batch,
)
from echozero.application.audio_engine_v2.rt_graph import RtGraph, RtRouteTarget
from echozero.application.audio_engine_v2.transport import (
    TransportCommandKind,
    TransportPlayState,
)

FloatBlock: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TransitionPolicy:
    """Small v2 declick policy for block-boundary gain changes."""

    ramp_frames: int = 16

    def __post_init__(self) -> None:
        if self.ramp_frames < 0:
            raise ValueError("Ramp frames must be non-negative.")


@dataclass(frozen=True, slots=True)
class OfflineSourceBank:
    """Immutable source buffers addressed by PreparedTrack source keys."""

    buffers: dict[str, FloatBlock]

    def read(
        self,
        source_key: str,
        *,
        start_frame: int,
        frame_count: int,
        channels: int,
    ) -> FloatBlock:
        """Read one zero-padded source block."""

        source: FloatBlock = _as_2d(self.buffers[source_key])
        block: FloatBlock = np.zeros((frame_count, channels), dtype=np.float32)
        if start_frame < source.shape[0]:
            available = min(frame_count, source.shape[0] - start_frame)
            block[:available] = _adapt_channels(
                source[start_frame : start_frame + available],
                channels,
            )
        return block


@dataclass(frozen=True, slots=True)
class OfflineRenderMemory:
    """Preallocated scratch buffers reused across offline render blocks."""

    bus_buffers: tuple[FloatBlock, ...]
    hardware_buffer: FloatBlock
    transition_bus_buffers: tuple[FloatBlock, ...]
    transition_hardware_buffer: FloatBlock

    @classmethod
    def create(
        cls,
        *,
        bus_count: int,
        block_frames: int,
        max_bus_channels: int,
        hardware_channels: int,
    ) -> OfflineRenderMemory:
        """Allocate bounded scratch memory for a fixed render shape."""

        return cls(
            bus_buffers=tuple(
                np.zeros((block_frames, max_bus_channels), dtype=np.float32)
                for _ in range(bus_count)
            ),
            hardware_buffer=np.zeros(
                (block_frames, hardware_channels),
                dtype=np.float32,
            ),
            transition_bus_buffers=tuple(
                np.zeros((block_frames, max_bus_channels), dtype=np.float32)
                for _ in range(bus_count)
            ),
            transition_hardware_buffer=np.zeros(
                (block_frames, hardware_channels),
                dtype=np.float32,
            ),
        )


@dataclass(frozen=True, slots=True)
class OfflineRenderState:
    """Persistent offline callback state between rendered blocks."""

    runtime: RtRuntimeState
    frame_position: int = 0
    sample_rate: int = 44100
    previous_gains: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OfflineRenderResult:
    """One rendered hardware block and the next offline render state."""

    block: FloatBlock
    state: OfflineRenderState
    applied_sequences: tuple[int, ...] = ()
    stale_sequences: tuple[int, ...] = ()


def render_offline_block(
    state: OfflineRenderState,
    *,
    sources: OfflineSourceBank,
    memory: OfflineRenderMemory,
    policy: TransitionPolicy,
    commands: RtCommandBatch | None = None,
) -> OfflineRenderResult:
    """Render one deterministic block, applying commands at the block boundary."""

    command_result = None
    runtime = state.runtime
    previous_runtime = state.runtime
    if commands is not None:
        command_result = apply_rt_command_batch(runtime, commands)
        runtime = command_result.state
    applied_sequences = () if command_result is None else command_result.applied_sequences
    seek_frame_position = _fresh_seek_frame_position(
        commands,
        applied_sequences=applied_sequences,
        sample_rate=state.sample_rate,
    )

    _clear_memory(memory)
    if _should_crossfade_running_seek(previous_runtime, runtime, seek_frame_position, policy):
        _render_playing_block(
            previous_runtime.graph,
            state,
            sources=sources,
            memory=memory,
            policy=policy,
            force_zero_targets=False,
        )
        memory.transition_hardware_buffer[:] = memory.hardware_buffer
        _clear_primary_buffers(memory)
        seek_state = replace(state, frame_position=int(seek_frame_position or 0))
        next_gains = _render_playing_block(
            runtime.graph,
            seek_state,
            sources=sources,
            memory=memory,
            policy=policy,
            force_zero_targets=False,
        )
        memory.hardware_buffer[:] = _crossfade_blocks(
            memory.transition_hardware_buffer,
            memory.hardware_buffer,
            policy=policy,
        )
        next_position = int(seek_frame_position or 0) + memory.hardware_buffer.shape[0]
    elif (
        seek_frame_position is not None
        and runtime.transport.play_state is not TransportPlayState.PLAYING
    ):
        next_gains = {}
        next_position = int(seek_frame_position)
    elif _should_crossfade_graph_commit(previous_runtime, runtime, policy):
        _render_playing_block(
            previous_runtime.graph,
            state,
            sources=sources,
            memory=memory,
            policy=policy,
            force_zero_targets=False,
        )
        memory.transition_hardware_buffer[:] = memory.hardware_buffer
        _clear_primary_buffers(memory)
        next_gains = _render_playing_block(
            runtime.graph,
            state,
            sources=sources,
            memory=memory,
            policy=policy,
            force_zero_targets=False,
        )
        memory.hardware_buffer[:] = _crossfade_blocks(
            memory.transition_hardware_buffer,
            memory.hardware_buffer,
            policy=policy,
        )
        next_position = state.frame_position + memory.hardware_buffer.shape[0]
    elif runtime.transport.play_state is TransportPlayState.PLAYING:
        render_state = (
            replace(state, frame_position=int(seek_frame_position))
            if seek_frame_position is not None
            else state
        )
        next_gains = _render_playing_block(
            runtime.graph,
            render_state,
            sources=sources,
            memory=memory,
            policy=policy,
            force_zero_targets=False,
        )
        next_position = render_state.frame_position + memory.hardware_buffer.shape[0]
    elif previous_runtime.transport.play_state is TransportPlayState.PLAYING:
        next_gains = _render_playing_block(
            previous_runtime.graph,
            state,
            sources=sources,
            memory=memory,
            policy=policy,
            force_zero_targets=True,
        )
        next_position = (
            0
            if runtime.transport.play_state is TransportPlayState.STOPPED
            else state.frame_position
        )
    else:
        next_gains = {}
        next_position = state.frame_position

    next_state = replace(
        state,
        runtime=runtime,
        frame_position=next_position,
        previous_gains=next_gains,
    )
    return OfflineRenderResult(
        block=memory.hardware_buffer.copy(),
        state=next_state,
        applied_sequences=applied_sequences,
        stale_sequences=() if command_result is None else command_result.stale_sequences,
    )


def _fresh_seek_frame_position(
    commands: RtCommandBatch | None,
    *,
    applied_sequences: tuple[int, ...],
    sample_rate: int,
) -> int | None:
    if commands is None or not applied_sequences:
        return None
    applied = set(applied_sequences)
    for command in sorted(commands.commands, key=lambda item: item.sequence, reverse=True):
        transport_command = command.transport_command
        if command.sequence not in applied or transport_command is None:
            continue
        if transport_command.kind is not TransportCommandKind.SEEK:
            continue
        seconds = max(0.0, float(transport_command.position_seconds or 0.0))
        return int(round(seconds * max(1, int(sample_rate))))
    return None


def _render_playing_block(
    graph: RtGraph,
    state: OfflineRenderState,
    *,
    sources: OfflineSourceBank,
    memory: OfflineRenderMemory,
    policy: TransitionPolicy,
    force_zero_targets: bool,
) -> dict[str, float]:
    track_solo_active = any(track.mix.soloed for track in graph.tracks)
    next_gains: dict[str, float] = {}
    for track in graph.tracks:
        gain = (
            0.0
            if force_zero_targets
            else _effective_gain(
                track.mix,
                solo_active=track_solo_active,
            )
        )
        node_key = f"track:{track.track_id}"
        next_gains[node_key] = gain
        source_block = sources.read(
            track.source_key,
            start_frame=state.frame_position,
            frame_count=memory.hardware_buffer.shape[0],
            channels=max(1, track.channels),
        )
        mixed = _apply_gain_ramp(
            source_block,
            previous_gain=state.previous_gains.get(node_key, 0.0),
            target_gain=gain,
            policy=policy,
        )
        _route_block(mixed, track.route_targets, memory=memory)

    for bus_index in graph.bus_render_order:
        bus = graph.buses[bus_index]
        node_key = f"bus:{bus.bus_id}"
        gain = 0.0 if force_zero_targets else _mix_gain(bus.mix)
        next_gains[node_key] = 0.0 if bus.mix.muted else gain
        bus_block = _apply_gain_ramp(
            memory.bus_buffers[bus_index].copy(),
            previous_gain=state.previous_gains.get(node_key, 0.0),
            target_gain=next_gains[node_key],
            policy=policy,
        )
        memory.bus_buffers[bus_index][:] = 0.0
        _route_block(bus_block, bus.route_targets, memory=memory)
    return next_gains


def _route_block(
    block: FloatBlock,
    targets: tuple[RtRouteTarget, ...],
    *,
    memory: OfflineRenderMemory,
) -> None:
    for target in targets:
        if target.bus_index is not None:
            _mix_into(
                memory.bus_buffers[target.bus_index],
                _adapt_channels(block, memory.bus_buffers[target.bus_index].shape[1]),
            )
        elif target.hardware_output is not None:
            start = target.hardware_output.first_channel - 1
            stop = target.hardware_output.last_channel
            span_channels = stop - start
            if stop <= memory.hardware_buffer.shape[1]:
                memory.hardware_buffer[:, start:stop] += _adapt_channels(
                    block,
                    span_channels,
                )


def _should_crossfade_graph_commit(
    previous_runtime: RtRuntimeState,
    runtime: RtRuntimeState,
    policy: TransitionPolicy,
) -> bool:
    if policy.ramp_frames <= 0:
        return False
    return (
        previous_runtime.transport.play_state is TransportPlayState.PLAYING
        and runtime.transport.play_state is TransportPlayState.PLAYING
        and previous_runtime.graph.identity_full_hash != runtime.graph.identity_full_hash
    )


def _should_crossfade_running_seek(
    previous_runtime: RtRuntimeState,
    runtime: RtRuntimeState,
    seek_frame_position: int | None,
    policy: TransitionPolicy,
) -> bool:
    if seek_frame_position is None or policy.ramp_frames <= 0:
        return False
    return (
        previous_runtime.transport.play_state is TransportPlayState.PLAYING
        and runtime.transport.play_state is TransportPlayState.PLAYING
    )


def _crossfade_blocks(
    previous_block: FloatBlock,
    next_block: FloatBlock,
    *,
    policy: TransitionPolicy,
) -> FloatBlock:
    ramp_frames = min(policy.ramp_frames, previous_block.shape[0])
    if ramp_frames <= 0:
        return next_block
    previous_gain: FloatBlock = np.zeros((previous_block.shape[0], 1), dtype=np.float32)
    next_gain: FloatBlock = np.ones((next_block.shape[0], 1), dtype=np.float32)
    previous_gain[:ramp_frames, 0] = np.linspace(
        1.0,
        0.0,
        ramp_frames + 1,
        endpoint=True,
        dtype=np.float32,
    )[1:]
    next_gain[:ramp_frames, 0] = np.linspace(
        0.0,
        1.0,
        ramp_frames + 1,
        endpoint=True,
        dtype=np.float32,
    )[1:]
    return previous_block * previous_gain + next_block * next_gain


def _apply_gain_ramp(
    block: FloatBlock,
    *,
    previous_gain: float,
    target_gain: float,
    policy: TransitionPolicy,
) -> FloatBlock:
    if block.size == 0 or previous_gain == target_gain:
        return block * np.float32(target_gain)
    ramp_frames = min(policy.ramp_frames, block.shape[0])
    gains: FloatBlock = np.full((block.shape[0], 1), target_gain, dtype=np.float32)
    if ramp_frames > 0:
        gains[:ramp_frames, 0] = np.linspace(
            previous_gain,
            target_gain,
            ramp_frames + 1,
            endpoint=True,
            dtype=np.float32,
        )[1:]
    return block * gains


def _effective_gain(mix: MixParameters, *, solo_active: bool) -> float:
    if mix.muted or (solo_active and not mix.soloed):
        return 0.0
    return _mix_gain(mix)


def _mix_gain(mix: MixParameters) -> float:
    return float(pow(10.0, mix.gain_db / 20.0))


def _mix_into(destination: FloatBlock, block: FloatBlock) -> None:
    destination[:, : block.shape[1]] += block


def _clear_memory(memory: OfflineRenderMemory) -> None:
    _clear_primary_buffers(memory)
    memory.transition_hardware_buffer[:] = 0.0
    for bus_buffer in memory.transition_bus_buffers:
        bus_buffer[:] = 0.0


def _clear_primary_buffers(memory: OfflineRenderMemory) -> None:
    memory.hardware_buffer[:] = 0.0
    for bus_buffer in memory.bus_buffers:
        bus_buffer[:] = 0.0


def _as_2d(buffer: FloatBlock) -> FloatBlock:
    array: FloatBlock = np.asarray(buffer, dtype=np.float32)
    if array.ndim == 1:
        return cast(FloatBlock, array.reshape((-1, 1)))
    if array.ndim != 2:
        raise ValueError("Offline source buffers must be mono or channel-major 2D.")
    return array


def _adapt_channels(block: FloatBlock, channels: int) -> FloatBlock:
    if block.shape[1] == channels:
        return block
    if block.shape[1] == 1:
        return cast(FloatBlock, np.repeat(block, channels, axis=1))
    if channels == 1:
        return cast(
            FloatBlock,
            np.mean(block, axis=1, dtype=np.float32).reshape((-1, 1)),
        )
    adapted: FloatBlock = np.zeros((block.shape[0], channels), dtype=np.float32)
    copy_channels = min(channels, block.shape[1])
    adapted[:, :copy_channels] = block[:, :copy_channels]
    return adapted
