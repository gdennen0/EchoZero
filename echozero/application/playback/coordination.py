"""
Playback coordination contracts for transport and graph preparation.
Exists to define the nonblocking playback boundary before replacing runtime internals.
Connects UI/app transport commands to cached snapshots and prepared graph generations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from echozero.application.operations import OperationSnapshot
from echozero.application.playback.models import PlaybackTimingSnapshot
from echozero.application.playback.sync_delta import (
    playback_mix_signature,
    playback_structure_signature,
)
from echozero.application.playback.sync_projection import PlaybackSyncPayload


class TransportCommandAction(StrEnum):
    """High-priority transport commands accepted by playback coordinators."""

    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    SEEK = "seek"
    SCRUB_UPDATE = "scrub_update"
    SCRUB_COMMIT = "scrub_commit"


class PlaybackGenerationStatus(StrEnum):
    """Lifecycle states for prepared playback graph generations."""

    QUEUED = "queued"
    PREPARING = "preparing"
    READY = "ready"
    STALE = "stale"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class TransportCommand:
    """One app-originated transport command."""

    action: TransportCommandAction
    command_id: str = ""
    position_seconds: float | None = None
    source: str = "app"
    monotonic_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _coerce_transport_action(self.action))
        object.__setattr__(self, "command_id", str(self.command_id or "").strip())
        object.__setattr__(self, "source", str(self.source or "app").strip() or "app")
        monotonic = (
            float(self.monotonic_seconds)
            if float(self.monotonic_seconds or 0.0) > 0.0
            else time.monotonic()
        )
        object.__setattr__(self, "monotonic_seconds", monotonic)
        if self.position_seconds is not None:
            object.__setattr__(self, "position_seconds", max(0.0, float(self.position_seconds)))


@dataclass(slots=True, frozen=True)
class TransportSnapshot:
    """Cached transport timing state safe for UI polling."""

    audible_time_seconds: float
    clock_time_seconds: float
    snapshot_monotonic_seconds: float | None
    is_playing: bool
    sample_position: int = 0
    display_label: str = ""
    generation_id: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_timing_snapshot(
        cls,
        snapshot: PlaybackTimingSnapshot,
        *,
        generation_id: str | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> "TransportSnapshot":
        """Build a transport snapshot from the legacy timing snapshot."""

        return cls(
            audible_time_seconds=float(snapshot.audible_time_seconds),
            clock_time_seconds=float(snapshot.clock_time_seconds),
            snapshot_monotonic_seconds=snapshot.snapshot_monotonic_seconds,
            is_playing=bool(snapshot.is_playing),
            sample_position=int(snapshot.sample_position),
            display_label=str(snapshot.display_label or snapshot.timecode_label or ""),
            generation_id=generation_id,
            diagnostics=dict(diagnostics or {}),
        )

    def to_timing_snapshot(self) -> PlaybackTimingSnapshot:
        """Return a legacy timing snapshot for compatibility callers."""

        return PlaybackTimingSnapshot(
            audible_time_seconds=max(0.0, float(self.audible_time_seconds)),
            clock_time_seconds=max(0.0, float(self.clock_time_seconds)),
            snapshot_monotonic_seconds=self.snapshot_monotonic_seconds,
            is_playing=bool(self.is_playing),
            sample_position=max(0, int(self.sample_position)),
            display_label=str(self.display_label or ""),
            timecode_label=str(self.display_label or ""),
        )


@dataclass(slots=True, frozen=True)
class PlaybackGraphRequest:
    """Compact request for preparing one playback graph generation."""

    request_id: str
    source_signature: tuple[tuple[str, str], ...]
    mix_signature: tuple[tuple[str, str], ...]
    payload: PlaybackSyncPayload

    @classmethod
    def from_sync_payload(
        cls,
        payload: PlaybackSyncPayload,
        *,
        request_id: str,
    ) -> "PlaybackGraphRequest":
        """Create a graph request from the compact playback sync payload."""

        return cls(
            request_id=str(request_id),
            source_signature=playback_structure_signature(payload),
            mix_signature=playback_mix_signature(payload),
            payload=payload,
        )


@dataclass(slots=True, frozen=True)
class PlaybackGenerationState:
    """Prepared playback graph generation state."""

    generation_id: str
    source_signature: tuple[tuple[str, str], ...] = ()
    mix_signature: tuple[tuple[str, str], ...] = ()
    status: PlaybackGenerationStatus = PlaybackGenerationStatus.QUEUED
    stale_reason: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)


class PlaybackCoordinator(Protocol):
    """Protocol for nonblocking playback coordinators."""

    def enqueue_transport_command(self, command: TransportCommand) -> None: ...
    def enqueue_graph_prepare(self, request: PlaybackGraphRequest) -> str: ...
    def enqueue_mix_update(self, request: PlaybackGraphRequest) -> None: ...
    def latest_transport_snapshot(self) -> TransportSnapshot | None: ...
    def latest_operation_snapshot(self) -> OperationSnapshot | None: ...
    def shutdown(self) -> None: ...


def _coerce_transport_action(
    value: TransportCommandAction | str,
) -> TransportCommandAction:
    return value if isinstance(value, TransportCommandAction) else TransportCommandAction(str(value))


__all__ = [
    "PlaybackCoordinator",
    "PlaybackGenerationState",
    "PlaybackGenerationStatus",
    "PlaybackGraphRequest",
    "TransportCommand",
    "TransportCommandAction",
    "TransportSnapshot",
]
