"""Sync application models."""

from dataclasses import dataclass
from enum import Enum

from echozero.application.shared.enums import SyncMode


class LiveSyncState(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    ARMED_WRITE = "armed_write"
    PAUSED = "paused"


def coerce_live_sync_state(value: LiveSyncState | str) -> LiveSyncState:
    if isinstance(value, LiveSyncState):
        return value
    try:
        return LiveSyncState(str(value).strip())
    except ValueError as exc:
        allowed = ", ".join(state.value for state in LiveSyncState)
        raise ValueError(f"live_sync_state must be one of: {allowed}") from exc


@dataclass(slots=True)
class SyncState:
    mode: SyncMode = SyncMode.NONE
    connected: bool = False
    leader_follower_state: str = "standalone"
    offset_ms: float = 0.0
    target_ref: str | None = None
    health: str = "unknown"
    experimental_live_sync_enabled: bool = False
