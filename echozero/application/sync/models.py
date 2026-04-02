"""Sync application models."""

from dataclasses import dataclass

from echozero.application.shared.enums import SyncMode


@dataclass(slots=True)
class SyncState:
    mode: SyncMode = SyncMode.NONE
    connected: bool = False
    leader_follower_state: str = "standalone"
    offset_ms: float = 0.0
    target_ref: str | None = None
    health: str = "unknown"
