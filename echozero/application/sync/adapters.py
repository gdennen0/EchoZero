"""Concrete SyncService adapters for application/runtime wiring."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState


class MA3SyncBridge(Protocol):
    """Minimal bridge contract expected from MA3/show-manager runtime layer."""

    def on_ma3_connected(self) -> None: ...

    def on_ma3_disconnected(self) -> None: ...


class InMemorySyncService(SyncService):
    """Simple stateful implementation used by demos/tests."""

    def __init__(self, state: SyncState | None = None):
        self._state = state or SyncState()

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        self._state.connected = True
        self._state.health = "healthy"
        return self._state

    def disconnect(self) -> SyncState:
        self._state.connected = False
        self._state.mode = SyncMode.NONE
        self._state.health = "offline"
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        if not self._state.connected:
            return transport
        if self._state.offset_ms == 0.0:
            return transport

        shifted = max(0.0, transport.playhead + (self._state.offset_ms / 1000.0))
        return replace(transport, playhead=shifted)


class MA3SyncAdapter(SyncService):
    """App-layer SyncService adapter over MA3/show-manager bridge callbacks."""

    def __init__(
        self,
        bridge: MA3SyncBridge,
        *,
        target_ref: str = "show_manager",
        state: SyncState | None = None,
    ):
        self._bridge = bridge
        self._state = state or SyncState(mode=SyncMode.MA3, connected=False, target_ref=target_ref)

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        if self._state.mode == SyncMode.NONE:
            self._state.mode = SyncMode.MA3

        try:
            self._bridge.on_ma3_connected()
        except Exception:
            self._state.connected = False
            self._state.health = "error"
            raise

        self._state.connected = True
        self._state.health = "healthy"
        return self._state

    def disconnect(self) -> SyncState:
        try:
            self._bridge.on_ma3_disconnected()
        finally:
            self._state.connected = False
            self._state.mode = SyncMode.NONE
            self._state.health = "offline"
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        if not self._state.connected:
            return transport
        if self._state.offset_ms == 0.0:
            return transport

        shifted = max(0.0, transport.playhead + (self._state.offset_ms / 1000.0))
        return replace(transport, playhead=shifted)
