"""Concrete SyncService adapters for application/runtime wiring."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Protocol

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
from echozero.infrastructure.sync.ma3_adapter import event_snapshot_payload, track_snapshot_payload


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

    def list_push_track_options(self) -> list[dict[str, object]]:
        return self._list_bridge_tracks()

    def list_pull_track_options(self) -> list[dict[str, object]]:
        return self._list_bridge_tracks()

    def list_pull_source_events(self, source_track_coord: str) -> list[dict[str, object]]:
        method = self._bridge_method(
            "list_track_events", "list_ma3_track_events", "get_available_ma3_events"
        )
        if method is None:
            return []
        return [
            event_snapshot_payload(raw_event) for raw_event in method(source_track_coord) or []
        ]

    def apply_push_transfer(
        self,
        *,
        target_track_coord,
        selected_events,
        transfer_mode: str = "merge",
    ) -> None:
        callback = self._bridge_method("apply_push_transfer", "execute_push_transfer")
        if callback is None:
            raise RuntimeError("MA3 bridge does not support push apply")

        kwargs: dict[str, Any] = {
            "target_track_coord": target_track_coord,
            "selected_events": selected_events,
        }
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "transfer_mode" in parameters:
            kwargs["transfer_mode"] = transfer_mode
        elif "mode" in parameters:
            kwargs["mode"] = transfer_mode
        callback(**kwargs)

    def _list_bridge_tracks(self) -> list[dict[str, object]]:
        method = self._bridge_method("list_tracks", "list_ma3_tracks", "get_available_ma3_tracks")
        if method is None:
            return []
        return [track_snapshot_payload(raw_track) for raw_track in method() or []]

    def _bridge_method(self, *names: str):
        for name in names:
            method = getattr(self._bridge, name, None)
            if callable(method):
                return method
        return None
