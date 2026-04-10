from __future__ import annotations

import pytest

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter
from echozero.application.sync.models import SyncState
from echozero.application.transport.models import TransportState


class _Bridge:
    def __init__(self):
        self.connected_calls = 0
        self.disconnected_calls = 0

    def on_ma3_connected(self) -> None:
        self.connected_calls += 1

    def on_ma3_disconnected(self) -> None:
        self.disconnected_calls += 1


class _FailingBridge(_Bridge):
    def on_ma3_connected(self) -> None:
        raise RuntimeError("bridge down")


def test_in_memory_sync_service_mode_connection_and_offset_alignment():
    service = InMemorySyncService()

    state = service.set_mode(SyncMode.MA3)
    assert state.mode == SyncMode.MA3

    state = service.connect()
    assert state.connected is True
    assert state.health == "healthy"

    state.offset_ms = 250.0
    aligned = service.align_transport(TransportState(playhead=10.0))
    assert aligned.playhead == pytest.approx(10.25)

    state = service.disconnect()
    assert state.connected is False
    assert state.mode == SyncMode.NONE
    assert state.health == "offline"


def test_ma3_sync_adapter_delegates_connect_disconnect_to_bridge():
    bridge = _Bridge()
    service = MA3SyncAdapter(bridge, target_ref="show_manager")

    state = service.get_state()
    assert state.mode == SyncMode.MA3
    assert state.target_ref == "show_manager"

    state = service.connect()
    assert state.connected is True
    assert state.health == "healthy"
    assert bridge.connected_calls == 1

    state = service.disconnect()
    assert state.connected is False
    assert state.mode == SyncMode.NONE
    assert state.health == "offline"
    assert bridge.disconnected_calls == 1


def test_ma3_sync_adapter_connect_failure_propagates_and_marks_error_health():
    service = MA3SyncAdapter(_FailingBridge())

    with pytest.raises(RuntimeError, match="bridge down"):
        service.connect()

    state = service.get_state()
    assert state.connected is False
    assert state.health == "error"


def test_ma3_sync_adapter_align_transport_applies_offset_only_when_connected():
    bridge = _Bridge()
    service = MA3SyncAdapter(bridge, state=SyncState(mode=SyncMode.MA3, connected=False, offset_ms=120.0))

    transport = TransportState(playhead=3.0)
    assert service.align_transport(transport).playhead == pytest.approx(3.0)

    service.connect()
    aligned = service.align_transport(transport)
    assert aligned.playhead == pytest.approx(3.12)

    # Should clamp at zero if offset would push negative.
    state = service.get_state()
    state.offset_ms = -5000.0
    aligned = service.align_transport(TransportState(playhead=1.0))
    assert aligned.playhead == 0.0
