from __future__ import annotations

import pytest

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter
from echozero.application.sync.models import SyncState
from echozero.application.timeline.models import Event
from echozero.application.transport.models import TransportState
from echozero.testing.ma3 import SimulatedMA3Bridge


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
    service = MA3SyncAdapter(
        bridge, state=SyncState(mode=SyncMode.MA3, connected=False, offset_ms=120.0)
    )

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


def test_ma3_sync_adapter_exposes_bridge_track_and_event_snapshots():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    push_tracks = service.list_push_track_options()
    pull_tracks = service.list_pull_track_options()
    pull_events = service.list_pull_source_events("tc1_tg2_tr3")

    assert [track["coord"] for track in push_tracks] == ["tc1_tg2_tr3", "tc1_tg2_tr4"]
    assert push_tracks == pull_tracks
    assert [event["event_id"] for event in pull_events] == ["ma3_evt_1", "ma3_evt_2"]
    assert pull_events[0]["label"] == "Cue 1"


def test_ma3_sync_adapter_apply_push_transfer_updates_bridge_snapshot():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    service.apply_push_transfer(
        target_track_coord="tc1_tg2_tr4",
        selected_events=[
            Event(id="evt_a", take_id="take_a", start=0.5, end=0.75, label="A"),
            Event(id="evt_b", take_id="take_a", start=1.0, end=1.25, label="B"),
        ],
        transfer_mode="overwrite",
    )

    events = bridge.list_track_events("tc1_tg2_tr4")
    assert [event.label for event in events] == ["A", "B"]
    assert all(event.event_id.startswith("tc1_tg2_tr4:evt:") for event in events)
    assert bridge.emitted_events[-1] == {
        "kind": "transfer.push_applied",
        "payload": {
            "target_track_coord": "tc1_tg2_tr4",
            "transfer_mode": "overwrite",
            "selected_count": 2,
        },
    }


def test_ma3_sync_adapter_apply_push_transfer_raises_when_bridge_cannot_execute():
    bridge = _Bridge()
    service = MA3SyncAdapter(bridge)

    with pytest.raises(RuntimeError, match="MA3 bridge does not support push apply"):
        service.apply_push_transfer(
            target_track_coord="tc1_tg2_tr3",
            selected_events=[],
        )
