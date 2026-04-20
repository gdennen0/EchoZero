from __future__ import annotations

from echozero.application.timeline.models import Event
from echozero.testing.ma3 import SimulatedMA3Bridge


def test_simulated_ma3_bridge_is_deterministic_and_records_events():
    bridge = SimulatedMA3Bridge()

    assert bridge.connected is False
    assert bridge.connect_calls == 0
    assert bridge.disconnect_calls == 0

    bridge.on_ma3_connected()
    bridge.on_ma3_connected()
    bridge.emit("osc", {"path": "/sync/start"})
    bridge.push_event("transport", {"state": "playing"})
    bridge.push_event("transport", {"state": "stopped"})
    first = bridge.pop_event()
    second = bridge.pop_event()
    third = bridge.pop_event()
    bridge.on_ma3_disconnected()

    assert bridge.connected is False
    assert bridge.connect_calls == 2
    assert bridge.disconnect_calls == 1
    assert bridge.emitted_events == [{"kind": "osc", "payload": {"path": "/sync/start"}}]
    assert first == {"kind": "transport", "payload": {"state": "playing"}}
    assert second == {"kind": "transport", "payload": {"state": "stopped"}}
    assert third is None
    assert bridge.pending_events() == []
    assert any(command == "EZ.Ping()" for command in bridge.commands)


def test_simulated_ma3_bridge_exposes_default_track_and_event_snapshots():
    bridge = SimulatedMA3Bridge()

    tracks = bridge.list_tracks()
    events = bridge.list_track_events("tc1_tg2_tr3")

    assert [track.coord for track in tracks] == ["tc1_tg2_tr3", "tc1_tg2_tr4"]
    assert tracks[0].event_count == 2
    assert [event.event_id for event in events] == ["ma3_evt_1", "ma3_evt_2"]


def test_simulated_ma3_bridge_merge_push_preserves_existing_events_and_deduplicates():
    bridge = SimulatedMA3Bridge()

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr3",
        selected_events=[
            Event(id="dup_evt", take_id="take_1", start=1.0, end=1.5, label="Cue 1"),
            Event(id="new_evt", take_id="take_1", start=4.0, end=4.5, label="Cue 4"),
        ],
        transfer_mode="merge",
    )

    events = bridge.list_track_events("tc1_tg2_tr3")
    assert [event.label for event in events] == ["Cue 1", "Cue 2", "Cue 4"]
    assert [event.event_id for event in events[:2]] == ["ma3_evt_1", "ma3_evt_2"]
    assert events[2].event_id.startswith("tc1_tg2_tr3:evt:")


def test_simulated_ma3_bridge_overwrite_push_replaces_track_events():
    bridge = SimulatedMA3Bridge()

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr3",
        selected_events=[
            Event(id="evt_a", take_id="take_1", start=0.25, end=0.5, label="A"),
            Event(id="evt_a", take_id="take_1", start=0.75, end=1.0, label="B"),
        ],
        transfer_mode="overwrite",
    )

    events = bridge.list_track_events("tc1_tg2_tr3")
    assert [event.label for event in events] == ["A", "B"]
    assert all(event.event_id.startswith("tc1_tg2_tr3:evt:") for event in events)
