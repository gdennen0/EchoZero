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
    assert tracks[0].sequence_no == 12
    assert tracks[1].sequence_no is None
    assert [event.event_id for event in events] == ["ma3_evt_1", "ma3_evt_2"]
    assert [event.cue_number for event in events] == [1, 2]


def test_simulated_ma3_bridge_sequence_management_is_deterministic():
    bridge = SimulatedMA3Bridge()

    current_song_range = bridge.get_current_song_sequence_range()
    created_next = bridge.create_sequence_next_available(preferred_name="Pad Stack")
    created_in_song = bridge.create_sequence_in_current_song_range(preferred_name="Song A - Lead")
    bridge.assign_track_sequence(
        target_track_coord="tc1_tg2_tr4",
        sequence_no=created_in_song.number,
    )
    bridge.prepare_track_for_events(target_track_coord="tc1_tg2_tr4")

    sequences = bridge.list_sequences()
    tracks = bridge.list_tracks()

    assert current_song_range is not None
    assert current_song_range.song_label == "Song A"
    assert current_song_range.start == 12
    assert current_song_range.end == 111
    assert created_next.number == 16
    assert created_in_song.number == 13
    assert [(sequence.number, sequence.name) for sequence in sequences] == [
        (12, "Song A"),
        (13, "Song A - Lead"),
        (15, "Lead Stack"),
        (16, "Pad Stack"),
    ]
    assert tracks[1].sequence_no == 13


def test_simulated_ma3_bridge_merge_push_preserves_existing_events_and_deduplicates():
    bridge = SimulatedMA3Bridge()

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr3",
        selected_events=[
            Event(
                id="dup_evt",
                take_id="take_1",
                start=1.0,
                end=1.5,
                cue_number=1,
                label="Cue 1",
            ),
            Event(
                id="new_evt",
                take_id="take_1",
                start=4.0,
                end=4.5,
                cue_number=4,
                label="Cue 4",
            ),
        ],
        transfer_mode="merge",
    )

    events = bridge.list_track_events("tc1_tg2_tr3")
    assert [event.label for event in events] == ["Cue 1", "Cue 2", "Cue 4"]
    assert [event.event_id for event in events[:2]] == ["ma3_evt_1", "ma3_evt_2"]
    assert [event.cue_number for event in events] == [1, 2, 4]
    assert events[2].cmd == "Go+ Cue 4"
    assert events[2].event_id.startswith("tc1_tg2_tr3:evt:")


def test_simulated_ma3_bridge_overwrite_push_replaces_track_events():
    bridge = SimulatedMA3Bridge()

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr3",
        selected_events=[
            Event(
                id="evt_a",
                take_id="take_1",
                start=0.25,
                end=0.5,
                cue_number=7,
                label="A",
            ),
            Event(
                id="evt_b",
                take_id="take_1",
                start=0.75,
                end=1.0,
                cue_number=8,
                label="B",
            ),
        ],
        transfer_mode="overwrite",
    )

    events = bridge.list_track_events("tc1_tg2_tr3")
    assert [event.label for event in events] == ["Cue 7", "Cue 8"]
    assert [event.cue_number for event in events] == [7, 8]
    assert [event.cmd for event in events] == ["Go+ Cue 7", "Go+ Cue 8"]
    assert all(event.event_id.startswith("tc1_tg2_tr3:evt:") for event in events)
