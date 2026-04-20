from __future__ import annotations

from echozero.infrastructure.sync.ma3_osc import parse_ma3_osc_payload
from echozero.testing.ma3 import SimulatedMA3Bridge


def test_parse_ma3_osc_payload_keeps_json_fields_and_pipe_containing_raw_values():
    message = parse_ma3_osc_payload(
        'type=track|change=changed|timestamp=1712860800|tc=101|tg=1|track=1|'
        'events=[{"idx":1,"time":1.25,"name":"Kick","cmd":"Go+ | Cue 5"}]|'
        'added=[{fingerprint:0.409233|Kick|Go+|Cue 5}]'
    )

    assert message.key == "track.changed"
    assert message.timestamp == 1712860800.0
    assert message.fields["tc"] == 101
    assert message.fields["events"] == [
        {
            "idx": 1,
            "time": 1.25,
            "name": "Kick",
            "cmd": "Go+ | Cue 5",
        }
    ]
    assert message.fields["added"] == "[{fingerprint:0.409233|Kick|Go+|Cue 5}]"


def test_simulated_ma3_bridge_fetches_tracks_and_events_via_osc_commands():
    bridge = SimulatedMA3Bridge()

    tracks = bridge.list_tracks()
    events = bridge.list_track_events("tc1_tg2_tr3")

    assert [track.coord for track in tracks] == ["tc1_tg2_tr3", "tc1_tg2_tr4"]
    assert [event.label for event in events] == ["Cue 1", "Cue 2"]
    assert any(command.startswith('EZ.SetTarget("127.0.0.1", ') for command in bridge.commands)
    assert "EZ.GetTrackGroups(1)" in bridge.commands
    assert "EZ.GetTracks(1, 2)" in bridge.commands
    assert any(command.startswith("EZ.GetEvents(1, 2, 3, ") for command in bridge.commands)


def test_simulated_ma3_bridge_connect_and_disconnect_use_production_osc_path():
    bridge = SimulatedMA3Bridge()

    bridge.on_ma3_connected()
    bridge.on_ma3_disconnected()

    assert bridge.connect_calls == 1
    assert bridge.disconnect_calls == 1
    assert "EZ.Ping()" in bridge.commands
    assert "EZ.UnhookAll()" in bridge.commands
