from __future__ import annotations

import errno
import threading

import pytest

from echozero.application.timeline.models import Event
from echozero.infrastructure.osc import OscUdpSendTransport
from echozero.infrastructure.sync.ma3_osc import (
    MA3OSCBridge,
    format_ma3_lua_command,
    parse_ma3_osc_payload,
)
from echozero.infrastructure.sync.ma3_adapter import MA3SequenceSnapshot, MA3TrackSnapshot
from echozero.testing.ma3 import SimulatedMA3Bridge
from echozero.testing.ma3.simulator import _SimulatedMA3OSCServer


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
    filtered_tracks = bridge.list_tracks(timecode_no=1)
    events = bridge.list_track_events("tc1_tg2_tr3")
    timecodes = bridge.list_timecodes()

    assert [track.coord for track in tracks] == ["tc1_tg2_tr3", "tc1_tg2_tr4"]
    assert filtered_tracks == tracks
    assert [event.label for event in events] == ["Cue 1", "Cue 2"]
    assert [(timecode.number, timecode.name) for timecode in timecodes] == [(1, "Song A")]
    assert any(command.startswith("EZ.SetTarget('127.0.0.1', ") for command in bridge.commands)
    assert "EZ.GetTrackGroups(1)" in bridge.commands
    assert any(command.startswith("EZ.GetTracks(1, 2, ") for command in bridge.commands)
    assert any(command.startswith("EZ.GetEvents(1, 2, 3, ") for command in bridge.commands)
    assert tracks[0].sequence_no == 12


def test_simulated_ma3_bridge_fetches_timecodes_without_needing_prior_track_queries():
    bridge = SimulatedMA3Bridge()

    timecodes = bridge.list_timecodes()

    assert [(timecode.number, timecode.name) for timecode in timecodes] == [(1, "Song A")]
    assert any(command.startswith("EZ.SetTarget('127.0.0.1', ") for command in bridge.commands)
    assert "EZ.GetTimecodes()" in bridge.commands


def test_simulated_ma3_bridge_fetches_sequences_and_current_song_range_via_osc_commands():
    bridge = SimulatedMA3Bridge()

    sequences = bridge.list_sequences()
    current_song_range = bridge.get_current_song_sequence_range()

    assert [(sequence.number, sequence.name) for sequence in sequences] == [
        (12, "Song A"),
        (15, "Lead Stack"),
    ]
    assert current_song_range is not None
    assert current_song_range.song_label == "Song A"
    assert current_song_range.start == 12
    assert current_song_range.end == 111
    assert any(command.startswith("EZ.GetSequences(") for command in bridge.commands)
    assert "EZ.GetCurrentSongSequenceRange()" in bridge.commands


def test_simulated_ma3_bridge_aggregates_chunked_sequence_responses():
    bridge = SimulatedMA3Bridge()
    bridge.set_sequences(
        [
            MA3SequenceSnapshot(number=index, name=f"Sequence {index}", cue_count=index % 4)
            for index in range(1, 86)
        ]
    )

    sequences = bridge.list_sequences()

    assert len(sequences) == 85
    assert sequences[0].number == 1
    assert sequences[-1].number == 85
    assert any(command.startswith("EZ.GetSequences(") for command in bridge.commands)


def test_simulated_ma3_bridge_aggregates_chunked_track_responses():
    bridge = SimulatedMA3Bridge()
    bridge.set_tracks(
        [
            MA3TrackSnapshot(
                coord=f"tc1_tg2_tr{index}",
                name=f"Track {index}",
                event_count=index % 3,
                sequence_no=(200 + index),
            )
            for index in range(1, 86)
        ]
    )

    tracks = bridge.list_tracks()

    assert len(tracks) == 85
    assert tracks[0].coord == "tc1_tg2_tr1"
    assert tracks[-1].coord == "tc1_tg2_tr85"
    assert any(command.startswith("EZ.GetTracks(1, 2, ") for command in bridge.commands)


def test_simulated_ma3_bridge_connect_and_disconnect_use_production_osc_path():
    bridge = SimulatedMA3Bridge()

    bridge.on_ma3_connected()
    bridge.on_ma3_disconnected()

    assert bridge.connect_calls == 1
    assert bridge.disconnect_calls == 1
    assert "EZ.Ping()" in bridge.commands
    assert "EZ.UnhookAll()" in bridge.commands


def test_format_ma3_lua_command_wraps_and_escapes_nested_quotes():
    assert (
        format_ma3_lua_command('EZ.SetTarget("127.0.0.1", 9000)')
        == 'Lua "EZ.SetTarget(\\"127.0.0.1\\", 9000)"'
    )


def test_ma3_osc_bridge_wraps_commands_for_ma3_builtin_cmd_path():
    class _CaptureTransport:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def send(self, command: str) -> None:
            self.commands.append(command)

        def close(self) -> None:
            return None

    transport = _CaptureTransport()
    bridge = MA3OSCBridge(
        listen_host="127.0.0.1",
        listen_port=0,
        command_transport=transport,
    )

    try:
        bridge.on_ma3_connected()

        assert len(transport.commands) >= 2
        assert transport.commands[0].startswith('Lua "EZ.SetTarget(')
        assert transport.commands[1] == 'Lua "EZ.Ping()"'
    finally:
        bridge.shutdown()


def test_ma3_osc_bridge_uses_routable_target_when_listener_binds_wildcard_host():
    server = _SimulatedMA3OSCServer().start()
    bridge = MA3OSCBridge(
        listen_host="0.0.0.0",
        listen_port=0,
        command_transport=OscUdpSendTransport(*server.endpoint, path="/cmd"),
    )

    try:
        timecodes = bridge.list_timecodes()

        assert [(timecode.number, timecode.name) for timecode in timecodes] == [(1, "Song A")]
        target_command = next(
            (command for command in server.commands if command.startswith("EZ.SetTarget(")),
            "",
        )
        assert target_command
        assert "0.0.0.0" not in target_command
    finally:
        bridge.shutdown()
        server.stop()


def test_ma3_osc_bridge_can_send_raw_console_commands_for_plugin_reload():
    class _CaptureTransport:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def send(self, command: str) -> None:
            self.commands.append(command)

        def close(self) -> None:
            return None

    transport = _CaptureTransport()
    bridge = MA3OSCBridge(
        listen_host="127.0.0.1",
        listen_port=0,
        command_transport=transport,
    )

    try:
        bridge.reload_plugins()
        assert transport.commands == ["RP"]
    finally:
        bridge.shutdown()


def test_ma3_osc_bridge_falls_back_to_localhost_when_bind_host_is_unavailable(monkeypatch):
    attempts: list[tuple[str, int]] = []
    stopped = threading.Event()

    class _FakeThreadingOSCUDPServer:
        def __init__(self, server_address, dispatcher):
            host, port = server_address
            attempts.append((str(host), int(port)))
            if host == "10.255.255.1":
                raise OSError(errno.EADDRNOTAVAIL, "can't assign requested address")
            self.server_address = (str(host), int(port))
            self._thread_active = False
            self._poll = threading.Event()

        def serve_forever(self, poll_interval: float = 0.01) -> None:
            self._thread_active = True
            while self._thread_active:
                stopped.wait(timeout=poll_interval)
                if stopped.is_set():
                    break

        def shutdown(self) -> None:
            self._thread_active = False

        def server_close(self) -> None:
            self._poll.set()

    monkeypatch.setattr("echozero.infrastructure.osc.service.ThreadingOSCUDPServer", _FakeThreadingOSCUDPServer)

    bridge = MA3OSCBridge(
        listen_host="10.255.255.1",
        listen_port=0,
    )
    try:
        bridge.start()
        assert attempts == [("10.255.255.1", 0), ("127.0.0.1", 0)]
        assert bridge.is_running
        assert bridge.listener_endpoint[0] == "127.0.0.1"
    finally:
        stopped.set()
        bridge.stop()


def test_ma3_osc_bridge_recovers_missing_cmd_subtrack_once_before_retrying_add_event():
    bridge = SimulatedMA3Bridge()
    bridge.set_tracks(
        [
            MA3TrackSnapshot(coord="tc1_tg2_tr3", name="Track 3", note="Bass", event_count=2),
            MA3TrackSnapshot(coord="tc1_tg2_tr5", name="Track 5", note="Empty", event_count=0),
        ]
    )
    bridge.set_track_events(
        {
            "tc1_tg2_tr3": bridge.list_track_events("tc1_tg2_tr3"),
            "tc1_tg2_tr5": [],
        }
    )
    bridge.set_track_write_ready("tc1_tg2_tr5", ready=False)

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr5",
        selected_events=[
            Event(
                id="evt_1",
                take_id="take_1",
                start=1.0,
                end=1.1,
                cue_number=5,
                label="Kick",
            )
        ],
        transfer_mode="overwrite",
    )

    assert "EZ.CreateCmdSubTrack(1, 2, 5, 1)" in bridge.commands
    events = bridge.list_track_events("tc1_tg2_tr5")
    assert len(events) == 1
    assert events[0].label == "Kick"
    assert events[0].cmd == "Go+ Cue 5"
    assert events[0].cue_number == 5


def test_ma3_osc_bridge_preserves_float_cue_numbers_in_commands_and_snapshots():
    bridge = SimulatedMA3Bridge()
    bridge.set_tracks(
        [
            MA3TrackSnapshot(coord="tc1_tg2_tr5", name="Track 5", note="Empty", event_count=0),
        ]
    )
    bridge.set_track_events({"tc1_tg2_tr5": []})

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr5",
        selected_events=[
            Event(
                id="evt_1",
                take_id="take_1",
                start=1.0,
                end=1.1,
                cue_number=5.5,
                label="Kick",
            )
        ],
        transfer_mode="overwrite",
    )

    events = bridge.list_track_events("tc1_tg2_tr5")

    assert events[0].cmd == "Go+ Cue 5.5"
    assert events[0].cue_number == 5.5
    assert "EZ.AddEvent(1, 2, 5, 1, 'Go+ Cue 5.5', 'Kick', 5.5, 'Kick')" in bridge.commands


def test_ma3_osc_bridge_surfaces_sequence_assignment_prerequisite_when_cmd_subtrack_retry_fails():
    bridge = SimulatedMA3Bridge()
    bridge.set_tracks(
        [
            MA3TrackSnapshot(coord="tc1_tg2_tr5", name="Track 5", note="Empty", event_count=0),
        ]
    )
    bridge.set_track_events({"tc1_tg2_tr5": []})
    bridge.set_track_write_ready("tc1_tg2_tr5", ready=False)
    bridge.set_cmd_subtrack_create_blocked("tc1_tg2_tr5", blocked=True)

    with pytest.raises(
        RuntimeError,
        match="Assign a sequence to the track in MA3, then retry the push",
    ):
        bridge.apply_push_transfer(
            target_track_coord="tc1_tg2_tr5",
            selected_events=[
                Event(
                    id="evt_1",
                    take_id="take_1",
                    start=1.0,
                    end=1.1,
                    cue_number=5,
                    label="Kick",
                )
            ],
            transfer_mode="overwrite",
        )

    assert "EZ.CreateCmdSubTrack(1, 2, 5, 1)" in bridge.commands


def test_ma3_osc_bridge_creates_assigns_and_prepares_track_for_push_without_cmd_subtrack_repair():
    bridge = SimulatedMA3Bridge()
    bridge.set_track_write_ready("tc1_tg2_tr4", ready=False)

    created = bridge.create_sequence_in_current_song_range(preferred_name="Song A - Lead")
    bridge.assign_track_sequence(
        target_track_coord="tc1_tg2_tr4",
        sequence_no=created.number,
    )
    bridge.prepare_track_for_events(target_track_coord="tc1_tg2_tr4")
    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr4",
        selected_events=[
            Event(
                id="evt_1",
                take_id="take_1",
                start=0.5,
                end=0.75,
                cue_number=27,
                label="Lead",
            )
        ],
        transfer_mode="overwrite",
    )

    track = next(track for track in bridge.list_tracks() if track.coord == "tc1_tg2_tr4")
    events = bridge.list_track_events("tc1_tg2_tr4")

    assert created.number == 13
    assert created.name == "Song A - Lead"
    assert track.sequence_no == 13
    assert [event.label for event in events] == ["Lead"]
    assert [event.cmd for event in events] == ["Go+ Cue 27"]
    assert [event.cue_number for event in events] == [27]
    assert "EZ.CreateSequenceInCurrentSongRange('Song A - Lead')" in bridge.commands
    assert "EZ.AssignTrackSequence(1, 2, 4, 13)" in bridge.commands
    assert "EZ.PrepareTrackForEvents(1, 2, 4)" in bridge.commands
    assert "EZ.AddEvent(1, 2, 4, 0.5, 'Go+ Cue 27', 'Lead', 27, 'Lead')" in bridge.commands
    assert "EZ.CreateCmdSubTrack(1, 2, 4, 1)" not in bridge.commands


def test_ma3_osc_bridge_creates_next_available_sequence_via_lua_method():
    bridge = SimulatedMA3Bridge()
    existing_numbers = {sequence.number for sequence in bridge.list_sequences()}

    created = bridge.create_sequence_next_available(preferred_name="Lead Next")

    assert created.number == max(existing_numbers) + 1
    assert created.number not in existing_numbers
    assert created.name == "Lead Next"
    assert "EZ.CreateSequenceNextAvailable('Lead Next')" in bridge.commands


def test_ma3_osc_bridge_creates_timecode_track_group_and_track():
    bridge = SimulatedMA3Bridge()

    created_timecode = bridge.create_timecode_next_available(preferred_name="Song B")
    created_group = bridge.create_track_group_next_available(
        timecode_no=created_timecode.number,
        preferred_name="FX",
    )
    created_track = bridge.create_track(
        timecode_no=created_timecode.number,
        track_group_no=created_group.number,
        preferred_name="Laser",
    )

    tracks = bridge.list_tracks(
        timecode_no=created_timecode.number,
        track_group_no=created_group.number,
    )
    groups = bridge.list_track_groups(timecode_no=created_timecode.number)
    timecodes = bridge.list_timecodes()

    assert created_timecode.number == 2
    assert created_timecode.name == "Song B"
    assert created_group.number == 1
    assert created_group.name == "FX"
    assert created_group.track_count == 0
    assert created_track.coord == "tc2_tg1_tr1"
    assert created_track.name == "Laser"
    assert [(group.number, group.name) for group in groups] == [(1, "FX")]
    assert [track.coord for track in tracks] == ["tc2_tg1_tr1"]
    assert (2, "Song B") in [(timecode.number, timecode.name) for timecode in timecodes]
    assert "EZ.CreateTimecode('Song B')" in bridge.commands
    assert "EZ.CreateTrackGroup(2, 'FX')" in bridge.commands
    assert "EZ.CreateTrack(2, 1, 'Laser')" in bridge.commands


def test_ma3_osc_bridge_prepare_track_raises_track_error_for_unassigned_target():
    bridge = SimulatedMA3Bridge()

    with pytest.raises(RuntimeError, match="Track has no assigned sequence"):
        bridge.prepare_track_for_events(target_track_coord="tc1_tg2_tr4")


def test_ma3_osc_bridge_overwrite_waits_for_delayed_clear_before_rewriting_track():
    bridge = SimulatedMA3Bridge()
    bridge.set_clear_delay("tc1_tg2_tr3", seconds=0.05)

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr3",
        selected_events=[
            Event(
                id="evt_7",
                take_id="take_1",
                start=0.25,
                end=0.5,
                cue_number=7,
                label="Lead",
            )
        ],
        transfer_mode="overwrite",
    )

    events = bridge.list_track_events("tc1_tg2_tr3")

    assert [event.label for event in events] == ["Lead"]
    assert [event.cmd for event in events] == ["Go+ Cue 7"]
    assert [event.cue_number for event in events] == [7]


def test_ma3_osc_bridge_throttles_large_pushes_in_small_write_batches(monkeypatch):
    bridge = SimulatedMA3Bridge()
    sleep_calls: list[float] = []
    batch_settle_seconds = 0.12345

    monkeypatch.setattr("echozero.infrastructure.sync.ma3_osc._PUSH_WRITE_BATCH_SIZE", 3)
    monkeypatch.setattr(
        "echozero.infrastructure.sync.ma3_osc._PUSH_WRITE_BATCH_SETTLE_SECONDS",
        batch_settle_seconds,
    )
    monkeypatch.setattr(
        "echozero.infrastructure.sync.ma3_osc.sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    bridge.apply_push_transfer(
        target_track_coord="tc1_tg2_tr3",
        selected_events=[
            Event(
                id=f"evt_{index}",
                take_id="take_1",
                start=0.1 * index,
                end=(0.1 * index) + 0.05,
                cue_number=index,
                label=f"E{index}",
            )
            for index in range(1, 8)
        ],
        transfer_mode="overwrite",
    )

    assert sleep_calls.count(batch_settle_seconds) == 2
