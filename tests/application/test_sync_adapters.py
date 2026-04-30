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


class _ConsoleBridge(_Bridge):
    def __init__(self):
        super().__init__()
        self.console_commands: list[str] = []

    def send_console_command(self, command: str) -> None:
        self.console_commands.append(str(command))


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
    filtered_tracks = service.list_push_track_options(timecode_no=1)
    pull_tracks = service.list_pull_track_options()
    pull_events = service.list_pull_source_events("tc1_tg2_tr3")
    timecodes = service.list_timecodes()

    assert [track["coord"] for track in push_tracks] == ["tc1_tg2_tr3", "tc1_tg2_tr4"]
    assert filtered_tracks == push_tracks
    assert push_tracks == pull_tracks
    assert push_tracks[0]["timecode_name"] == "Song A"
    assert push_tracks[0]["sequence_no"] == 12
    assert push_tracks[1]["sequence_no"] is None
    assert [event["event_id"] for event in pull_events] == ["ma3_evt_1", "ma3_evt_2"]
    assert pull_events[0]["label"] == "Cue 1"
    assert pull_events[0]["cue_number"] == 1
    assert timecodes == [{"number": 1, "name": "Song A"}]


def test_ma3_sync_adapter_preserves_shared_cue_metadata_fields_when_present():
    class _MetadataBridge(_Bridge):
        def list_tracks(self, *, timecode_no=None):
            del timecode_no
            return []

        def list_timecodes(self):
            return []

        def list_track_groups(self, *, timecode_no):
            del timecode_no
            return []

        def list_track_events(self, track_coord: str):
            del track_coord
            return [
                {
                    "event_id": "ma3_evt_1",
                    "name": "Verse",
                    "start": 1.0,
                    "cue_number": 11,
                    "cue_ref": "Q11A",
                    "color": "#ffaa00",
                    "notes": "Imported exactly",
                    "payload_ref": "payload://ma3_evt_1",
                }
            ]

        def list_sequences(self, *, start_no=None, end_no=None):
            del start_no, end_no
            return []

        def get_current_song_sequence_range(self):
            return None

        def assign_track_sequence(self, *, target_track_coord: str, sequence_no: int) -> None:
            del target_track_coord, sequence_no

        def create_sequence_next_available(self, *, preferred_name=None):
            del preferred_name
            return {"number": 1, "name": "Seq"}

        def create_sequence_in_current_song_range(self, *, preferred_name=None):
            del preferred_name
            return {"number": 1, "name": "Seq"}

        def create_timecode_next_available(self, *, preferred_name=None):
            del preferred_name
            return {"number": 1, "name": "Song"}

        def create_track_group_next_available(self, *, timecode_no: int, preferred_name=None):
            del timecode_no, preferred_name
            return {"number": 1, "name": "Group"}

        def create_track(self, *, timecode_no: int, track_group_no: int, preferred_name=None):
            del timecode_no, track_group_no, preferred_name
            return {"coord": "tc1_tg1_tr1", "name": "Track"}

        def prepare_track_for_events(self, *, target_track_coord: str) -> None:
            del target_track_coord

        def send_console_command(self, command: str) -> None:
            del command

        def reload_plugins(self) -> None:
            return None

        def apply_push_transfer(
            self,
            *,
            target_track_coord: str,
            selected_events: list[object],
            transfer_mode: str = "merge",
        ) -> None:
            del target_track_coord, selected_events, transfer_mode

    service = MA3SyncAdapter(_MetadataBridge())

    pull_events = service.list_pull_source_events("tc1_tg2_tr3")

    assert pull_events == [
        {
            "event_id": "ma3_evt_1",
            "label": "Verse",
            "start": 1.0,
            "end": None,
            "cue_number": 11,
            "cue_ref": "Q11A",
            "color": "#ffaa00",
            "notes": "Imported exactly",
            "payload_ref": "payload://ma3_evt_1",
        }
    ]


def test_ma3_sync_adapter_exposes_bridge_sequence_snapshots_and_current_song_range():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    sequences = service.list_sequences()
    current_song_range = service.get_current_song_sequence_range()

    assert [(sequence["number"], sequence["name"]) for sequence in sequences] == [
        (12, "Song A"),
        (15, "Lead Stack"),
    ]
    assert current_song_range == {
        "song_label": "Song A",
        "start": 12,
        "end": 111,
    }


def test_ma3_sync_adapter_assigns_creates_and_prepares_track_sequences():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    created = service.create_sequence_next_available(preferred_name="Lead Next")
    service.assign_track_sequence(
        target_track_coord="tc1_tg2_tr4",
        sequence_no=created["number"],
    )
    service.prepare_track_for_events(target_track_coord="tc1_tg2_tr4")

    tracks = service.list_push_track_options()

    assert created["name"] == "Lead Next"
    assert any(track["sequence_no"] == created["number"] for track in tracks if track["coord"] == "tc1_tg2_tr4")


def test_ma3_sync_adapter_creates_sequence_in_current_song_range():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    created = service.create_sequence_in_current_song_range(preferred_name="Song A Layer")
    sequences = service.list_sequences()

    assert created["number"] == 13
    assert created["name"] == "Song A Layer"
    assert any(
        sequence["number"] == created["number"] and sequence["name"] == "Song A Layer"
        for sequence in sequences
    )


def test_ma3_sync_adapter_creates_timecode_track_group_and_track():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    created_timecode = service.create_timecode_next_available(preferred_name="Song B")
    created_group = service.create_track_group_next_available(
        timecode_no=created_timecode["number"],
        preferred_name="FX",
    )
    created_track = service.create_track(
        timecode_no=created_timecode["number"],
        track_group_no=created_group["number"],
        preferred_name="Laser",
    )

    timecodes = service.list_timecodes()
    groups = service.list_track_groups(timecode_no=created_timecode["number"])
    tracks = service.list_push_track_options(
        timecode_no=created_timecode["number"],
        track_group_no=created_group["number"],
    )

    assert created_timecode == {"number": 2, "name": "Song B"}
    assert created_group == {"number": 1, "name": "FX", "track_count": 0}
    assert created_track["coord"] == "tc2_tg1_tr1"
    assert created_track["name"] == "Laser"
    assert (2, "Song B") in [(timecode["number"], timecode["name"]) for timecode in timecodes]
    assert [(group["number"], group["name"]) for group in groups] == [(1, "FX")]
    assert [track["coord"] for track in tracks] == ["tc2_tg1_tr1"]


def test_ma3_sync_adapter_reloads_plugins_through_console_command_capability():
    bridge = _ConsoleBridge()
    service = MA3SyncAdapter(bridge)

    service.reload_plugins()

    assert bridge.console_commands == ["RP"]


class _EventBridge(_Bridge):
    def list_track_events(self, source_track_coord: str):
        assert source_track_coord == "tc1_tg2_tr3"
        return [
            {
                "id": "ma3_evt_1",
                "name": "Go+",
                "time": 1.25,
                "duration": 0,
            }
        ]


def test_ma3_sync_adapter_treats_zero_duration_ma3_events_as_one_shots():
    service = MA3SyncAdapter(_EventBridge())

    pull_events = service.list_pull_source_events("tc1_tg2_tr3")

    assert pull_events == [
        {
            "event_id": "ma3_evt_1",
            "label": "Go+",
            "start": 1.25,
            "end": None,
            "cue_number": None,
            "cue_ref": None,
            "color": None,
            "notes": None,
            "payload_ref": None,
        }
    ]


class _PrefixedCueRefBridge(_Bridge):
    def list_track_events(self, source_track_coord: str):
        assert source_track_coord == "tc1_tg2_tr3"
        return [
            {
                "id": "ma3_evt_1",
                "name": "Q11A Verse",
                "time": 1.25,
                "cueNo": 11,
        }
    ]


def test_ma3_sync_adapter_infers_cue_ref_from_prefixed_label_when_bridge_omits_explicit_field():
    service = MA3SyncAdapter(_PrefixedCueRefBridge())

    pull_events = service.list_pull_source_events("tc1_tg2_tr3")

    assert pull_events == [
        {
            "event_id": "ma3_evt_1",
            "label": "Verse",
            "start": 1.25,
            "end": None,
            "cue_number": 11,
            "cue_ref": "Q11A",
            "color": None,
            "notes": None,
            "payload_ref": None,
        }
    ]


class _FloatPrefixedCueRefBridge(_Bridge):
    def list_track_events(self, source_track_coord: str):
        assert source_track_coord == "tc1_tg2_tr3"
        return [
            {
                "id": "ma3_evt_1",
                "name": "Q11.5 Verse",
                "time": 1.25,
                "cueNo": 11.5,
            }
        ]


def test_ma3_sync_adapter_preserves_float_cue_numbers_and_infers_float_cue_refs():
    service = MA3SyncAdapter(_FloatPrefixedCueRefBridge())

    pull_events = service.list_pull_source_events("tc1_tg2_tr3")

    assert pull_events == [
        {
            "event_id": "ma3_evt_1",
            "label": "Verse",
            "start": 1.25,
            "end": None,
            "cue_number": 11.5,
            "cue_ref": "Q11.5",
            "color": None,
            "notes": None,
            "payload_ref": None,
        }
    ]


def test_ma3_sync_adapter_apply_push_transfer_updates_bridge_snapshot():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    service.apply_push_transfer(
        target_track_coord="tc1_tg2_tr4",
        selected_events=[
            Event(
                id="evt_a",
                take_id="take_a",
                start=0.5,
                end=0.75,
                cue_number=7,
                label="A",
            ),
            Event(
                id="evt_b",
                take_id="take_a",
                start=1.0,
                end=1.25,
                cue_number=8,
                label="B",
            ),
        ],
        transfer_mode="overwrite",
    )

    events = bridge.list_track_events("tc1_tg2_tr4")
    assert [event.label for event in events] == ["A", "B"]
    assert [event.cue_number for event in events] == [7, 8]
    assert [event.cmd for event in events] == ["Go+ Cue 7", "Go+ Cue 8"]
    assert all(event.event_id.startswith("tc1_tg2_tr4:evt:") for event in events)
    assert bridge.emitted_events[-1] == {
        "kind": "transfer.push_applied",
        "payload": {
            "target_track_coord": "tc1_tg2_tr4",
            "transfer_mode": "overwrite",
            "selected_count": 2,
        },
    }


def test_ma3_sync_adapter_apply_push_transfer_applies_start_offset_seconds():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    service.apply_push_transfer(
        target_track_coord="tc1_tg2_tr4",
        selected_events=[
            Event(
                id="evt_a",
                take_id="take_a",
                start=1.5,
                end=1.75,
                cue_number=7,
                label="A",
            ),
            Event(
                id="evt_b",
                take_id="take_a",
                start=2.0,
                end=2.25,
                cue_number=8,
                label="B",
            ),
        ],
        transfer_mode="overwrite",
        start_offset_seconds=-1.0,
    )

    events = bridge.list_track_events("tc1_tg2_tr4")
    assert [event.start for event in events] == pytest.approx([0.5, 1.0])


def test_ma3_sync_adapter_apply_push_transfer_round_trips_preserved_cue_ref():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    service.apply_push_transfer(
        target_track_coord="tc1_tg2_tr4",
        selected_events=[
            Event(
                id="evt_section",
                take_id="take_a",
                start=0.5,
                end=0.75,
                cue_number=11,
                cue_ref="Q11A",
                label="Verse",
            )
        ],
        transfer_mode="overwrite",
    )

    pull_events = service.list_pull_source_events("tc1_tg2_tr4")

    assert len(pull_events) == 1
    assert pull_events[0]["label"] == "Verse"
    assert pull_events[0]["start"] == pytest.approx(0.5)
    assert pull_events[0]["cue_number"] == 11
    assert pull_events[0]["cue_ref"] == "Q11A"


def test_ma3_sync_adapter_apply_push_transfer_round_trips_float_cue_numbers():
    bridge = SimulatedMA3Bridge()
    service = MA3SyncAdapter(bridge)

    service.apply_push_transfer(
        target_track_coord="tc1_tg2_tr4",
        selected_events=[
            Event(
                id="evt_section",
                take_id="take_a",
                start=0.5,
                end=0.75,
                cue_number=11.5,
                label="Verse",
            )
        ],
        transfer_mode="overwrite",
    )

    pull_events = service.list_pull_source_events("tc1_tg2_tr4")

    assert len(pull_events) == 1
    assert pull_events[0]["cue_number"] == 11.5
    assert pull_events[0]["cue_ref"] == "11.5"


def test_ma3_sync_adapter_apply_push_transfer_raises_when_bridge_cannot_execute():
    bridge = _Bridge()
    service = MA3SyncAdapter(bridge)

    with pytest.raises(RuntimeError, match="MA3 bridge does not support push apply"):
        service.apply_push_transfer(
            target_track_coord="tc1_tg2_tr3",
            selected_events=[],
        )


def test_ma3_sync_adapter_refresh_push_track_options_uses_bridge_refresh_tracks_when_available():
    class _RefreshTracksBridge(_Bridge):
        def __init__(self) -> None:
            super().__init__()
            self.refresh_tracks_calls = 0

        def refresh_tracks(self, *, timecode_no=None, track_group_no=None):
            assert timecode_no is None
            assert track_group_no is None
            self.refresh_tracks_calls += 1
            return []

    bridge = _RefreshTracksBridge()
    service = MA3SyncAdapter(bridge)

    service.refresh_push_track_options(target_track_coord="tc1_tg2_tr3")

    assert bridge.refresh_tracks_calls == 1


def test_ma3_sync_adapter_refresh_push_track_options_forwards_timecode_and_track_group():
    class _RefreshTracksBridge(_Bridge):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, int | None]] = []

        def refresh_tracks(self, *, timecode_no=None, track_group_no=None):
            self.calls.append(
                {
                    "timecode_no": timecode_no,
                    "track_group_no": track_group_no,
                }
            )
            return []

    bridge = _RefreshTracksBridge()
    service = MA3SyncAdapter(bridge)

    service.refresh_push_track_options(
        target_track_coord="tc2_tg4_tr8",
        timecode_no=2,
        track_group_no=4,
    )

    assert bridge.calls == [{"timecode_no": 2, "track_group_no": 4}]
