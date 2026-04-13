import json
from pathlib import Path

from src.application.events.event_bus import EventBus
from src.features.ma3.application.ma3_communication_service import MA3CommunicationService


def _fixture_payload() -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "ma3" / "protocol_receive_v1.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _make_service() -> MA3CommunicationService:
    return MA3CommunicationService(event_bus=EventBus())


def test_handle_message_parses_trackgroups_list_payload():
    service = _make_service()
    received = []
    service.register_handler("trackgroups", "list", received.append)

    fixture = _fixture_payload()
    trackgroups = fixture["expected"]["trackgroups"]

    service._handle_message(fixture["messages"]["trackgroups.list"], ("127.0.0.1", 9000))

    assert len(received) == 1
    parsed = received[0]
    assert parsed.object_type == "trackgroups"
    assert parsed.change_type == "list"
    assert parsed.tc == 101
    assert parsed.data["count"] == 2
    assert parsed.data["trackgroups"] == trackgroups


def test_handle_message_parses_tracks_list_payload_with_note_and_sequence():
    service = _make_service()
    received = []
    service.register_handler("tracks", "list", received.append)

    fixture = _fixture_payload()
    tracks = fixture["expected"]["tracks"]

    service._handle_message(fixture["messages"]["tracks.list"], ("127.0.0.1", 9000))

    assert len(received) == 1
    parsed = received[0]
    assert parsed.tc == 101
    assert parsed.tg == 1
    assert parsed.data["count"] == 2
    assert parsed.data["tracks"] == tracks
    assert parsed.data["tracks"][0]["note"] == "ez:kick_layer"
    assert parsed.data["tracks"][0]["sequence_no"] == 12


def test_handle_message_parses_events_list_payload_with_request_id():
    service = _make_service()
    received = []
    service.register_handler("events", "list", received.append)

    fixture = _fixture_payload()
    events = fixture["expected"]["events"]

    service._handle_message(fixture["messages"]["events.list"], ("127.0.0.1", 9000))

    assert len(received) == 1
    parsed = received[0]
    assert parsed.tc == 101
    assert parsed.tg == 1
    assert parsed.track == 1
    assert parsed.data["request_id"] == 7
    assert parsed.data["count"] == 2
    assert parsed.events == events


def test_handle_message_preserves_pipe_characters_inside_track_changed_payload():
    service = _make_service()
    received = []
    service.register_handler("track", "changed", received.append)

    fixture = _fixture_payload()
    message = fixture["messages"]["track.changed"]

    service._handle_message(message, ("127.0.0.1", 9000))

    assert len(received) == 1
    parsed = received[0]
    assert parsed.object_type == "track"
    assert parsed.change_type == "changed"
    assert parsed.tc == 101
    assert parsed.tg == 1
    assert parsed.track == 1
    assert parsed.data["count"] == 3
    assert parsed.events == [
        {"idx": 1, "time": 1.25, "name": "Kick", "cmd": "Go+ | Cue 5", "tc": 101, "tg": 1, "track": 1},
    ]
    assert parsed.data["changes"] == {"added": 1, "modified": 0, "deleted": 0, "moved": 0}
    assert parsed.data["added"] == "[{fingerprint:0.409233|Kick|Go+|Cue 5}]"
    assert parsed.data["deleted"] == []
    assert parsed.data["moved"] == []
