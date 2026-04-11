import json

from src.application.events.event_bus import EventBus
from src.features.ma3.application.ma3_communication_service import MA3CommunicationService


def _make_service() -> MA3CommunicationService:
    return MA3CommunicationService(event_bus=EventBus())


def test_handle_message_parses_trackgroups_list_payload():
    service = _make_service()
    received = []
    service.register_handler("trackgroups", "list", received.append)

    trackgroups = [
        {"no": 1, "name": "Drums", "track_count": 3},
        {"no": 2, "name": "Bass", "track_count": 1},
    ]
    message = (
        "type=trackgroups|change=list|timestamp=1712860800|tc=101|count=2|"
        f"trackgroups={json.dumps(trackgroups, separators=(',', ':'))}"
    )

    service._handle_message(message, ("127.0.0.1", 9000))

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

    tracks = [
        {"no": 1, "name": "Kick", "event_count": 4, "sequence_no": 12, "note": "ez:kick_layer"},
        {"no": 2, "name": "Snare", "event_count": 0, "sequence_no": None, "note": ""},
    ]
    message = (
        "type=tracks|change=list|timestamp=1712860800|tc=101|tg=1|count=2|"
        f"tracks={json.dumps(tracks, separators=(',', ':'))}"
    )

    service._handle_message(message, ("127.0.0.1", 9000))

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

    events = [
        {"idx": 1, "time": 1.25, "name": "Kick", "cmd": "Go+", "tc": 101, "tg": 1, "track": 2},
        {"idx": 2, "time": 2.5, "name": "Snare", "cmd": "Go+ Sequence 5", "tc": 101, "tg": 1, "track": 2},
    ]
    message = (
        "type=events|change=list|timestamp=1712860800|tc=101|tg=1|track=2|request_id=7|count=2|"
        f"events={json.dumps(events, separators=(',', ':'))}"
    )

    service._handle_message(message, ("127.0.0.1", 9000))

    assert len(received) == 1
    parsed = received[0]
    assert parsed.tc == 101
    assert parsed.tg == 1
    assert parsed.track == 2
    assert parsed.data["request_id"] == 7
    assert parsed.data["count"] == 2
    assert parsed.events == events


def test_handle_message_preserves_pipe_characters_inside_track_changed_payload():
    service = _make_service()
    received = []
    service.register_handler("track", "changed", received.append)

    events = [
        {"idx": 1, "time": 1.25, "name": "Kick", "cmd": "Go+", "tc": 101, "tg": 1, "track": 2},
    ]
    changes = {"added": 1, "modified": 0, "deleted": 0, "moved": 0}
    added = "[{fingerprint:0.409233|Kick|Go+}]"
    message = (
        "type=track|change=changed|timestamp=1712860800|tc=101|tg=1|track=2|count=3|"
        f"events={json.dumps(events, separators=(',', ':'))}|"
        f"changes={json.dumps(changes, separators=(',', ':'))}|"
        f"added={added}|deleted=[]|moved=[]"
    )

    service._handle_message(message, ("127.0.0.1", 9000))

    assert len(received) == 1
    parsed = received[0]
    assert parsed.object_type == "track"
    assert parsed.change_type == "changed"
    assert parsed.tc == 101
    assert parsed.tg == 1
    assert parsed.track == 2
    assert parsed.data["count"] == 3
    assert parsed.events == events
    assert parsed.data["changes"] == changes
    assert parsed.data["added"] == added
    assert parsed.data["deleted"] == []
    assert parsed.data["moved"] == []
