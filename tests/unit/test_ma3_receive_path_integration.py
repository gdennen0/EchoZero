from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from PyQt6.QtWidgets import QApplication

from src.application.events.events import MA3OscInbound
from src.features.show_manager.application.sync_system_manager import SyncSystemManager


def _fixture_payload() -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "ma3" / "protocol_receive_v1.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def manager(qapp):
    facade = MagicMock()
    facade.command_bus = MagicMock()
    facade.event_bus = MagicMock()
    facade.ma3_comm_service = None

    settings = MagicMock()
    settings.synced_layers = []
    settings.target_timecode = 101
    settings.ma3_ip = "127.0.0.1"
    settings.ma3_port = 9001

    manager = SyncSystemManager(
        facade=facade,
        show_manager_block_id="show_manager_1",
        settings_manager=settings,
    )
    manager._connection_state = "connected"
    manager._send_lua_command_with_target = MagicMock(return_value=True)
    return manager


def _osc_event(message: str) -> MA3OscInbound:
    return MA3OscInbound(
        data={
            "address": "/ez/message",
            "args": [message],
            "addr": ("127.0.0.1", 9000),
            "raw_data": message.encode("utf-8"),
        }
    )


def test_receive_path_updates_structure_and_events_via_osc_entrypoint(manager):
    fixture = _fixture_payload()

    manager._on_osc_inbound_event(_osc_event(fixture["messages"]["trackgroups.list"]))

    groups = manager._ma3_track_groups[fixture["timecode_no"]]
    assert [(group.track_group_no, group.name, group.track_count) for group in groups] == [
        (1, "Drums", 2),
        (2, "FX", 1),
    ]
    manager._send_lua_command_with_target.assert_has_calls(
        [call("EZ.GetTracks(101, 1)"), call("EZ.GetTracks(101, 2)")],
        any_order=False,
    )

    manager._on_osc_inbound_event(_osc_event(fixture["messages"]["tracks.list"]))

    kick_track = manager._ma3_tracks["tc101_tg1_tr1"]
    snare_track = manager._ma3_tracks["tc101_tg1_tr2"]
    assert kick_track.name == "Kick Layer"
    assert kick_track.event_count == 3
    assert kick_track.sequence_no == 12
    assert kick_track.note == "ez:kick_layer"
    assert snare_track.name == "Snare Layer"
    assert snare_track.event_count == 1
    assert snare_track.sequence_no is None

    manager._on_osc_inbound_event(_osc_event(fixture["messages"]["events.list"]))

    cached_events = manager._ma3_track_events["tc101_tg1_tr1"]
    assert len(cached_events) == 2
    assert cached_events[0].name == "Kick"
    assert cached_events[0].cmd == "Go+"
    assert cached_events[1].name == "Kick Accent"
    assert cached_events[1].cmd == "Go+ Sequence 12"


def test_receive_path_ignores_track_changed_with_embedded_delimiters_without_corrupting_state(manager):
    fixture = _fixture_payload()

    manager._on_osc_inbound_event(_osc_event(fixture["messages"]["tracks.list"]))
    manager._on_osc_inbound_event(_osc_event(fixture["messages"]["events.list"]))

    before_tracks = dict(manager._ma3_tracks)
    before_events = list(manager._ma3_track_events["tc101_tg1_tr1"])

    manager._on_osc_inbound_event(_osc_event(fixture["messages"]["track.changed"]))

    assert manager._ma3_tracks.keys() == before_tracks.keys()
    assert manager._ma3_tracks["tc101_tg1_tr1"].name == before_tracks["tc101_tg1_tr1"].name
    assert manager._ma3_track_events["tc101_tg1_tr1"] == before_events
