"""MA3 payload contract tests for SyncSystemManager.

These tests intentionally use fixture-like payload variants so we can evolve
MA3 plugin payloads without breaking sync parsing.
"""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from src.features.show_manager.application.sync_system_manager import SyncSystemManager


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
    facade.ma3_comm_service = None

    settings = MagicMock()
    settings.synced_layers = []

    return SyncSystemManager(
        facade=facade,
        show_manager_block_id="show_manager_1",
        settings_manager=settings,
    )


def test_normalize_ma3_event_accepts_aliases_and_weird_types(manager):
    raw = {
        "timestamp": "12.75",
        "label": b"Snare",
        "command": 123,
        "id": "9",
        "timecode": "101",
        "track_group": "2",
        "tr": "4",
        "extra_field": {"future": True},
    }

    normalized = manager._normalize_ma3_event(raw, fallback_idx=1)

    assert normalized.time == pytest.approx(12.75)
    assert normalized.name == "Snare"
    assert normalized.cmd == "123"
    assert normalized.idx == 9
    assert normalized.tc == 101
    assert normalized.tg == 2
    assert normalized.track == 4


def test_on_track_events_received_normalizes_and_defaults_fields(manager):
    payload = [
        {"time": "1.5", "name": "Kick", "idx": "2", "tc": "1", "tg": "1", "track": "7"},
        {"t": 2.25, "text": "Hat", "no": "3", "timecode": 1, "track_group": 1, "tr": 7},
        {"name": None},
    ]

    manager.on_track_events_received("tc1_tg1_tr7", payload)

    stored = manager._ma3_track_events["tc1_tg1_tr7"]
    assert len(stored) == 3

    assert stored[0].time == pytest.approx(1.5)
    assert stored[0].name == "Kick"
    assert stored[0].idx == 2

    assert stored[1].time == pytest.approx(2.25)
    assert stored[1].name == "Hat"
    assert stored[1].idx == 3

    # Missing fields should be safe/defaulted and idx should use fallback (1-based enumerate)
    assert stored[2].time == 0.0
    assert stored[2].name == ""
    assert stored[2].idx == 3


def test_add_all_editor_events_to_ma3_filters_explicit_non_main_events(manager):
    manager._send_lua_command_with_target = MagicMock()

    events = [
        {"time": 1.0, "name": "Main Kick", "metadata": {"is_main": True}},
        {"time": 2.0, "name": "Alt 1", "metadata": {"is_main": False}},
        {"time": 3.0, "name": "Alt 2", "metadata": {"take_role": "alternate"}},
        {"time": 4.0, "name": "Main Snare", "metadata": {}},
    ]

    result = manager._add_all_editor_events_to_ma3(events, tc_no=1, tg_no=1, tr_no=1)

    assert result["added"] == 2
    assert result["failed"] == 0
    assert manager._send_lua_command_with_target.call_count == 2

    calls = [c.args[0] for c in manager._send_lua_command_with_target.call_args_list]
    assert any("Main Kick" in c for c in calls)
    assert any("Main Snare" in c for c in calls)
    assert all("Alt 1" not in c for c in calls)
    assert all("Alt 2" not in c for c in calls)


def test_add_all_editor_events_to_ma3_defaults_to_main_when_metadata_absent(manager):
    manager._send_lua_command_with_target = MagicMock()

    events = [
        {"time": 1.0, "name": "Kick"},
        {"time": 2.0, "name": "Snare"},
    ]

    result = manager._add_all_editor_events_to_ma3(events, tc_no=1, tg_no=1, tr_no=2)

    assert result["added"] == 2
    assert result["failed"] == 0
    assert manager._send_lua_command_with_target.call_count == 2
