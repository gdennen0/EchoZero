"""Fixture-replay contract tests for MA3 reconnect + event apply flow.

These tests replay captured MA3-like payloads through the public manager hooks
(`on_track_groups_received`, `on_tracks_received`, `on_track_events_received`)
to prove app-boundary behavior without requiring a live MA3 session.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from src.features.show_manager.application.sync_system_manager import SyncSystemManager
from src.features.show_manager.domain.sync_layer_entity import SyncLayerEntity, SyncSource, SyncStatus


def _fixture_payload() -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "ma3" / "reconnect_replay_v1.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _comparison(diverged: bool, ma3_count: int = 1, editor_count: int = 1, matched_count: int = 1):
    c = MagicMock()
    c.diverged = diverged
    c.ma3_count = ma3_count
    c.editor_count = editor_count
    c.matched_count = matched_count
    return c


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
    facade.ma3_comm_service = MagicMock()
    facade.ma3_comm_service.hook_cmdsubtrack.return_value = True
    facade.ma3_comm_service.hook_track_group_changes.return_value = True
    facade.ma3_comm_service.get_events.return_value = True

    settings = MagicMock()
    settings.synced_layers = []

    return SyncSystemManager(
        facade=facade,
        show_manager_block_id="show_manager_1",
        settings_manager=settings,
    )


def _replay_structure(manager: SyncSystemManager, payload: dict) -> None:
    tc = payload["timecode_no"]
    groups = payload["track_groups"]
    manager.on_track_groups_received(tc, groups)
    for g in groups:
        tracks = payload["tracks_by_group"][str(g["no"])]
        manager.on_tracks_received(tc, g["no"], tracks)


def test_reconnect_replay_rebinds_entity_by_ez_track_id_and_requests_events(manager):
    payload = _fixture_payload()

    entity = SyncLayerEntity(
        id="sync-kick",
        source=SyncSource.EDITOR,
        name="ez_Kick Layer",
        editor_layer_id="kick_layer",
        editor_block_id="editor_1",
        ma3_coord="tc101_tg1_tr99",
        ma3_timecode_no=101,
        ma3_track_group=1,
        ma3_track=99,
        ez_track_id="ez:kick_layer",
        sync_status=SyncStatus.AWAITING_CONNECTION,
    )
    manager._synced_layers[entity.id] = entity

    _replay_structure(manager, payload)

    assert entity.ma3_coord == "tc101_tg1_tr2"
    assert entity.ma3_track == 2
    assert entity.sync_status == SyncStatus.PENDING

    manager._facade.ma3_comm_service.hook_cmdsubtrack.assert_any_call(101, 1, 2)
    manager._facade.ma3_comm_service.get_events.assert_any_call(101, 1, 2, request_id=None)


def test_reconnect_replay_pending_to_synced_on_matching_events(manager):
    payload = _fixture_payload()

    entity = SyncLayerEntity(
        id="sync-kick-2",
        source=SyncSource.EDITOR,
        name="ez_Kick Layer",
        editor_layer_id="kick_layer",
        editor_block_id="editor_1",
        ma3_coord="tc101_tg1_tr99",
        ma3_timecode_no=101,
        ma3_track_group=1,
        ma3_track=99,
        ez_track_id="ez:kick_layer",
        sync_status=SyncStatus.AWAITING_CONNECTION,
    )
    manager._synced_layers[entity.id] = entity

    manager._compare_entity = MagicMock(return_value=_comparison(diverged=False, ma3_count=3, editor_count=3, matched_count=3))
    manager._schedule_push_ma3_to_editor = MagicMock()
    manager._save_to_settings = MagicMock()
    manager._push_sync_state_to_all_layers = MagicMock()

    _replay_structure(manager, payload)
    events = payload["events_by_coord"]["tc101_tg1_tr2"]
    manager.on_track_events_received("tc101_tg1_tr2", events)

    assert entity.sync_status == SyncStatus.SYNCED
    manager._schedule_push_ma3_to_editor.assert_not_called()
    manager._save_to_settings.assert_called()
    manager._push_sync_state_to_all_layers.assert_called()


def test_reconnect_replay_keeps_unmatched_entity_awaiting_connection(manager):
    payload = _fixture_payload()

    entity = SyncLayerEntity(
        id="sync-unmatched",
        source=SyncSource.EDITOR,
        name="ez_Unknown Layer",
        editor_layer_id="unknown_layer",
        editor_block_id="editor_1",
        ma3_coord="tc101_tg1_tr77",
        ma3_timecode_no=101,
        ma3_track_group=1,
        ma3_track=77,
        ez_track_id="ez:missing_layer",
        sync_status=SyncStatus.AWAITING_CONNECTION,
    )
    manager._synced_layers[entity.id] = entity

    _replay_structure(manager, payload)

    assert entity.ma3_coord == "tc101_tg1_tr77"
    assert entity.sync_status == SyncStatus.AWAITING_CONNECTION
