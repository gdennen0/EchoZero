"""
Unit tests for SetlistSnapshotService scoped switch behavior.
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from src.features.setlists.application.setlist_snapshot_service import SetlistSnapshotService
from src.shared.domain.entities import DataStateSnapshot


def _make_service():
    return SetlistSnapshotService(
        setlist_repo=Mock(),
        setlist_song_repo=Mock(),
        snapshot_service=Mock(),
        project_service=Mock(),
    )


def test_expand_with_upstream_dependencies_adds_sources():
    service = _make_service()
    conn_repo = Mock()
    conn_repo.list_by_block.side_effect = lambda block_id: [
        SimpleNamespace(source_block_id="upstream-1", target_block_id="target-1")
    ] if block_id == "target-1" else []
    facade = SimpleNamespace(
        connection_repo=conn_repo,
        list_blocks=Mock(return_value=SimpleNamespace(
            success=True,
            data=[
                SimpleNamespace(id="target-1", type="Editor", metadata={"state_scope": "per_song"}),
                SimpleNamespace(id="upstream-1", type="Editor", metadata={"state_scope": "per_song"}),
            ],
        )),
    )

    expanded = service._expand_with_upstream_dependencies(facade, {"target-1"}, set())

    assert expanded == {"target-1", "upstream-1"}


def test_expand_with_upstream_dependencies_skips_global_unless_in_payload():
    service = _make_service()
    conn_repo = Mock()
    conn_repo.list_by_block.side_effect = lambda block_id: [
        SimpleNamespace(source_block_id="global-upstream", target_block_id="target-1")
    ] if block_id == "target-1" else []
    facade = SimpleNamespace(
        connection_repo=conn_repo,
        list_blocks=Mock(return_value=SimpleNamespace(
            success=True,
            data=[
                SimpleNamespace(id="target-1", type="Editor", metadata={"state_scope": "per_song"}),
                SimpleNamespace(id="global-upstream", type="Editor", metadata={"state_scope": "global"}),
            ],
        )),
    )

    expanded = service._expand_with_upstream_dependencies(facade, {"target-1"}, set())
    assert expanded == {"target-1"}

    expanded_with_payload = service._expand_with_upstream_dependencies(
        facade,
        {"target-1"},
        {"global-upstream"},
    )
    assert expanded_with_payload == {"target-1", "global-upstream"}


def test_resolve_song_relevant_block_ids_defaults_to_per_song():
    service = _make_service()
    snapshot = DataStateSnapshot(
        id="snap-1",
        song_id="song-1",
        created_at=datetime.now(timezone.utc),
        data_items=[],
        block_local_state={},
        block_settings_overrides={},
    )
    facade = SimpleNamespace(
        list_blocks=Mock(return_value=SimpleNamespace(
            success=True,
            data=[
                SimpleNamespace(id="per-song-1", type="Editor", metadata={}),
                SimpleNamespace(id="global-1", type="Editor", metadata={"state_scope": "global"}),
                SimpleNamespace(id="show-manager-1", type="ShowManager", metadata={"state_scope": "global"}),
            ],
        ))
    )

    resolved = service._resolve_song_relevant_block_ids(facade, snapshot)
    assert "per-song-1" in resolved
    assert "show-manager-1" in resolved
    assert "global-1" not in resolved


def test_switch_returns_degraded_message_when_restore_or_showmanager_fails():
    service = _make_service()
    project = SimpleNamespace(id="project-1")
    setlist = SimpleNamespace(id="setlist-1", project_id="project-1")
    song = SimpleNamespace(
        id="song-1",
        setlist_id="setlist-1",
        status="completed",
        audio_path="/tmp/song.wav",
    )
    snapshot = DataStateSnapshot(
        id="snap-1",
        song_id="song-1",
        created_at=datetime.now(timezone.utc),
        data_items=[],
        block_local_state={},
        block_settings_overrides={},
    )

    service._setlist_repo.get.return_value = setlist
    service._setlist_song_repo.get.return_value = song
    service._project_service.load_project.return_value = project
    service._project_service.get_snapshot.return_value = snapshot.to_dict()
    service._snapshot_service.restore_snapshot.return_value = {
        "failed_blocks": [{"block_id": "block-x", "error": "boom"}]
    }

    facade = Mock()
    facade.current_project_id = "project-1"
    facade.event_bus = None
    facade.list_blocks.return_value = SimpleNamespace(success=True, data=[])

    success, message = service.switch_active_song("setlist-1", "song-1", facade)

    assert success is True
    assert message is not None
    assert "completed with issues" in message


# ---- Auto-save outgoing song tests ----


def _make_switch_fixtures(*, outgoing_song_id="song-A", target_song_id="song-B"):
    """Shared setup for auto-save tests: two processed songs with snapshots."""
    service = _make_service()
    project = SimpleNamespace(id="project-1")
    setlist = SimpleNamespace(
        id="setlist-1",
        project_id="project-1",
        metadata={"active_song_id": outgoing_song_id},
    )
    outgoing_song = SimpleNamespace(
        id=outgoing_song_id,
        setlist_id="setlist-1",
        status="completed",
        audio_path=f"/tmp/{outgoing_song_id}.wav",
    )
    target_song = SimpleNamespace(
        id=target_song_id,
        setlist_id="setlist-1",
        status="completed",
        audio_path=f"/tmp/{target_song_id}.wav",
    )
    target_snapshot = DataStateSnapshot(
        id="snap-target",
        song_id=target_song_id,
        created_at=datetime.now(timezone.utc),
        data_items=[],
        block_local_state={},
        block_settings_overrides={},
    )
    outgoing_snapshot = DataStateSnapshot(
        id="snap-outgoing",
        song_id=outgoing_song_id,
        created_at=datetime.now(timezone.utc),
        data_items=[{"id": "item-1", "block_id": "b1", "name": "x", "type": "EventDataItem"}],
        block_local_state={"b1": {"events": ["item-1"]}},
        block_settings_overrides={},
    )

    service._setlist_repo.get.return_value = setlist
    service._setlist_song_repo.get.side_effect = (
        lambda sid: outgoing_song if sid == outgoing_song_id else target_song
    )
    service._project_service.load_project.return_value = project
    service._project_service.get_snapshot.return_value = target_snapshot.to_dict()
    service._snapshot_service.restore_snapshot.return_value = {"failed_blocks": []}
    service._snapshot_service.save_snapshot.return_value = outgoing_snapshot

    facade = Mock()
    facade.current_project_id = "project-1"
    facade.event_bus = None
    facade.list_blocks.return_value = SimpleNamespace(success=True, data=[])

    return service, facade, outgoing_song_id, target_song_id


def test_auto_save_called_on_song_switch():
    """When switching songs, the outgoing song's snapshot is saved first."""
    service, facade, outgoing_id, target_id = _make_switch_fixtures()

    success, message = service.switch_active_song("setlist-1", target_id, facade)

    assert success is True
    service._snapshot_service.save_snapshot.assert_called_once()
    call_kwargs = service._snapshot_service.save_snapshot.call_args
    assert call_kwargs[1]["song_id"] == outgoing_id
    service._project_service.set_snapshot.assert_called()


def test_auto_save_failure_does_not_block_switch():
    """If saving the outgoing song fails, the switch still completes (degraded)."""
    service, facade, _, target_id = _make_switch_fixtures()
    service._snapshot_service.save_snapshot.side_effect = RuntimeError("disk full")

    success, message = service.switch_active_song("setlist-1", target_id, facade)

    assert success is True
    assert message is not None
    assert "Auto-save outgoing song failed" in message


def test_no_auto_save_when_no_active_song():
    """On first switch (no prior active song, no metadata), no save is attempted."""
    service, facade, _, target_id = _make_switch_fixtures()
    setlist_no_meta = SimpleNamespace(
        id="setlist-1",
        project_id="project-1",
        metadata={},
    )
    service._setlist_repo.get.return_value = setlist_no_meta

    success, message = service.switch_active_song("setlist-1", target_id, facade)

    assert success is True
    service._snapshot_service.save_snapshot.assert_not_called()


def test_no_auto_save_when_switching_to_same_song():
    """Switching to the already-active song does not re-save."""
    service, facade, outgoing_id, _ = _make_switch_fixtures()
    service._active_song_id = outgoing_id
    service._project_service.get_snapshot.return_value = DataStateSnapshot(
        id="snap-same",
        song_id=outgoing_id,
        created_at=datetime.now(timezone.utc),
        data_items=[],
        block_local_state={},
        block_settings_overrides={},
    ).to_dict()

    success, _ = service.switch_active_song("setlist-1", outgoing_id, facade)

    assert success is True
    service._snapshot_service.save_snapshot.assert_not_called()


def test_active_song_id_updated_after_switch():
    """After a successful switch, _active_song_id tracks the new song."""
    service, facade, _, target_id = _make_switch_fixtures()
    assert service._active_song_id is None

    service.switch_active_song("setlist-1", target_id, facade)

    assert service._active_song_id == target_id
