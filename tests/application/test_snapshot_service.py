"""
Integration tests for SnapshotService

Tests snapshot save/restore functionality using BlockStateHelper.
Verifies backward compatibility and snapshot format.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone

from src.features.projects.application import SnapshotService
from src.shared.domain.entities import DataStateSnapshot
from src.features.blocks.domain import Block
from src.domain.entities.data_item import AudioDataItem, EventDataItem


class TestSnapshotServiceIntegration:
    """Integration tests for SnapshotService with BlockStateHelper"""
    
    @pytest.fixture
    def mock_block_repo(self):
        """Create mock block repository"""
        return Mock()
    
    @pytest.fixture
    def mock_data_item_repo(self):
        """Create mock data item repository"""
        return Mock()
    
    @pytest.fixture
    def mock_local_state_repo(self):
        """Create mock block local state repository"""
        return Mock()
    
    @pytest.fixture
    def mock_project_service(self):
        """Create mock project service"""
        return Mock()
    
    @pytest.fixture
    def snapshot_service(
        self,
        mock_block_repo,
        mock_data_item_repo,
        mock_local_state_repo,
        mock_project_service
    ):
        """Create SnapshotService instance"""
        return SnapshotService(
            mock_block_repo,
            mock_data_item_repo,
            mock_local_state_repo,
            mock_project_service
        )
    
    @pytest.fixture
    def sample_project_blocks(self):
        """Create sample blocks for testing"""
        return [
            Block(
                id="block-1",
                project_id="project-1",
                name="LoadAudio",
                type="LoadAudio",
                metadata={"file_path": "/path/to/audio.wav"}
            ),
            Block(
                id="block-2",
                project_id="project-1",
                name="DetectOnsets",
                type="DetectOnsets",
                metadata={"threshold": 0.5}
            )
        ]
    
    @pytest.fixture
    def sample_data_items(self):
        """Create sample data items"""
        return {
            "block-1": [
                AudioDataItem(
                    id="item-1",
                    block_id="block-1",
                    name="Audio1",
                    type="Audio",
                    created_at=datetime.now(timezone.utc),
                    file_path="/path/to/audio.wav",
                    sample_rate=44100,
                    length_ms=1000.0
                )
            ],
            "block-2": [
                EventDataItem(
                    id="item-2",
                    block_id="block-2",
                    name="Events1",
                    type="Event",
                    created_at=datetime.now(timezone.utc),
                    event_count=5
                )
            ]
        }
    
    def test_save_snapshot_uses_block_state_helper(
        self,
        snapshot_service,
        mock_block_repo,
        sample_project_blocks,
        sample_data_items
    ):
        """Test that save_snapshot uses BlockStateHelper.get_project_state"""
        # Arrange
        project_id = "project-1"
        song_id = "song-1"
        
        mock_block_repo.list_by_project.return_value = sample_project_blocks
        
        # Mock BlockStateHelper methods
        project_state = {
            "block-1": {
                "block_id": "block-1",
                "settings": {"file_path": "/path/to/audio.wav"},
                "local_state": {"input1": "item-1"},
                "data_items": [item.to_dict() for item in sample_data_items["block-1"]]
            },
            "block-2": {
                "block_id": "block-2",
                "settings": {"threshold": 0.5},
                "local_state": {},
                "data_items": [item.to_dict() for item in sample_data_items["block-2"]]
            }
        }
        
        # Mock the helper's get_project_state
        snapshot_service._state_helper.get_project_state = Mock(return_value=project_state)
        
        # Act
        snapshot = snapshot_service.save_snapshot(project_id, song_id)
        
        # Assert
        assert snapshot.song_id == song_id
        assert len(snapshot.data_items) == 2  # One from each block
        assert len(snapshot.block_local_state) == 1  # Only block-1 has local state
        assert "block-1" in snapshot.block_local_state
        
        # Verify helper was called
        snapshot_service._state_helper.get_project_state.assert_called_once_with(project_id)
    
    def test_save_snapshot_backward_compatible_format(
        self,
        snapshot_service,
        sample_project_blocks
    ):
        """Test that saved snapshot format is backward compatible"""
        # Arrange
        project_id = "project-1"
        song_id = "song-1"
        
        project_state = {
            "block-1": {
                "block_id": "block-1",
                "settings": {},
                "local_state": {"input1": "item-1"},
                "data_items": [
                    {
                        "id": "item-1",
                        "block_id": "block-1",
                        "name": "Audio1",
                        "type": "Audio",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": "/path/to/audio.wav"
                    }
                ]
            }
        }
        
        snapshot_service._state_helper.get_project_state = Mock(return_value=project_state)
        
        # Act
        snapshot = snapshot_service.save_snapshot(project_id, song_id)
        
        # Assert - verify snapshot has expected structure
        assert hasattr(snapshot, "id")
        assert hasattr(snapshot, "song_id")
        assert hasattr(snapshot, "created_at")
        assert hasattr(snapshot, "data_items")
        assert hasattr(snapshot, "block_local_state")
        assert hasattr(snapshot, "block_settings_overrides")
        
        # Verify data items format
        assert isinstance(snapshot.data_items, list)
        assert len(snapshot.data_items) == 1
        assert snapshot.data_items[0]["id"] == "item-1"
        
        # Verify block local state format
        assert isinstance(snapshot.block_local_state, dict)
        assert "block-1" in snapshot.block_local_state
    
    def test_restore_snapshot_uses_block_state_helper(
        self,
        snapshot_service,
        mock_block_repo,
        mock_data_item_repo,
        mock_local_state_repo
    ):
        """Test that restore_snapshot uses BlockStateHelper.restore_block_state"""
        # Arrange
        project_id = "project-1"
        snapshot = DataStateSnapshot(
            id="snapshot-1",
            song_id="song-1",
            created_at=datetime.now(timezone.utc),
            data_items=[
                {
                    "id": "item-1",
                    "block_id": "block-1",
                    "name": "Audio1",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": "/path/to/audio.wav"
                }
            ],
            block_local_state={
                "block-1": {"input1": "item-1"}
            }
        )
        
        # Mock block repository for clearing metadata
        mock_block_repo.list_by_project.return_value = [
            Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        ]
        mock_block_repo.get_by_id.return_value = Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        mock_block_repo.update.return_value = Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        
        # Mock the helper's restore_block_state method
        restore_calls = []
        original_restore = snapshot_service._state_helper.restore_block_state
        
        def track_restore(block_id, state, project_dir=None):
            restore_calls.append((block_id, state, project_dir))
            return original_restore(block_id, state, project_dir)
        
        snapshot_service._state_helper.restore_block_state = track_restore
        
        # Mock delete_by_project
        mock_data_item_repo.delete_by_project.return_value = 0
        
        # Act
        snapshot_service.restore_snapshot(project_id, snapshot)
        
        # Assert - verify helper's restore_block_state was called
        assert len(restore_calls) == 1
        assert restore_calls[0][0] == "block-1"
        assert restore_calls[0][1]["local_state"] == {"input1": "item-1"}
        
        # Verify data items were cleared first
        mock_data_item_repo.delete_by_project.assert_called_once_with(project_id)
    
    def test_restore_snapshot_groups_by_block(
        self,
        snapshot_service,
        mock_block_repo,
        mock_data_item_repo
    ):
        """Test that restore_snapshot groups data items by block"""
        # Arrange
        project_id = "project-1"
        snapshot = DataStateSnapshot(
            id="snapshot-1",
            song_id="song-1",
            created_at=datetime.now(timezone.utc),
            data_items=[
                {
                    "id": "item-1",
                    "block_id": "block-1",
                    "name": "Audio1",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat()
                },
                {
                    "id": "item-2",
                    "block_id": "block-1",
                    "name": "Audio2",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat()
                },
                {
                    "id": "item-3",
                    "block_id": "block-2",
                    "name": "Events1",
                    "type": "Event",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            ],
            block_local_state={
                "block-1": {"input1": "item-1"},
                "block-2": {}
            }
        )
        
        # Mock block repository for clearing metadata
        mock_block_repo.list_by_project.return_value = [
            Block(id="block-1", project_id=project_id, name="Block1", type="TestType", metadata={}),
            Block(id="block-2", project_id=project_id, name="Block2", type="TestType", metadata={})
        ]
        mock_block_repo.get_by_id.side_effect = lambda bid: next(
            (b for b in mock_block_repo.list_by_project.return_value if b.id == bid),
            None
        )
        mock_block_repo.update.return_value = Block(id="block-1", project_id=project_id, name="Block1", type="TestType", metadata={})
        
        mock_data_item_repo.delete_by_project.return_value = 0
        
        # Track calls to restore_block_state
        restore_calls = []
        original_restore = snapshot_service._state_helper.restore_block_state
        
        def track_restore(block_id, state, project_dir=None):
            restore_calls.append((block_id, state))
            return original_restore(block_id, state, project_dir)
        
        snapshot_service._state_helper.restore_block_state = track_restore
        
        # Act
        snapshot_service.restore_snapshot(project_id, snapshot)
        
        # Assert - verify blocks were restored separately
        assert len(restore_calls) == 2
        
        # Verify block-1 has 2 data items
        block1_call = next(call for call in restore_calls if call[0] == "block-1")
        assert len(block1_call[1]["data_items"]) == 2
        assert block1_call[1]["local_state"] == {"input1": "item-1"}
        
        # Verify block-2 has 1 data item
        block2_call = next(call for call in restore_calls if call[0] == "block-2")
        assert len(block2_call[1]["data_items"]) == 1
        assert block2_call[1]["local_state"] == {}
    
    def test_restore_snapshot_with_project_dir(
        self,
        snapshot_service,
        mock_block_repo,
        mock_data_item_repo
    ):
        """Test that restore_snapshot resolves relative paths with project_dir"""
        # Arrange
        project_id = "project-1"
        project_dir = Path("/project/dir")
        snapshot = DataStateSnapshot(
            id="snapshot-1",
            song_id="song-1",
            created_at=datetime.now(timezone.utc),
            data_items=[
                {
                    "id": "item-1",
                    "block_id": "block-1",
                    "name": "Audio1",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": "relative/path/audio.wav"
                }
            ],
            block_local_state={}
        )
        
        # Mock block repository for clearing metadata
        mock_block_repo.list_by_project.return_value = [
            Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        ]
        mock_block_repo.get_by_id.return_value = Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        mock_block_repo.update.return_value = Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        
        mock_data_item_repo.delete_by_project.return_value = 0
        
        # Track project_dir passed to restore
        restore_calls = []
        original_restore = snapshot_service._state_helper.restore_block_state
        
        def track_restore(block_id, state, project_dir=None):
            restore_calls.append((block_id, state, project_dir))
            return original_restore(block_id, state, project_dir)
        
        snapshot_service._state_helper.restore_block_state = track_restore
        
        # Act
        snapshot_service.restore_snapshot(project_id, snapshot, project_dir)
        
        # Assert - verify project_dir was passed through
        assert len(restore_calls) == 1
        assert restore_calls[0][2] == project_dir
    
    def test_restore_snapshot_backward_compatible(
        self,
        snapshot_service,
        mock_block_repo,
        mock_data_item_repo,
        mock_local_state_repo
    ):
        """Test that restore_snapshot handles old snapshot format"""
        # Arrange - old format (direct data_items and block_local_state lists)
        project_id = "project-1"
        snapshot = DataStateSnapshot(
            id="snapshot-1",
            song_id="song-1",
            created_at=datetime.now(timezone.utc),
            data_items=[
                {
                    "id": "item-1",
                    "block_id": "block-1",
                    "name": "Audio1",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            ],
            block_local_state={
                "block-1": {"input1": "item-1"}
            }
        )
        
        # Mock block repository for clearing metadata
        mock_block_repo.list_by_project.return_value = [
            Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        ]
        mock_block_repo.get_by_id.return_value = Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        mock_block_repo.update.return_value = Block(id="block-1", project_id=project_id, name="TestBlock", type="TestType", metadata={})
        
        # Track restore calls
        restore_calls = []
        original_restore = snapshot_service._state_helper.restore_block_state
        
        def track_restore(block_id, state, project_dir=None):
            restore_calls.append((block_id, state, project_dir))
            return original_restore(block_id, state, project_dir)
        
        snapshot_service._state_helper.restore_block_state = track_restore
        
        mock_data_item_repo.delete_by_project.return_value = 0
        
        # Act
        snapshot_service.restore_snapshot(project_id, snapshot)
        
        # Assert - should handle old format without errors
        assert mock_data_item_repo.delete_by_project.called
        assert len(restore_calls) == 1  # Should have called restore_block_state
    
    def test_snapshot_save_load_all_blocks_comprehensive(
        self,
        snapshot_service,
        mock_block_repo,
        mock_data_item_repo,
        mock_local_state_repo
    ):
        """
        Comprehensive test: Save snapshot with multiple blocks, then restore and verify
        all block metadata, local state, and data items are preserved correctly.
        """
        # Arrange - Create multiple blocks with different metadata
        project_id = "project-1"
        song_id = "song-1"
        
        blocks = [
            Block(
                id="block-loadaudio",
                project_id=project_id,
                name="LoadAudio1",
                type="LoadAudio",
                metadata={
                    "audio_path": "/path/to/song1.wav",
                    "expected_outputs": {"audio": ["audio:main"]}
                }
            ),
            Block(
                id="block-detectonsets",
                project_id=project_id,
                name="DetectOnsets1",
                type="DetectOnsets",
                metadata={
                    "onset_method": "default",
                    "onset_threshold": 0.05,
                    "min_silence": 0.02,
                    "use_backtrack": True,
                    "energy_frame_length": 2048,
                    "output_mode": "clips",
                    "filter_selections": {"audio": ["audio:main"]},
                    "expected_outputs": {"events": ["events:main"]}
                }
            ),
            Block(
                id="block-separator",
                project_id=project_id,
                name="Separator1",
                type="Separator",
                metadata={
                    "model_path": "/path/to/model",
                    "sample_rate": 22050,
                    "filter_selections": {"audio": ["audio:main"]},
                    "expected_outputs": {
                        "audio": ["audio:vocals", "audio:drums", "audio:bass", "audio:other"]
                    }
                }
            ),
            Block(
                id="block-editor",
                project_id=project_id,
                name="Editor1",
                type="Editor",
                metadata={
                    "last_processed": True,
                    "filter_selections": {
                        "events": ["events:main"],
                        "audio": ["audio:main"]
                    },
                    "expected_outputs": {
                        "audio": ["audio:main"],
                        "events": ["events:edited"]
                    }
                }
            )
        ]
        
        # Create data items for each block
        data_items_by_block = {
            "block-loadaudio": [
                AudioDataItem(
                    id="item-audio-1",
                    block_id="block-loadaudio",
                    name="Song1_Audio",
                    type="Audio",
                    created_at=datetime.now(timezone.utc),
                    file_path="/path/to/song1.wav",
                    sample_rate=44100,
                    length_ms=180000.0
                )
            ],
            "block-detectonsets": [
                EventDataItem(
                    id="item-events-1",
                    block_id="block-detectonsets",
                    name="Song1_Events",
                    type="Event",
                    created_at=datetime.now(timezone.utc),
                    event_count=150
                )
            ],
            "block-separator": [
                AudioDataItem(
                    id="item-vocals-1",
                    block_id="block-separator",
                    name="Song1_Vocals",
                    type="Audio",
                    created_at=datetime.now(timezone.utc),
                    file_path="/path/to/vocals.wav",
                    sample_rate=22050,
                    length_ms=180000.0
                ),
                AudioDataItem(
                    id="item-drums-1",
                    block_id="block-separator",
                    name="Song1_Drums",
                    type="Audio",
                    created_at=datetime.now(timezone.utc),
                    file_path="/path/to/drums.wav",
                    sample_rate=22050,
                    length_ms=180000.0
                )
            ],
            "block-editor": []
        }
        
        # Create local state for blocks
        local_state_by_block = {
            "block-loadaudio": {},
            "block-detectonsets": {
                "audio": ["item-audio-1"]
            },
            "block-separator": {
                "audio": ["item-audio-1"]
            },
            "block-editor": {
                "events": ["item-events-1"],
                "audio": ["item-audio-1"]
            }
        }
        
        # Mock repositories to return our test data
        mock_block_repo.list_by_project.return_value = blocks
        
        def mock_get_by_id(block_id):
            return next((b for b in blocks if b.id == block_id), None)
        mock_block_repo.get_by_id.side_effect = mock_get_by_id
        
        def mock_list_by_block(block_id):
            return data_items_by_block.get(block_id, [])
        mock_data_item_repo.list_by_block.side_effect = mock_list_by_block
        
        def mock_get_inputs(block_id):
            return local_state_by_block.get(block_id, {})
        mock_local_state_repo.get_inputs.side_effect = mock_get_inputs
        
        # Mock the helper's get_project_state to return our test data
        project_state = {}
        for block in blocks:
            project_state[block.id] = {
                "block_id": block.id,
                "settings": block.metadata.copy(),
                "local_state": local_state_by_block.get(block.id, {}),
                "data_items": [item.to_dict() for item in data_items_by_block.get(block.id, [])]
            }
        
        snapshot_service._state_helper.get_project_state = Mock(return_value=project_state)
        
        # Act 1: Save snapshot
        saved_snapshot = snapshot_service.save_snapshot(project_id, song_id)
        
        # Verify snapshot was saved correctly
        assert saved_snapshot.song_id == song_id
        assert len(saved_snapshot.data_items) == 4  # 1 + 1 + 2 + 0
        # Empty local states are not saved (only non-empty ones)
        expected_local_state_count = sum(1 for state in local_state_by_block.values() if state)
        assert len(saved_snapshot.block_local_state) == expected_local_state_count
        assert len(saved_snapshot.block_settings_overrides) == 4  # All blocks have metadata
        
        # Verify all block metadata is in snapshot
        for block in blocks:
            assert block.id in saved_snapshot.block_settings_overrides
            assert saved_snapshot.block_settings_overrides[block.id] == block.metadata
            # Only verify non-empty local states are saved
            block_local_state = local_state_by_block.get(block.id, {})
            if block_local_state:
                assert block.id in saved_snapshot.block_local_state
                assert saved_snapshot.block_local_state[block.id] == block_local_state
        
        # Verify all data items are in snapshot
        saved_item_ids = {item["id"] for item in saved_snapshot.data_items}
        expected_item_ids = set()
        for items in data_items_by_block.values():
            expected_item_ids.update(item.id for item in items)
        assert saved_item_ids == expected_item_ids
        
        # Act 2: Clear everything (simulate switching songs)
        mock_data_item_repo.delete_by_project.return_value = len(saved_snapshot.data_items)
        
        # Track what gets restored
        restored_data_items = []
        restored_local_states = {}
        
        def track_create(item):
            restored_data_items.append(item)
            return item
        
        def track_set_inputs(block_id, local_state):
            restored_local_states[block_id] = local_state
        
        mock_data_item_repo.create.side_effect = track_create
        mock_local_state_repo.set_inputs.side_effect = track_set_inputs
        
        # Mock ProjectService._build_data_item_from_dict to return actual DataItem instances
        def mock_build_data_item_from_dict(data):
            """Build actual DataItem from dict"""
            item_type = (data.get("type") or "").lower()
            if item_type == "audio":
                return AudioDataItem(
                    id=data.get("id", ""),
                    block_id=data.get("block_id", ""),
                    name=data.get("name", "AudioItem"),
                    type="Audio",
                    created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else datetime.now(timezone.utc),
                    file_path=data.get("file_path"),
                    sample_rate=data.get("sample_rate"),
                    length_ms=data.get("length_ms")
                )
            elif item_type == "event":
                return EventDataItem(
                    id=data.get("id", ""),
                    block_id=data.get("block_id", ""),
                    name=data.get("name", "EventItem"),
                    type="Event",
                    created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else datetime.now(timezone.utc),
                    event_count=data.get("event_count", 0)
                )
            else:
                from src.shared.domain.entities import DataItem
                return DataItem(
                    id=data.get("id", ""),
                    block_id=data.get("block_id", ""),
                    name=data.get("name", "DataItem"),
                    type=data.get("type", "Data"),
                    created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else datetime.now(timezone.utc),
                    file_path=data.get("file_path"),
                    metadata=data.get("metadata", {})
                )
        
        # Mock the helper's _build_data_item_from_dict method
        snapshot_service._state_helper._build_data_item_from_dict = mock_build_data_item_from_dict
        
        # Track block metadata updates
        restored_metadata = {}
        
        def track_update(block):
            restored_metadata[block.id] = block.metadata.copy()
            return block
        mock_block_repo.update.side_effect = track_update
        
        # Mock apply_block_overrides to track metadata restoration
        original_apply = snapshot_service.apply_block_overrides
        applied_overrides = {}
        
        def track_apply_overrides(proj_id, overrides):
            applied_overrides.update(overrides)
            return original_apply(proj_id, overrides)
        
        snapshot_service.apply_block_overrides = track_apply_overrides
        
        # Act 3: Restore snapshot
        snapshot_service.restore_snapshot(project_id, saved_snapshot)
        
        # Assert: Verify all data items were restored
        assert len(restored_data_items) == len(saved_snapshot.data_items)
        restored_item_ids = {item.id for item in restored_data_items}
        assert restored_item_ids == expected_item_ids
        
        # Verify data items have correct block_id and properties
        for item in restored_data_items:
            assert item.block_id in [b.id for b in blocks]
            # Verify item properties match original
            original_item = next(
                (orig for items in data_items_by_block.values() for orig in items if orig.id == item.id),
                None
            )
            assert original_item is not None
            assert item.name == original_item.name
            assert item.type == original_item.type
            if hasattr(item, 'file_path') and hasattr(original_item, 'file_path'):
                assert item.file_path == original_item.file_path
        
        # Assert: Verify all local states were restored (only non-empty ones)
        assert len(restored_local_states) == len(saved_snapshot.block_local_state)
        for block_id, expected_local_state in local_state_by_block.items():
            if expected_local_state:  # Only verify non-empty local states
                assert block_id in restored_local_states
                assert restored_local_states[block_id] == expected_local_state
        
        # Assert: Verify block metadata was restored via apply_block_overrides
        assert len(applied_overrides) == len(blocks)
        for block in blocks:
            assert block.id in applied_overrides
            # Verify metadata matches exactly
            assert applied_overrides[block.id] == block.metadata
            # Verify metadata was applied to block (via update)
            assert block.id in restored_metadata
            assert restored_metadata[block.id] == block.metadata
        
        # Verify data items were cleared first
        mock_data_item_repo.delete_by_project.assert_called_once_with(project_id)
        
        # Verify all blocks were updated (once for clearing metadata, once for applying overrides)
        # Each block gets updated twice: once when clearing metadata, once when applying overrides
        assert mock_block_repo.update.call_count == len(blocks) * 2
        
        # Verify the final metadata matches what was saved
        for block in blocks:
            assert block.id in restored_metadata
            assert restored_metadata[block.id] == block.metadata

