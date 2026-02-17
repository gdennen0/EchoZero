"""
Unit tests for BlockStateHelper

Tests unified block state access (read + restore) functionality.
"""
import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
from datetime import datetime, timezone

from src.features.blocks.application.block_state_helper import BlockStateHelper
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem, AudioDataItem, EventDataItem
from src.shared.domain.value_objects.port_type import PortType


class TestBlockStateHelper:
    """Test BlockStateHelper implementation"""
    
    @pytest.fixture
    def mock_block_repo(self):
        """Create mock block repository"""
        return Mock()
    
    @pytest.fixture
    def mock_local_state_repo(self):
        """Create mock block local state repository"""
        return Mock()
    
    @pytest.fixture
    def mock_data_item_repo(self):
        """Create mock data item repository"""
        return Mock()
    
    @pytest.fixture
    def mock_project_service(self):
        """Create mock project service"""
        return Mock()
    
    @pytest.fixture
    def helper(
        self,
        mock_block_repo,
        mock_local_state_repo,
        mock_data_item_repo,
        mock_project_service
    ):
        """Create BlockStateHelper instance with mocked dependencies"""
        return BlockStateHelper(
            mock_block_repo,
            mock_local_state_repo,
            mock_data_item_repo,
            mock_project_service
        )
    
    @pytest.fixture
    def sample_block(self):
        """Create sample block for testing"""
        return Block(
            id="block-1",
            project_id="project-1",
            name="TestBlock",
            type="TestType",
            metadata={"setting1": "value1", "setting2": 42}
        )
    
    @pytest.fixture
    def sample_data_items(self):
        """Create sample data items for testing"""
        return [
            AudioDataItem(
                id="item-1",
                block_id="block-1",
                name="Audio1",
                type="Audio",
                created_at=datetime.now(timezone.utc),
                file_path="/path/to/audio.wav",
                sample_rate=44100,
                length_ms=1000.0
            ),
            EventDataItem(
                id="item-2",
                block_id="block-1",
                name="Events1",
                type="Event",
                created_at=datetime.now(timezone.utc),
                event_count=5
            )
        ]
    
    def test_get_block_state_success(self, helper, mock_block_repo, mock_local_state_repo, 
                                     mock_data_item_repo, sample_block, sample_data_items):
        """Test successful block state retrieval"""
        # Arrange
        block_id = "block-1"
        local_state = {"input1": "item-1", "input2": ["item-2"]}
        
        mock_block_repo.get_by_id.return_value = sample_block
        mock_local_state_repo.get_inputs.return_value = local_state
        mock_data_item_repo.list_by_block.return_value = sample_data_items
        
        # Act
        result = helper.get_block_state(block_id)
        
        # Assert
        assert result["block_id"] == block_id
        assert result["settings"] == sample_block.metadata
        assert result["local_state"] == local_state
        assert len(result["data_items"]) == 2
        assert result["data_items"][0]["id"] == "item-1"
        assert result["data_items"][1]["id"] == "item-2"
        
        mock_block_repo.get_by_id.assert_called_once_with(block_id)
        mock_local_state_repo.get_inputs.assert_called_once_with(block_id)
        mock_data_item_repo.list_by_block.assert_called_once_with(block_id)
    
    def test_get_block_state_no_local_state(self, helper, mock_block_repo, mock_local_state_repo,
                                           mock_data_item_repo, sample_block):
        """Test block state retrieval when no local state exists"""
        # Arrange
        block_id = "block-1"
        
        mock_block_repo.get_by_id.return_value = sample_block
        mock_local_state_repo.get_inputs.return_value = None
        mock_data_item_repo.list_by_block.return_value = []
        
        # Act
        result = helper.get_block_state(block_id)
        
        # Assert
        assert result["local_state"] == {}
        assert result["data_items"] == []
    
    def test_get_block_state_no_metadata(self, helper, mock_block_repo, mock_local_state_repo,
                                        mock_data_item_repo):
        """Test block state retrieval when block has no metadata"""
        # Arrange
        block_id = "block-1"
        block = Block(
            id=block_id,
            project_id="project-1",
            name="TestBlock",
            type="TestType",
            metadata=None
        )
        
        mock_block_repo.get_by_id.return_value = block
        mock_local_state_repo.get_inputs.return_value = {}
        mock_data_item_repo.list_by_block.return_value = []
        
        # Act
        result = helper.get_block_state(block_id)
        
        # Assert
        assert result["settings"] == {}
    
    def test_get_block_state_block_not_found(self, helper, mock_block_repo):
        """Test block state retrieval when block doesn't exist"""
        # Arrange
        block_id = "nonexistent"
        mock_block_repo.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Block not found"):
            helper.get_block_state(block_id)
    
    def test_get_project_state_success(self, helper, mock_block_repo, mock_local_state_repo,
                                      mock_data_item_repo, sample_block):
        """Test successful project state retrieval"""
        # Arrange
        project_id = "project-1"
        block1 = sample_block
        block2 = Block(
            id="block-2",
            project_id=project_id,
            name="Block2",
            type="TestType",
            metadata={"setting": "value"}
        )
        
        mock_block_repo.list_by_project.return_value = [block1, block2]
        mock_block_repo.get_by_id.side_effect = lambda bid: block1 if bid == "block-1" else block2
        mock_local_state_repo.get_inputs.return_value = {}
        mock_data_item_repo.list_by_block.return_value = []
        
        # Act
        result = helper.get_project_state(project_id)
        
        # Assert
        assert len(result) == 2
        assert "block-1" in result
        assert "block-2" in result
        assert result["block-1"]["block_id"] == "block-1"
        assert result["block-2"]["block_id"] == "block-2"
        
        mock_block_repo.list_by_project.assert_called_once_with(project_id)
    
    def test_get_project_state_empty_project(self, helper, mock_block_repo):
        """Test project state retrieval for empty project"""
        # Arrange
        project_id = "project-1"
        mock_block_repo.list_by_project.return_value = []
        
        # Act
        result = helper.get_project_state(project_id)
        
        # Assert
        assert result == {}
    
    def test_restore_block_state_success(self, helper, mock_local_state_repo, mock_data_item_repo,
                                        mock_project_service):
        """Test successful block state restoration"""
        # Arrange
        block_id = "block-1"
        state = {
            "local_state": {"input1": "item-1"},
            "data_items": [
                {
                    "id": "item-1",
                    "block_id": block_id,
                    "name": "Audio1",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": "/path/to/audio.wav"
                }
            ]
        }
        
        # Mock ProjectService deserialization
        mock_data_item = Mock(spec=AudioDataItem)
        mock_data_item.file_path = "/path/to/audio.wav"
        mock_project_service._build_data_item_from_dict.return_value = mock_data_item
        
        # Act
        helper.restore_block_state(block_id, state)
        
        # Assert
        mock_local_state_repo.set_inputs.assert_called_once_with(block_id, {"input1": "item-1"})
        mock_data_item_repo.create.assert_called_once()
        assert mock_data_item_repo.create.call_args[0][0] == mock_data_item
    
    def test_restore_block_state_no_local_state(self, helper, mock_local_state_repo, 
                                                mock_data_item_repo, mock_project_service):
        """Test block state restoration when no local state provided"""
        # Arrange
        block_id = "block-1"
        state = {
            "data_items": []
        }
        
        # Act
        helper.restore_block_state(block_id, state)
        
        # Assert
        mock_local_state_repo.set_inputs.assert_not_called()
        mock_data_item_repo.create.assert_not_called()
    
    def test_restore_block_state_with_project_dir(self, helper, mock_local_state_repo,
                                                  mock_data_item_repo, mock_project_service):
        """Test block state restoration with project directory for path resolution"""
        # Arrange
        block_id = "block-1"
        project_dir = Path("/project/dir")
        state = {
            "local_state": {},
            "data_items": [
                {
                    "id": "item-1",
                    "block_id": block_id,
                    "name": "Audio1",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": "relative/path/audio.wav"  # Relative path
                }
            ]
        }
        
        # Mock ProjectService deserialization
        mock_data_item = Mock(spec=AudioDataItem)
        mock_data_item.file_path = "relative/path/audio.wav"
        mock_project_service._build_data_item_from_dict.return_value = mock_data_item
        
        # Act
        helper.restore_block_state(block_id, state, project_dir)
        
        # Assert
        assert mock_data_item.file_path == str(project_dir / "relative/path/audio.wav")
        mock_data_item_repo.create.assert_called_once()
    
    def test_restore_block_state_absolute_path(self, helper, mock_local_state_repo,
                                              mock_data_item_repo, mock_project_service):
        """Test block state restoration preserves absolute paths"""
        # Arrange
        block_id = "block-1"
        project_dir = Path("/project/dir")
        absolute_path = "/absolute/path/audio.wav"
        state = {
            "local_state": {},
            "data_items": [
                {
                    "id": "item-1",
                    "block_id": block_id,
                    "name": "Audio1",
                    "type": "Audio",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": absolute_path
                }
            ]
        }
        
        # Mock ProjectService deserialization
        mock_data_item = Mock(spec=AudioDataItem)
        mock_data_item.file_path = absolute_path
        mock_project_service._build_data_item_from_dict.return_value = mock_data_item
        
        # Act
        helper.restore_block_state(block_id, state, project_dir)
        
        # Assert - absolute path should not be modified
        assert mock_data_item.file_path == absolute_path
        mock_data_item_repo.create.assert_called_once()
    
    def test_restore_project_state_success(self, helper, mock_local_state_repo,
                                          mock_data_item_repo, mock_project_service):
        """Test successful project state restoration"""
        # Arrange
        project_id = "project-1"
        project_state = {
            "block-1": {
                "local_state": {"input1": "item-1"},
                "data_items": []
            },
            "block-2": {
                "local_state": {"input2": "item-2"},  # Non-empty to trigger set_inputs
                "data_items": []
            }
        }
        
        # Act
        helper.restore_project_state(project_id, project_state)
        
        # Assert
        assert mock_local_state_repo.set_inputs.call_count == 2
        mock_local_state_repo.set_inputs.assert_any_call("block-1", {"input1": "item-1"})
        mock_local_state_repo.set_inputs.assert_any_call("block-2", {"input2": "item-2"})
    
    def test_build_data_item_from_dict_with_project_service(self, helper, mock_project_service):
        """Test data item deserialization using ProjectService"""
        # Arrange
        item_dict = {
            "id": "item-1",
            "block_id": "block-1",
            "name": "TestItem",
            "type": "Audio",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        mock_item = Mock(spec=AudioDataItem)
        mock_project_service._build_data_item_from_dict.return_value = mock_item
        
        # Act
        result = helper._build_data_item_from_dict(item_dict)
        
        # Assert
        assert result == mock_item
        mock_project_service._build_data_item_from_dict.assert_called_once_with(item_dict)
    
    def test_build_data_item_from_dict_fallback_audio(self, helper):
        """Test data item deserialization fallback for audio items"""
        # Arrange
        helper._project_service = None  # No ProjectService
        item_dict = {
            "id": "item-1",
            "block_id": "block-1",
            "name": "Audio1",
            "type": "Audio",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file_path": "/path/to/audio.wav",
            "sample_rate": 44100,
            "length_ms": 1000.0
        }
        
        # Act
        result = helper._build_data_item_from_dict(item_dict)
        
        # Assert
        assert isinstance(result, AudioDataItem)
        assert result.id == "item-1"
        assert result.block_id == "block-1"
        assert result.name == "Audio1"
        assert result.sample_rate == 44100
    
    def test_build_data_item_from_dict_fallback_event(self, helper):
        """Test data item deserialization fallback for event items"""
        # Arrange
        helper._project_service = None  # No ProjectService
        # EventDataItem.from_dict calculates event_count from events list
        item_dict = {
            "id": "item-1",
            "block_id": "block-1",
            "name": "Events1",
            "type": "Event",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "events": [
                {"time": 0.0, "classification": "kick"},
                {"time": 0.5, "classification": "snare"},
                {"time": 1.0, "classification": "kick"},
                {"time": 1.5, "classification": "snare"},
                {"time": 2.0, "classification": "kick"}
            ]
        }
        
        # Act
        result = helper._build_data_item_from_dict(item_dict)
        
        # Assert
        assert isinstance(result, EventDataItem)
        assert result.id == "item-1"
        assert result.event_count == 5  # Calculated from events list
    
    def test_build_data_item_from_dict_fallback_default(self, helper):
        """Test data item deserialization fallback for default items"""
        # Arrange
        helper._project_service = None  # No ProjectService
        item_dict = {
            "id": "item-1",
            "block_id": "block-1",
            "name": "DataItem",
            "type": "Data",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file_path": "/path/to/data",
            "metadata": {"key": "value"}
        }
        
        # Act
        result = helper._build_data_item_from_dict(item_dict)
        
        # Assert
        assert isinstance(result, DataItem)
        assert result.id == "item-1"
        assert result.type == "Data"
        assert result.metadata == {"key": "value"}

