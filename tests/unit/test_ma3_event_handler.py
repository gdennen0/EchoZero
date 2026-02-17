"""
Unit tests for MA3EventHandler.

Tests handling of incoming event changes from MA3.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from PyQt6.QtCore import QObject

from src.features.show_manager.application.ma3_event_handler import (
    MA3EventHandler,
    MA3EventChange,
    MA3ChangeType,
)


def _make_mock_message(change_type: str, data: dict):
    """Helper to create mock MA3Message objects for testing."""
    message = Mock()
    message.change_type = change_type
    message.data = data
    return message


class TestMA3EventChange:
    """Tests for MA3EventChange dataclass."""
    
    def test_from_message_event_added(self):
        """Test creating MA3EventChange from event.added message."""
        message = _make_mock_message('added', {
            'tc': '101',
            'tg': '1',
            'tr': '2',
            'idx': '5',
            'time': '1.5',
            'name': 'Kick',
            'cmd': 'Go+',
        })
        
        change = MA3EventChange.from_message(message)
        
        assert change is not None
        assert change.change_type == MA3ChangeType.EVENT_ADDED
        assert change.timecode_no == 101
        assert change.track_group == 1
        assert change.track == 2
        assert change.event_index == 5
        assert change.time == 1.5
        assert change.name == 'Kick'
        assert change.cmd == 'Go+'
        assert change.track_coord == 'tc101_tg1_tr2'
    
    def test_from_message_event_moved(self):
        """Test creating MA3EventChange from event.moved message."""
        message = _make_mock_message('moved', {
            'tc': '101',
            'tg': '1',
            'tr': '3',
            'idx': '2',
            'old_time': '1.5',
            'new_time': '2.0',
        })
        
        change = MA3EventChange.from_message(message)
        
        assert change is not None
        assert change.change_type == MA3ChangeType.EVENT_MOVED
        assert change.old_time == 1.5
        assert change.new_time == 2.0
    
    def test_from_message_track_hooked(self):
        """Test creating MA3EventChange from track.hooked message."""
        message = _make_mock_message('hooked', {
            'tc': '101',
            'tg': '1',
            'tr': '1',
            'name': 'Kicks',
        })
        
        change = MA3EventChange.from_message(message)
        
        assert change is not None
        assert change.change_type == MA3ChangeType.TRACK_HOOKED
        assert change.name == 'Kicks'
    
    def test_from_message_invalid_change(self):
        """Test from_message returns None for invalid change type."""
        message = _make_mock_message('invalid', {'tc': '101'})
        
        change = MA3EventChange.from_message(message)
        
        assert change is None
    
    def test_from_message_missing_fields(self):
        """Test from_message handles missing fields with defaults."""
        message = _make_mock_message('added', {})
        
        change = MA3EventChange.from_message(message)
        
        assert change is not None
        assert change.timecode_no == 0
        assert change.track_group == 0
        assert change.track == 0
        assert change.event_index == 0
        assert change.time == 0.0


class TestMA3EventHandler:
    """Tests for MA3EventHandler."""
    
    @pytest.fixture
    def mock_facade(self):
        """Create mock ApplicationFacade."""
        facade = Mock()
        facade.command_bus = Mock()
        facade.data_item_repo = Mock()
        return facade
    
    @pytest.fixture
    def handler(self, mock_facade):
        """Create MA3EventHandler instance."""
        return MA3EventHandler(
            facade=mock_facade,
            show_manager_block_id="show_manager_1"
        )
    
    def test_initialization(self, handler):
        """Test handler initializes correctly."""
        assert handler._show_manager_block_id == "show_manager_1"
        assert handler._sync_manager is None
        assert len(handler._editor_apis) == 0
        assert len(handler._hooked_tracks) == 0
    
    def test_set_sync_manager(self, handler):
        """Test setting sync manager reference."""
        mock_manager = Mock()
        handler.set_sync_manager(mock_manager)
        
        assert handler._sync_manager is mock_manager
    
    def test_register_editor_api(self, handler):
        """Test registering EditorAPI."""
        mock_api = Mock()
        handler.register_editor_api("editor_1", mock_api)
        
        assert "editor_1" in handler._editor_apis
        assert handler._editor_apis["editor_1"] is mock_api
    
    def test_handle_track_hooked(self, handler):
        """Test handling track hooked message."""
        change = MA3EventChange(
            change_type=MA3ChangeType.TRACK_HOOKED,
            timecode_no=101,
            track_group=1,
            track=1,
            name="Kicks"
        )
        
        handler._handle_track_hooked(change)
        
        assert "tc101_tg1_tr1" in handler._hooked_tracks
        assert handler._hooked_tracks["tc101_tg1_tr1"]["name"] == "Kicks"
    
    def test_handle_track_unhooked(self, handler):
        """Test handling track unhooked message."""
        # First hook a track
        hook_change = MA3EventChange(
            change_type=MA3ChangeType.TRACK_HOOKED,
            timecode_no=101,
            track_group=1,
            track=1,
            name="Kicks"
        )
        handler._handle_track_hooked(hook_change)
        
        # Then unhook it
        unhook_change = MA3EventChange(
            change_type=MA3ChangeType.TRACK_UNHOOKED,
            timecode_no=101,
            track_group=1,
            track=1
        )
        handler._handle_track_unhooked(unhook_change)
        
        assert "tc101_tg1_tr1" not in handler._hooked_tracks
    
    def test_handle_event_added_no_sync(self, handler):
        """Test handling event added when track not synced."""
        change = MA3EventChange(
            change_type=MA3ChangeType.EVENT_ADDED,
            timecode_no=101,
            track_group=1,
            track=1,
            time=1.5,
            name="Kick"
        )
        
        # No controller set, so no sync should happen
        handler._handle_event_added(change)
        
        # Should not crash, just log that track is not synced
    
    def test_handle_event_added_with_sync(self, handler, mock_facade):
        """Test handling event added when track is synced."""
        # Set up sync manager with synced entity
        mock_entity = Mock()
        mock_entity.editor_layer_id = "kicks"
        mock_entity.editor_block_id = "editor_1"
        
        mock_manager = Mock()
        mock_manager.get_synced_layer_by_ma3_coord.return_value = mock_entity
        handler.set_sync_manager(mock_manager)
        
        # Set up mock EditorAPI
        mock_editor_api = Mock()
        handler.register_editor_api("editor_1", mock_editor_api)
        
        change = MA3EventChange(
            change_type=MA3ChangeType.EVENT_ADDED,
            timecode_no=101,
            track_group=1,
            track=1,
            event_index=5,
            time=1.5,
            name="Kick",
            cmd="Go+"
        )
        
        handler._handle_event_added(change)
        
        # Verify add_event was called
        mock_editor_api.add_event.assert_called_once()
        call_kwargs = mock_editor_api.add_event.call_args[1]
        assert call_kwargs['time'] == 1.5
        assert call_kwargs['classification'] == 'kicks'
        assert call_kwargs['source'] == 'ma3_live'
    
    def test_get_hooked_tracks(self, handler):
        """Test getting hooked tracks."""
        # Hook some tracks
        handler._hooked_tracks = {
            "tc101_tg1_tr1": {"name": "Kicks", "timecode_no": 101, "track_group": 1, "track": 1},
            "tc101_tg1_tr2": {"name": "Snares", "timecode_no": 101, "track_group": 1, "track": 2},
        }
        
        hooked = handler.get_hooked_tracks()
        
        assert len(hooked) == 2
        assert "tc101_tg1_tr1" in hooked
        assert hooked["tc101_tg1_tr1"]["name"] == "Kicks"
    
    def test_cleanup(self, handler):
        """Test cleanup clears all state."""
        handler._hooked_tracks = {"tc101_tg1_tr1": {}}
        handler._editor_apis = {"editor_1": Mock()}
        handler._sync_manager = Mock()
        
        handler.cleanup()
        
        assert len(handler._hooked_tracks) == 0
        assert len(handler._editor_apis) == 0
        assert handler._sync_manager is None
    
    def test_event_received_signal(self, handler):
        """Test event_received signal is emitted."""
        received_changes = []
        handler.event_received.connect(lambda c: received_changes.append(c))
        
        # Create a mock message with proper format (change_type attr + data dict)
        mock_message = _make_mock_message('added', {
            'tc': '101',
            'tg': '1',
            'tr': '1',
            'idx': '1',
            'time': '1.0',
        })
        
        handler.handle_event_message(mock_message)
        
        assert len(received_changes) == 1
        assert received_changes[0].change_type == MA3ChangeType.EVENT_ADDED


class TestMA3ChangeType:
    """Tests for MA3ChangeType enum."""
    
    def test_event_types(self):
        """Test event change types exist."""
        assert MA3ChangeType.EVENT_ADDED is not None
        assert MA3ChangeType.EVENT_MODIFIED is not None
        assert MA3ChangeType.EVENT_DELETED is not None
        assert MA3ChangeType.EVENT_MOVED is not None
    
    def test_track_types(self):
        """Test track change types exist."""
        assert MA3ChangeType.TRACK_HOOKED is not None
        assert MA3ChangeType.TRACK_UNHOOKED is not None
        assert MA3ChangeType.TRACK_RENAMED is not None
