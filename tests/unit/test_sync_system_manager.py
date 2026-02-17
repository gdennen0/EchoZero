"""
Tests for SyncSystemManager

Tests the sync system manager's orchestration of sync operations.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from PyQt6.QtWidgets import QApplication

from src.features.show_manager.application.sync_system_manager import SyncSystemManager
from src.features.show_manager.domain.sync_layer_entity import (
    SyncLayerEntity,
    SyncSource,
    SyncStatus,
)


# Ensure QApplication exists for PyQt signals
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_facade():
    """Create mock ApplicationFacade."""
    facade = MagicMock()
    facade.command_bus = MagicMock()
    facade.ma3_comm_service = None
    return facade


@pytest.fixture
def mock_settings_manager():
    """Create mock ShowManagerSettingsManager."""
    settings = MagicMock()
    settings.synced_layers = []
    return settings


@pytest.fixture
def manager(mock_facade, mock_settings_manager, qapp):
    """Create SyncSystemManager instance."""
    return SyncSystemManager(
        facade=mock_facade,
        show_manager_block_id="show_manager_1",
        settings_manager=mock_settings_manager,
    )


class TestSyncSystemManagerInit:
    """Tests for manager initialization."""
    
    def test_init(self, manager):
        """Test manager initializes correctly."""
        assert manager._show_manager_block_id == "show_manager_1"
        assert manager._synced_layers == {}
        assert manager._ma3_connected is False
    
    def test_init_loads_from_settings(self, mock_facade, qapp):
        """Test manager loads synced layers from settings on init."""
        settings = MagicMock()
        settings.synced_layers = [
            {
                "id": "test-1",
                "source": "ma3",
                "name": "ma3_Test",
                "ma3_coord": "tc1_tg1_tr1",
                "ma3_timecode_no": 1,
                "ma3_track_group": 1,
                "ma3_track": 1,
                "sync_status": "synced",
                "settings": {},
            }
        ]
        
        manager = SyncSystemManager(
            facade=mock_facade,
            show_manager_block_id="sm_1",
            settings_manager=settings,
        )
        
        assert len(manager._synced_layers) == 1
        assert "test-1" in manager._synced_layers


class TestSyncSystemManagerSyncOperations:
    """Tests for sync/unsync operations."""
    
    def test_sync_layer_ma3_creates_entity(self, manager, mock_settings_manager):
        """Test syncing MA3 track creates entity."""
        # Mock getting MA3 track info
        with patch.object(manager, '_get_ma3_track_info') as mock_get_info:
            mock_get_info.return_value = {
                "coord": "tc1_tg1_tr1",
                "timecode_no": 1,
                "track_group": 1,
                "track": 1,
                "name": "Kick",
                "event_count": 10,
            }
            
            with patch.object(manager, '_create_editor_layer') as mock_create:
                mock_create.return_value = "ma3_Kick"
                
                with patch.object(manager, '_push_ma3_to_editor'):
                    entity_id = manager.sync_layer("ma3", "tc1_tg1_tr1")
        
        assert entity_id is not None
        assert entity_id in manager._synced_layers
        
        entity = manager._synced_layers[entity_id]
        assert entity.source == SyncSource.MA3
        assert entity.ma3_coord == "tc1_tg1_tr1"
    
    def test_sync_layer_editor_creates_entity(self, manager, mock_settings_manager):
        """Test syncing Editor layer creates entity."""
        # Mock getting Editor layer info
        with patch.object(manager, '_get_editor_layer_info') as mock_get_info:
            mock_get_info.return_value = {
                "layer_id": "kick_layer",
                "block_id": "editor_1",
                "name": "Kick",
                "event_count": 20,
            }
            
            with patch.object(manager, '_get_editor_block_id') as mock_block:
                mock_block.return_value = "editor_1"
                
                entity_id = manager.sync_layer("editor", "kick_layer")
        
        assert entity_id is not None
        assert entity_id in manager._synced_layers
        
        entity = manager._synced_layers[entity_id]
        assert entity.source == SyncSource.EDITOR
        assert entity.editor_layer_id == "kick_layer"
    
    def test_sync_layer_already_synced_returns_existing(self, manager):
        """Test syncing already-synced layer returns existing entity ID."""
        # Add existing entity
        existing = SyncLayerEntity.from_ma3_track(
            id="existing-id",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        manager._synced_layers["existing-id"] = existing
        
        # Try to sync same coord
        entity_id = manager.sync_layer("ma3", "tc1_tg1_tr1")
        
        assert entity_id == "existing-id"
        assert len(manager._synced_layers) == 1
    
    def test_unsync_layer_removes_entity(self, manager, mock_settings_manager):
        """Test unsyncing layer removes entity."""
        # Add entity
        entity = SyncLayerEntity.from_ma3_track(
            id="to-remove",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        manager._synced_layers["to-remove"] = entity
        
        with patch.object(manager, '_delete_editor_layer'):
            success = manager.unsync_layer("to-remove")
        
        assert success is True
        assert "to-remove" not in manager._synced_layers
    
    def test_unsync_layer_ma3_sourced_deletes_editor(self, manager, mock_settings_manager):
        """Test unsyncing MA3-sourced layer deletes from Editor."""
        # Add MA3-sourced entity with editor side
        entity = SyncLayerEntity.from_ma3_track(
            id="ma3-sourced",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        entity.link_to_editor("synced_layer", "editor_1")
        manager._synced_layers["ma3-sourced"] = entity
        
        with patch.object(manager, '_delete_editor_layer') as mock_delete:
            success = manager.unsync_layer("ma3-sourced")
        
        assert success is True
        mock_delete.assert_called_once_with("editor_1", "synced_layer")
    
    def test_unsync_layer_editor_sourced_keeps_ma3(self, manager, mock_settings_manager):
        """Test unsyncing Editor-sourced layer keeps MA3 copy."""
        # Add Editor-sourced entity with ma3 side
        entity = SyncLayerEntity.from_editor_layer(
            id="editor-sourced",
            layer_id="my_layer",
            block_id="editor_1",
            name="Test",
        )
        entity.link_to_ma3("tc1_tg1_tr1", 1, 1, 1)
        manager._synced_layers["editor-sourced"] = entity
        
        with patch.object(manager, '_delete_editor_layer') as mock_delete:
            success = manager.unsync_layer("editor-sourced")
        
        assert success is True
        # Should NOT delete editor layer (asymmetric behavior)
        mock_delete.assert_not_called()
    
    def test_unsync_nonexistent_returns_false(self, manager):
        """Test unsyncing nonexistent entity returns False."""
        success = manager.unsync_layer("does-not-exist")
        
        assert success is False


class TestSyncSystemManagerDataAccess:
    """Tests for data access methods."""
    
    def test_get_synced_layers(self, manager):
        """Test getting list of synced layers."""
        # Add some entities
        entity1 = SyncLayerEntity.from_ma3_track(
            id="e1", coord="tc1_tg1_tr1", timecode_no=1, track_group=1, track=1, name="A",
        )
        entity2 = SyncLayerEntity.from_editor_layer(
            id="e2", layer_id="layer", block_id="editor", name="B",
        )
        manager._synced_layers["e1"] = entity1
        manager._synced_layers["e2"] = entity2
        
        layers = manager.get_synced_layers()
        
        assert len(layers) == 2
        assert entity1 in layers
        assert entity2 in layers
    
    def test_get_synced_layer_by_id(self, manager):
        """Test getting synced layer by ID."""
        entity = SyncLayerEntity.from_ma3_track(
            id="find-me", coord="tc1_tg1_tr1", timecode_no=1, track_group=1, track=1, name="Test",
        )
        manager._synced_layers["find-me"] = entity
        
        found = manager.get_synced_layer("find-me")
        
        assert found is entity
    
    def test_get_synced_layer_not_found(self, manager):
        """Test getting nonexistent synced layer returns None."""
        found = manager.get_synced_layer("does-not-exist")
        
        assert found is None
    
    def test_get_synced_layer_by_ma3_coord(self, manager):
        """Test getting synced layer by MA3 coordinate."""
        entity = SyncLayerEntity.from_ma3_track(
            id="e1", coord="tc2_tg3_tr4", timecode_no=2, track_group=3, track=4, name="Test",
        )
        manager._synced_layers["e1"] = entity
        
        found = manager.get_synced_layer_by_ma3_coord("tc2_tg3_tr4")
        
        assert found is entity
    
    def test_get_synced_layer_by_editor_layer(self, manager):
        """Test getting synced layer by Editor layer ID."""
        entity = SyncLayerEntity.from_editor_layer(
            id="e1", layer_id="my_special_layer", block_id="editor", name="Test",
        )
        manager._synced_layers["e1"] = entity
        
        found = manager.get_synced_layer_by_editor_layer("my_special_layer")
        
        assert found is entity


class TestSyncSystemManagerSequence:
    """Tests for sequence management."""
    
    def test_set_sequence(self, manager, mock_settings_manager):
        """Test setting sequence number."""
        entity = SyncLayerEntity.from_ma3_track(
            id="e1", coord="tc1_tg1_tr1", timecode_no=1, track_group=1, track=1, name="Test",
        )
        entity.settings.sequence_no = 1
        manager._synced_layers["e1"] = entity
        
        success = manager.set_sequence("e1", 42)
        
        assert success is True
        assert entity.settings.sequence_no == 42
    
    def test_set_sequence_nonexistent_returns_false(self, manager):
        """Test setting sequence for nonexistent entity returns False."""
        success = manager.set_sequence("does-not-exist", 42)
        
        assert success is False


class TestSyncSystemManagerReconnection:
    """Tests for reconnection handling."""
    
    def test_on_ma3_connected(self, manager):
        """Test handling MA3 connection."""
        # Add synced entity with MA3 coord
        entity = SyncLayerEntity.from_ma3_track(
            id="e1", coord="tc1_tg1_tr1", timecode_no=1, track_group=1, track=1, name="Test",
        )
        manager._synced_layers["e1"] = entity
        
        with patch.object(manager, '_hook_ma3_track') as mock_hook:
            manager.on_ma3_connected()
        
        assert manager._ma3_connected is True
        mock_hook.assert_called_with("tc1_tg1_tr1")
    
    def test_on_ma3_disconnected(self, manager):
        """Test handling MA3 disconnection."""
        manager._ma3_connected = True
        manager._hooked_tracks = {"tc1_tg1_tr1": lambda: None}
        
        manager.on_ma3_disconnected()
        
        assert manager._ma3_connected is False
        assert len(manager._hooked_tracks) == 0


class TestSyncSystemManagerSignals:
    """Tests for signal emission."""
    
    def test_entities_changed_emitted_on_sync(self, manager, mock_settings_manager, qapp):
        """Test entities_changed signal emitted when syncing."""
        signal_received = []
        manager.entities_changed.connect(lambda: signal_received.append(True))
        
        with patch.object(manager, '_get_ma3_track_info') as mock_get_info:
            mock_get_info.return_value = {
                "coord": "tc1_tg1_tr1",
                "timecode_no": 1,
                "track_group": 1,
                "track": 1,
                "name": "Test",
            }
            with patch.object(manager, '_create_editor_layer', return_value="test"):
                with patch.object(manager, '_push_ma3_to_editor'):
                    manager.sync_layer("ma3", "tc1_tg1_tr1")
        
        assert len(signal_received) == 1
    
    def test_entities_changed_emitted_on_unsync(self, manager, mock_settings_manager, qapp):
        """Test entities_changed signal emitted when unsyncing."""
        # Add entity
        entity = SyncLayerEntity.from_ma3_track(
            id="e1", coord="tc1_tg1_tr1", timecode_no=1, track_group=1, track=1, name="Test",
        )
        manager._synced_layers["e1"] = entity
        
        signal_received = []
        manager.entities_changed.connect(lambda: signal_received.append(True))
        
        with patch.object(manager, '_delete_editor_layer'):
            manager.unsync_layer("e1")
        
        assert len(signal_received) == 1


class TestSyncSystemManagerPersistence:
    """Tests for settings persistence."""
    
    def test_save_to_settings(self, manager, mock_settings_manager):
        """Test saving synced layers to settings."""
        entity = SyncLayerEntity.from_ma3_track(
            id="e1", coord="tc1_tg1_tr1", timecode_no=1, track_group=1, track=1, name="Test",
        )
        manager._synced_layers["e1"] = entity
        
        manager._save_to_settings()
        
        # Check that settings_manager.synced_layers was set
        mock_settings_manager.synced_layers = [entity.to_dict()]
        assert len(mock_settings_manager.synced_layers) == 1
    
    def test_legacy_migration_ma3_entity(self, mock_facade, qapp):
        """Test migration of legacy MA3TrackEntity format."""
        legacy_data = {
            "coord": "tc1_tg1_tr1",
            "timecode_no": 1,
            "track_group": 1,
            "track": 1,
            "name": "LegacyMA3",
            "mapped_editor_layer_id": "synced_layer",
            "event_count": 50,
            "settings": {"sequence_no": 5},
        }
        
        settings = MagicMock()
        settings.synced_layers = [legacy_data]
        
        manager = SyncSystemManager(
            facade=mock_facade,
            show_manager_block_id="sm_1",
            settings_manager=settings,
        )
        
        assert len(manager._synced_layers) == 1
        entity = list(manager._synced_layers.values())[0]
        assert entity.source == SyncSource.MA3
        assert entity.ma3_coord == "tc1_tg1_tr1"
        assert entity.editor_layer_id == "synced_layer"
    
    def test_legacy_migration_editor_entity(self, mock_facade, qapp):
        """Test migration of legacy EditorLayerEntity format."""
        legacy_data = {
            "layer_id": "my_layer",
            "block_id": "editor_1",
            "name": "LegacyEditor",
            "mapped_ma3_track_id": "tc1_tg1_tr1",
            "event_count": 30,
            "settings": {},
        }
        
        settings = MagicMock()
        settings.synced_layers = [legacy_data]
        
        manager = SyncSystemManager(
            facade=mock_facade,
            show_manager_block_id="sm_1",
            settings_manager=settings,
        )
        
        assert len(manager._synced_layers) == 1
        entity = list(manager._synced_layers.values())[0]
        assert entity.source == SyncSource.EDITOR
        assert entity.editor_layer_id == "my_layer"
        assert entity.ma3_coord == "tc1_tg1_tr1"
