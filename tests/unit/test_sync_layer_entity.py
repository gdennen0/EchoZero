"""
Tests for SyncLayerEntity

Tests the unified sync layer entity's serialization, factory methods, and state management.
"""
import pytest
from datetime import datetime

from src.features.show_manager.domain.sync_layer_entity import (
    SyncLayerEntity,
    SyncLayerSettings,
    SyncSource,
    SyncStatus,
    SyncDirection,
    ConflictStrategy,
)


class TestSyncLayerSettings:
    """Tests for SyncLayerSettings dataclass."""
    
    def test_default_values(self):
        """Test default settings values."""
        settings = SyncLayerSettings()
        
        assert settings.direction == SyncDirection.BIDIRECTIONAL
        assert settings.conflict_strategy == ConflictStrategy.PROMPT_USER
        assert settings.auto_apply is False
        assert settings.apply_updates_enabled is True
        assert settings.sync_on_change is True
        assert settings.sequence_no == 1
    
    def test_to_dict(self):
        """Test settings serialization."""
        settings = SyncLayerSettings(
            direction=SyncDirection.MA3_TO_EZ,
            conflict_strategy=ConflictStrategy.MA3_WINS,
            auto_apply=True,
            sequence_no=42,
        )
        
        data = settings.to_dict()
        
        assert data["direction"] == "ma3_to_ez"
        assert data["conflict_strategy"] == "ma3_wins"
        assert data["auto_apply"] is True
        assert data["sequence_no"] == 42
    
    def test_from_dict(self):
        """Test settings deserialization."""
        data = {
            "direction": "ez_to_ma3",
            "conflict_strategy": "ez_wins",
            "auto_apply": True,
            "apply_updates_enabled": False,
            "sync_on_change": False,
            "sequence_no": 99,
        }
        
        settings = SyncLayerSettings.from_dict(data)
        
        assert settings.direction == SyncDirection.EZ_TO_MA3
        assert settings.conflict_strategy == ConflictStrategy.EZ_WINS
        assert settings.auto_apply is True
        assert settings.apply_updates_enabled is False
        assert settings.sync_on_change is False
        assert settings.sequence_no == 99
    
    def test_from_dict_legacy_values(self):
        """Test settings deserialization with legacy value names."""
        data = {
            "direction": "MA3_TO_EDITOR",  # Legacy format
            "conflict_strategy": "USE_MA3",  # Legacy format
        }
        
        settings = SyncLayerSettings.from_dict(data)
        
        assert settings.direction == SyncDirection.MA3_TO_EZ
        assert settings.conflict_strategy == ConflictStrategy.MA3_WINS
    
    def test_from_dict_empty(self):
        """Test settings deserialization from empty dict uses defaults."""
        settings = SyncLayerSettings.from_dict({})
        
        assert settings.direction == SyncDirection.BIDIRECTIONAL
        assert settings.conflict_strategy == ConflictStrategy.PROMPT_USER


class TestSyncLayerEntityBasic:
    """Tests for SyncLayerEntity basic functionality."""
    
    def test_create_ma3_sourced(self):
        """Test creating MA3-sourced entity."""
        entity = SyncLayerEntity(
            id="test-uuid-1",
            source=SyncSource.MA3,
            name="ma3_Kick",
            ma3_coord="tc1_tg1_tr1",
            ma3_timecode_no=1,
            ma3_track_group=1,
            ma3_track=1,
        )
        
        assert entity.id == "test-uuid-1"
        assert entity.source == SyncSource.MA3
        assert entity.name == "ma3_Kick"
        assert entity.ma3_coord == "tc1_tg1_tr1"
        assert entity.sync_status == SyncStatus.UNMAPPED
    
    def test_create_editor_sourced(self):
        """Test creating Editor-sourced entity."""
        entity = SyncLayerEntity(
            id="test-uuid-2",
            source=SyncSource.EDITOR,
            name="ez_Snare",
            editor_layer_id="snare_layer",
            editor_block_id="editor_1",
        )
        
        assert entity.id == "test-uuid-2"
        assert entity.source == SyncSource.EDITOR
        assert entity.name == "ez_Snare"
        assert entity.editor_layer_id == "snare_layer"
    
    def test_ma3_sourced_requires_coord(self):
        """Test that MA3-sourced entity requires ma3_coord."""
        with pytest.raises(ValueError, match="MA3-sourced entity requires ma3_coord"):
            SyncLayerEntity(
                id="test-uuid",
                source=SyncSource.MA3,
                name="test",
                # Missing ma3_coord
            )
    
    def test_editor_sourced_requires_layer_id(self):
        """Test that Editor-sourced entity requires editor_layer_id."""
        with pytest.raises(ValueError, match="Editor-sourced entity requires editor_layer_id"):
            SyncLayerEntity(
                id="test-uuid",
                source=SyncSource.EDITOR,
                name="test",
                # Missing editor_layer_id
            )


class TestSyncLayerEntityFactoryMethods:
    """Tests for SyncLayerEntity factory methods."""
    
    def test_from_ma3_track(self):
        """Test creating entity from MA3 track info."""
        entity = SyncLayerEntity.from_ma3_track(
            id="uuid-1",
            coord="tc1_tg2_tr3",
            timecode_no=1,
            track_group=2,
            track=3,
            name="Kick",
            group_name="Drums",
            event_count=42,
        )
        
        assert entity.id == "uuid-1"
        assert entity.source == SyncSource.MA3
        assert entity.name == "ma3_Kick"  # Prefixed
        assert entity.ma3_coord == "tc1_tg2_tr3"
        assert entity.ma3_timecode_no == 1
        assert entity.ma3_track_group == 2
        assert entity.ma3_track == 3
        assert entity.group_name == "Drums"
        assert entity.event_count == 42
    
    def test_from_ma3_track_auto_prefix(self):
        """Test that from_ma3_track adds ma3_ prefix if not present."""
        entity = SyncLayerEntity.from_ma3_track(
            id="uuid-1",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Snare",  # No prefix
        )
        
        assert entity.name == "ma3_Snare"
    
    def test_from_ma3_track_keeps_existing_prefix(self):
        """Test that from_ma3_track keeps existing ma3_ prefix."""
        entity = SyncLayerEntity.from_ma3_track(
            id="uuid-1",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="ma3_Snare",  # Already prefixed
        )
        
        assert entity.name == "ma3_Snare"  # Not double-prefixed
    
    def test_from_editor_layer(self):
        """Test creating entity from Editor layer info."""
        entity = SyncLayerEntity.from_editor_layer(
            id="uuid-2",
            layer_id="kick_layer",
            block_id="editor_1",
            name="Kick",
            group_name="Drums",
            event_count=100,
        )
        
        assert entity.id == "uuid-2"
        assert entity.source == SyncSource.EDITOR
        assert entity.name == "ez_Kick"  # Prefixed
        assert entity.editor_layer_id == "kick_layer"
        assert entity.editor_block_id == "editor_1"
        assert entity.group_name == "Drums"
        assert entity.event_count == 100
    
    def test_from_editor_layer_auto_prefix(self):
        """Test that from_editor_layer adds ez_ prefix if not present."""
        entity = SyncLayerEntity.from_editor_layer(
            id="uuid-1",
            layer_id="snare",
            block_id="editor_1",
            name="Snare",
        )
        
        assert entity.name == "ez_Snare"


class TestSyncLayerEntitySerialization:
    """Tests for SyncLayerEntity serialization."""
    
    def test_to_dict(self):
        """Test entity serialization."""
        entity = SyncLayerEntity(
            id="test-uuid",
            source=SyncSource.MA3,
            name="ma3_Test",
            ma3_coord="tc1_tg1_tr1",
            ma3_timecode_no=1,
            ma3_track_group=1,
            ma3_track=1,
            editor_layer_id="synced_layer",
            editor_block_id="editor_1",
            sync_status=SyncStatus.SYNCED,
            event_count=50,
            group_name="TestGroup",
        )
        
        data = entity.to_dict()
        
        assert data["id"] == "test-uuid"
        assert data["source"] == "ma3"
        assert data["name"] == "ma3_Test"
        assert data["ma3_coord"] == "tc1_tg1_tr1"
        assert data["editor_layer_id"] == "synced_layer"
        assert data["sync_status"] == "synced"
        assert data["event_count"] == 50
        assert data["group_name"] == "TestGroup"
        assert "settings" in data
    
    def test_from_dict(self):
        """Test entity deserialization."""
        data = {
            "id": "test-uuid",
            "source": "editor",
            "name": "ez_Test",
            "ma3_coord": "tc1_tg1_tr1",
            "ma3_timecode_no": 1,
            "ma3_track_group": 1,
            "ma3_track": 1,
            "editor_layer_id": "test_layer",
            "editor_block_id": "editor_1",
            "sync_status": "synced",
            "event_count": 25,
            "settings": {
                "direction": "bidirectional",
                "sequence_no": 5,
            },
            "group_name": "MyGroup",
        }
        
        entity = SyncLayerEntity.from_dict(data)
        
        assert entity.id == "test-uuid"
        assert entity.source == SyncSource.EDITOR
        assert entity.name == "ez_Test"
        assert entity.ma3_coord == "tc1_tg1_tr1"
        assert entity.editor_layer_id == "test_layer"
        assert entity.sync_status == SyncStatus.SYNCED
        assert entity.event_count == 25
        assert entity.settings.sequence_no == 5
        assert entity.group_name == "MyGroup"
    
    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization are inverse operations."""
        original = SyncLayerEntity.from_ma3_track(
            id="roundtrip-test",
            coord="tc2_tg3_tr4",
            timecode_no=2,
            track_group=3,
            track=4,
            name="RoundtripTest",
            event_count=99,
        )
        original.link_to_editor("synced_layer", "editor_1")
        original.mark_synced()
        original.settings.sequence_no = 42
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = SyncLayerEntity.from_dict(data)
        
        assert restored.id == original.id
        assert restored.source == original.source
        assert restored.name == original.name
        assert restored.ma3_coord == original.ma3_coord
        assert restored.editor_layer_id == original.editor_layer_id
        assert restored.sync_status == original.sync_status
        assert restored.event_count == original.event_count
        assert restored.settings.sequence_no == original.settings.sequence_no


class TestSyncLayerEntityStateMethods:
    """Tests for SyncLayerEntity state management methods."""
    
    def test_link_to_ma3(self):
        """Test linking Editor-sourced entity to MA3 track."""
        entity = SyncLayerEntity.from_editor_layer(
            id="test-uuid",
            layer_id="my_layer",
            block_id="editor_1",
            name="Test",
        )
        
        entity.link_to_ma3("tc1_tg1_tr1", 1, 1, 1)
        
        assert entity.ma3_coord == "tc1_tg1_tr1"
        assert entity.ma3_timecode_no == 1
        assert entity.ma3_track_group == 1
        assert entity.ma3_track == 1
        assert entity.has_ma3_side is True
    
    def test_link_to_editor(self):
        """Test linking MA3-sourced entity to Editor layer."""
        entity = SyncLayerEntity.from_ma3_track(
            id="test-uuid",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        
        entity.link_to_editor("synced_layer", "editor_1")
        
        assert entity.editor_layer_id == "synced_layer"
        assert entity.editor_block_id == "editor_1"
        assert entity.has_editor_side is True
    
    def test_mark_synced(self):
        """Test marking entity as synced."""
        entity = SyncLayerEntity.from_ma3_track(
            id="test-uuid",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        
        assert entity.sync_status == SyncStatus.UNMAPPED
        
        entity.mark_synced()
        
        assert entity.sync_status == SyncStatus.SYNCED
        assert entity.last_sync_time is not None
        assert entity.error_message is None
        assert entity.is_synced is True
    
    def test_mark_diverged(self):
        """Test marking entity as diverged."""
        entity = SyncLayerEntity.from_ma3_track(
            id="test-uuid",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        entity.mark_synced()
        
        entity.mark_diverged()
        
        assert entity.sync_status == SyncStatus.DIVERGED
        assert entity.is_synced is True  # DIVERGED is still considered "synced" (has both sides)
    
    def test_mark_error(self):
        """Test marking entity with error."""
        entity = SyncLayerEntity.from_ma3_track(
            id="test-uuid",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        
        entity.mark_error("Connection failed")
        
        assert entity.sync_status == SyncStatus.ERROR
        assert entity.error_message == "Connection failed"
        assert entity.is_synced is False


class TestSyncLayerEntityProperties:
    """Tests for SyncLayerEntity computed properties."""
    
    def test_display_name_ma3(self):
        """Test display name for MA3-sourced entity."""
        entity = SyncLayerEntity.from_ma3_track(
            id="test-uuid",
            coord="tc1_tg2_tr3",
            timecode_no=1,
            track_group=2,
            track=3,
            name="Kick",
        )
        
        display = entity.display_name
        
        assert "TC 1" in display
        assert "TG 2" in display
        assert "Track 3" in display
    
    def test_display_name_editor(self):
        """Test display name for Editor-sourced entity."""
        entity = SyncLayerEntity.from_editor_layer(
            id="test-uuid",
            layer_id="kick_layer",
            block_id="editor_1",
            name="Kick",
        )
        
        assert entity.display_name == "ez_Kick"
    
    def test_has_ma3_side(self):
        """Test has_ma3_side property."""
        entity = SyncLayerEntity.from_editor_layer(
            id="test-uuid",
            layer_id="layer",
            block_id="editor",
            name="Test",
        )
        
        assert entity.has_ma3_side is False
        
        entity.link_to_ma3("tc1_tg1_tr1", 1, 1, 1)
        
        assert entity.has_ma3_side is True
    
    def test_has_editor_side(self):
        """Test has_editor_side property."""
        entity = SyncLayerEntity.from_ma3_track(
            id="test-uuid",
            coord="tc1_tg1_tr1",
            timecode_no=1,
            track_group=1,
            track=1,
            name="Test",
        )
        
        assert entity.has_editor_side is False
        
        entity.link_to_editor("layer", "editor")
        
        assert entity.has_editor_side is True
