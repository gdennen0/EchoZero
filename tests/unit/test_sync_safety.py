"""
Unit tests for Sync Safety Framework.

Tests backup, validation, and safe sync operations.
"""
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.features.show_manager.application.sync_safety import (
    SyncAction,
    ValidationResult,
    EventSnapshot,
    LayerSnapshot,
    SyncResult,
    SyncBackupManager,
    SyncValidator,
    SafeSyncService,
)


class TestEventSnapshot:
    """Tests for EventSnapshot dataclass."""
    
    def test_from_event(self):
        """Test creating snapshot from event object."""
        event = Mock()
        event.id = "evt_1"
        event.time = 1.5
        event.duration = 0.1
        event.classification = "kick"
        event.metadata = {"source": "test"}
        
        snapshot = EventSnapshot.from_event(event)
        
        assert snapshot.id == "evt_1"
        assert snapshot.time == 1.5
        assert snapshot.duration == 0.1
        assert snapshot.classification == "kick"
        assert snapshot.metadata == {"source": "test"}
    
    def test_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = EventSnapshot(
            id="evt_1",
            time=1.5,
            duration=0.1,
            classification="kick",
            metadata={"source": "test"}
        )
        
        result = snapshot.to_dict()
        
        assert result["id"] == "evt_1"
        assert result["time"] == 1.5
        assert result["duration"] == 0.1
        assert result["classification"] == "kick"
        assert result["metadata"] == {"source": "test"}


class TestLayerSnapshot:
    """Tests for LayerSnapshot dataclass."""
    
    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        original = LayerSnapshot(
            layer_id="kicks",
            block_id="editor_1",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            events=[
                EventSnapshot(id="e1", time=1.0, duration=0.0, classification="kick"),
                EventSnapshot(id="e2", time=2.0, duration=0.0, classification="kick"),
            ],
            source="editor"
        )
        
        data = original.to_dict()
        restored = LayerSnapshot.from_dict(data)
        
        assert restored.layer_id == original.layer_id
        assert restored.block_id == original.block_id
        assert len(restored.events) == 2
        assert restored.source == "editor"


class TestSyncResult:
    """Tests for SyncResult dataclass."""
    
    def test_success_result(self):
        """Test creating success result."""
        result = SyncResult.success_result(SyncAction.ADD, "kicks", 5)
        
        assert result.success is True
        assert result.action == SyncAction.ADD
        assert result.layer_id == "kicks"
        assert result.events_affected == 5
        assert result.backup_created is True
    
    def test_failure_result(self):
        """Test creating failure result."""
        result = SyncResult.failure_result(
            SyncAction.DELETE, "kicks", "Event not found",
            ValidationResult.MISSING_TARGET
        )
        
        assert result.success is False
        assert result.action == SyncAction.DELETE
        assert result.error_message == "Event not found"
        assert result.validation_result == ValidationResult.MISSING_TARGET


class TestSyncBackupManager:
    """Tests for SyncBackupManager."""
    
    @pytest.fixture
    def manager(self):
        return SyncBackupManager()
    
    def test_create_backup(self, manager):
        """Test creating a backup."""
        events = [
            Mock(id="e1", time=1.0, duration=0.0, classification="kick", metadata={}),
            Mock(id="e2", time=2.0, duration=0.0, classification="kick", metadata={}),
        ]
        
        snapshot = manager.create_backup("kicks", "editor_1", events, "editor")
        
        assert snapshot.layer_id == "kicks"
        assert snapshot.block_id == "editor_1"
        assert len(snapshot.events) == 2
        assert snapshot.source == "editor"
    
    def test_get_latest_backup(self, manager):
        """Test getting latest backup."""
        events1 = [Mock(id="e1", time=1.0, duration=0.0, classification="kick", metadata={})]
        events2 = [Mock(id="e2", time=2.0, duration=0.0, classification="kick", metadata={})]
        
        manager.create_backup("kicks", "editor_1", events1, "editor")
        manager.create_backup("kicks", "editor_1", events2, "editor")
        
        latest = manager.get_latest_backup("kicks")
        
        assert latest is not None
        assert latest.events[0].id == "e2"  # Most recent
    
    def test_backup_limit(self, manager):
        """Test that old backups are trimmed."""
        for i in range(10):
            events = [Mock(id=f"e{i}", time=float(i), duration=0.0, classification="kick", metadata={})]
            manager.create_backup("kicks", "editor_1", events, "editor")
        
        history = manager.get_backup_history("kicks")
        
        assert len(history) == SyncBackupManager.MAX_BACKUPS_PER_LAYER
    
    def test_restore_from_backup(self, manager):
        """Test restoring from backup."""
        events = [Mock(id="e1", time=1.0, duration=0.0, classification="kick", metadata={})]
        snapshot = manager.create_backup("kicks", "editor_1", events, "editor")
        
        mock_api = Mock()
        result = manager.restore_from_backup(snapshot, mock_api)
        
        assert result is True
        mock_api.clear_layer_events.assert_called_once_with("kicks")
        mock_api.add_events.assert_called_once()
    
    def test_clear_backups(self, manager):
        """Test clearing backups."""
        events = [Mock(id="e1", time=1.0, duration=0.0, classification="kick", metadata={})]
        manager.create_backup("kicks", "editor_1", events, "editor")
        manager.create_backup("snares", "editor_1", events, "editor")
        
        manager.clear_backups("kicks")
        
        assert manager.get_latest_backup("kicks") is None
        assert manager.get_latest_backup("snares") is not None


class TestSyncValidator:
    """Tests for SyncValidator."""
    
    @pytest.fixture
    def validator(self):
        return SyncValidator()
    
    def test_validate_valid_event(self, validator):
        """Test validating a valid event."""
        event = {"time": 1.5, "duration": 0.1}
        
        result, msg = validator.validate_event(event)
        
        assert result == ValidationResult.VALID
        assert msg == ""
    
    def test_validate_negative_time(self, validator):
        """Test validating event with negative time."""
        event = {"time": -1.0, "duration": 0.0}
        
        result, msg = validator.validate_event(event)
        
        assert result == ValidationResult.INVALID_TIME
        assert "negative" in msg.lower()
    
    def test_validate_excessive_time(self, validator):
        """Test validating event with time beyond maximum."""
        event = {"time": 100000.0, "duration": 0.0}  # Beyond 24 hours
        
        result, msg = validator.validate_event(event)
        
        assert result == ValidationResult.INVALID_TIME
        assert "exceeds" in msg.lower()
    
    def test_validate_negative_duration(self, validator):
        """Test validating event with negative duration."""
        event = {"time": 1.0, "duration": -1.0}
        
        result, msg = validator.validate_event(event)
        
        assert result == ValidationResult.INVALID_DURATION
    
    def test_validate_events_batch(self, validator):
        """Test validating batch of events."""
        events = [
            {"time": 1.0, "duration": 0.0},  # Valid
            {"time": -1.0, "duration": 0.0},  # Invalid
            {"time": 2.0, "duration": 0.0},  # Valid
        ]
        
        is_valid, errors = validator.validate_events(events)
        
        assert is_valid is False
        assert len(errors) == 1
        assert "Event 1" in errors[0]
    
    def test_check_for_duplicates(self, validator):
        """Test duplicate detection."""
        existing = [
            Mock(time=1.0),
            Mock(time=2.0),
            Mock(time=3.0),
        ]
        new = [
            {"time": 0.5},   # New
            {"time": 1.005},  # Duplicate (within tolerance)
            {"time": 2.5},   # New
        ]
        
        duplicates = validator.check_for_duplicates(new, existing, time_tolerance=0.01)
        
        assert duplicates == [1]  # Only index 1 is duplicate


class TestSafeSyncService:
    """Tests for SafeSyncService."""
    
    @pytest.fixture
    def mock_facade(self):
        return Mock()
    
    @pytest.fixture
    def service(self, mock_facade):
        return SafeSyncService(mock_facade, "show_manager_1")
    
    @pytest.fixture
    def mock_editor_api(self):
        api = Mock()
        api.get_events_in_layer.return_value = []
        api.add_events.return_value = 5
        return api
    
    def test_sync_to_editor_valid_events(self, service, mock_editor_api):
        """Test syncing valid events to Editor."""
        events = [
            {"time": 1.0, "duration": 0.0, "classification": "kick"},
            {"time": 2.0, "duration": 0.0, "classification": "kick"},
        ]
        
        result = service.sync_to_editor(
            ma3_events=events,
            layer_id="kicks",
            block_id="editor_1",
            editor_api=mock_editor_api,
            clear_existing=False
        )
        
        assert result.success is True
        assert result.layer_id == "kicks"
        mock_editor_api.add_events.assert_called_once()
    
    def test_sync_to_editor_with_invalid_events(self, service, mock_editor_api):
        """Test syncing with some invalid events filters them out."""
        events = [
            {"time": 1.0, "duration": 0.0},   # Valid
            {"time": -1.0, "duration": 0.0},  # Invalid
            {"time": 2.0, "duration": 0.0},   # Valid
        ]
        
        result = service.sync_to_editor(
            ma3_events=events,
            layer_id="kicks",
            block_id="editor_1",
            editor_api=mock_editor_api
        )
        
        assert result.success is True
        # Only 2 valid events should be added
        call_args = mock_editor_api.add_events.call_args
        assert len(call_args[0][0]) == 2
    
    def test_sync_to_editor_all_invalid_fails(self, service, mock_editor_api):
        """Test that sync fails if all events are invalid."""
        events = [
            {"time": -1.0, "duration": 0.0},  # Invalid
            {"time": -2.0, "duration": 0.0},  # Invalid
        ]
        
        result = service.sync_to_editor(
            ma3_events=events,
            layer_id="kicks",
            block_id="editor_1",
            editor_api=mock_editor_api
        )
        
        assert result.success is False
        assert result.validation_result == ValidationResult.INVALID_TIME
    
    def test_sync_to_editor_creates_backup(self, service, mock_editor_api):
        """Test that sync creates backup of existing events."""
        existing_events = [
            Mock(id="e1", time=0.5, duration=0.0, classification="kick", metadata={}),
        ]
        mock_editor_api.get_events_in_layer.return_value = existing_events
        
        events = [{"time": 1.0, "duration": 0.0}]
        
        service.sync_to_editor(
            ma3_events=events,
            layer_id="kicks",
            block_id="editor_1",
            editor_api=mock_editor_api
        )
        
        # Verify backup was created
        backup = service._backup_manager.get_latest_backup("kicks")
        assert backup is not None
        assert len(backup.events) == 1
    
    def test_rollback(self, service, mock_editor_api):
        """Test rollback restores from backup."""
        # Create initial state
        existing_events = [
            Mock(id="e1", time=0.5, duration=0.0, classification="kick", metadata={}),
        ]
        mock_editor_api.get_events_in_layer.return_value = existing_events
        
        # Sync (creates backup)
        service.sync_to_editor(
            ma3_events=[{"time": 1.0, "duration": 0.0}],
            layer_id="kicks",
            block_id="editor_1",
            editor_api=mock_editor_api,
            clear_existing=True
        )
        
        # Reset mock for rollback
        mock_editor_api.reset_mock()
        
        # Rollback
        result = service.rollback("kicks", mock_editor_api)
        
        assert result is True
        mock_editor_api.clear_layer_events.assert_called_once_with("kicks")
        mock_editor_api.add_events.assert_called_once()
    
    def test_sync_skips_duplicates(self, service, mock_editor_api):
        """Test that sync skips duplicate events by default."""
        existing_events = [
            Mock(time=1.0),
            Mock(time=2.0),
        ]
        mock_editor_api.get_events_in_layer.return_value = existing_events
        
        events = [
            {"time": 1.005, "duration": 0.0},  # Duplicate
            {"time": 3.0, "duration": 0.0},    # New
        ]
        
        result = service.sync_to_editor(
            ma3_events=events,
            layer_id="kicks",
            block_id="editor_1",
            editor_api=mock_editor_api,
            skip_duplicates=True
        )
        
        assert result.success is True
        # Only 1 event should be added (the non-duplicate)
        call_args = mock_editor_api.add_events.call_args
        assert len(call_args[0][0]) == 1
        assert call_args[0][0][0]["time"] == 3.0
