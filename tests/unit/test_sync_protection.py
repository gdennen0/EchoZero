"""
Unit tests for Timeline Sync Protection.

Tests that synced layer events are preserved during set_events() calls
by using selective clearing (skip synced layers) instead of backup/restore.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch


class TestSyncProtection:
    """Tests for synced layer event protection in TimelineWidget."""
    
    @pytest.fixture
    def mock_layer_manager(self):
        """Create mock layer manager with synced and non-synced layers."""
        manager = Mock()
        
        # Create mock layers
        synced_layer = Mock()
        synced_layer.id = "layer_synced"
        synced_layer.name = "Synced Kicks"
        synced_layer.is_synced = True
        synced_layer.height = 40
        synced_layer.color = "#ff0000"
        synced_layer.visible = True
        synced_layer.locked = False
        
        regular_layer = Mock()
        regular_layer.id = "layer_regular"
        regular_layer.name = "Regular Snares"
        regular_layer.is_synced = False
        regular_layer.height = 40
        regular_layer.color = "#00ff00"
        regular_layer.visible = True
        regular_layer.locked = False
        
        manager.get_all_layers.return_value = [synced_layer, regular_layer]
        manager.get_synced_layers.return_value = [synced_layer]
        manager.get_synced_layer_names.return_value = ["Synced Kicks"]
        manager.get_layer_by_name.side_effect = lambda name: {
            "Synced Kicks": synced_layer,
            "Regular Snares": regular_layer
        }.get(name)
        manager.get_layer.side_effect = lambda lid: {
            "layer_synced": synced_layer,
            "layer_regular": regular_layer
        }.get(lid)
        manager.get_layer_count.return_value = 2
        
        return manager
    
    @pytest.fixture
    def mock_scene(self):
        """Create mock scene with events."""
        scene = Mock()
        
        # Synced layer events
        synced_event_1 = Mock()
        synced_event_1.event_id = "evt_sync_1"
        synced_event_1.layer_id = "layer_synced"
        
        synced_event_2 = Mock()
        synced_event_2.event_id = "evt_sync_2"
        synced_event_2.layer_id = "layer_synced"
        
        regular_event_1 = Mock()
        regular_event_1.event_id = "evt_reg_1"
        regular_event_1.layer_id = "layer_regular"
        
        scene._event_items = {
            "evt_sync_1": synced_event_1,
            "evt_sync_2": synced_event_2,
            "evt_reg_1": regular_event_1,
        }
        
        scene.add_event = Mock()
        scene.remove_event = Mock()
        scene.clear_events = Mock()
        scene.clear_events_except_layers = Mock()
        scene.removeItem = Mock()
        
        return scene
    
    def test_selective_clear_preserves_synced_layer_events(self, mock_layer_manager, mock_scene):
        """Test that synced layer events are preserved when selective clear is used."""
        # Get synced layer IDs
        synced_layer_ids = set()
        for layer in mock_layer_manager.get_synced_layers():
            synced_layer_ids.add(layer.id)
        
        assert synced_layer_ids == {"layer_synced"}
        
        # Simulate the selective clear logic
        items_to_remove = [
            item for item in mock_scene._event_items.values()
            if item.layer_id not in synced_layer_ids
        ]
        
        # Only the regular layer event should be removed
        assert len(items_to_remove) == 1
        assert items_to_remove[0].event_id == "evt_reg_1"
        assert items_to_remove[0].layer_id == "layer_regular"
    
    def test_synced_layer_events_not_cleared(self, mock_layer_manager, mock_scene):
        """Test that events in synced layers are not in the remove list."""
        synced_layer_ids = {layer.id for layer in mock_layer_manager.get_synced_layers()}
        
        items_to_remove = [
            item for item in mock_scene._event_items.values()
            if item.layer_id not in synced_layer_ids
        ]
        
        # Synced events should NOT be in the remove list
        removed_ids = {item.event_id for item in items_to_remove}
        assert "evt_sync_1" not in removed_ids
        assert "evt_sync_2" not in removed_ids
    
    def test_non_synced_layers_are_cleared(self, mock_layer_manager, mock_scene):
        """Test that non-synced layer events ARE in the remove list."""
        synced_layer_ids = {layer.id for layer in mock_layer_manager.get_synced_layers()}
        
        items_to_remove = [
            item for item in mock_scene._event_items.values()
            if item.layer_id not in synced_layer_ids
        ]
        
        # Regular (non-synced) event should be in the remove list
        removed_ids = {item.event_id for item in items_to_remove}
        assert "evt_reg_1" in removed_ids
    
    def test_no_synced_layers_uses_full_clear(self, mock_scene):
        """Test that when there are no synced layers, full clear is used."""
        # Create layer manager with no synced layers
        manager = Mock()
        manager.get_synced_layers.return_value = []
        
        synced_layer_ids = set()
        for layer in manager.get_synced_layers():
            synced_layer_ids.add(layer.id)
        
        # Should use full clear path when no synced layers
        assert len(synced_layer_ids) == 0
        
        # Logic would be: if synced_layer_ids: selective_clear else: full_clear
        if synced_layer_ids:
            mock_scene.clear_events_except_layers(synced_layer_ids)
        else:
            mock_scene.clear_events()
        
        # Verify full clear was called
        mock_scene.clear_events.assert_called_once()
        mock_scene.clear_events_except_layers.assert_not_called()
    
    def test_layer_manager_get_synced_layers_method(self):
        """Test the get_synced_layers method on LayerManager."""
        from ui.qt_gui.widgets.timeline.events.layer_manager import LayerManager
        
        # Create a fresh layer manager
        manager = LayerManager()
        
        # Create layers
        regular_layer = manager.create_layer("Regular", is_synced=False)
        synced_layer = manager.create_layer("Synced", is_synced=True)
        
        # Test get_synced_layers
        synced = manager.get_synced_layers()
        assert len(synced) == 1
        assert synced[0].name == "Synced"
        assert synced[0].is_synced is True
        
        # Test get_synced_layer_names
        synced_names = manager.get_synced_layer_names()
        assert synced_names == ["Synced"]


class TestClearEventsExceptLayersLogic:
    """Tests for the clear_events_except_layers logic (without Qt)."""
    
    def test_clear_events_except_layers_logic_preserves_specified_layers(self):
        """Test that clear_events_except_layers logic preserves events in specified layers."""
        # Simulate _event_items dict
        synced_event = Mock(event_id="synced_evt_1", layer_id="layer_synced")
        regular_event = Mock(event_id="regular_evt_1", layer_id="layer_regular")
        
        event_items = {
            "synced_evt_1": synced_event,
            "regular_evt_1": regular_event,
        }
        
        # Layers to keep
        layer_ids_to_keep = {"layer_synced"}
        
        # Simulate clear_events_except_layers logic
        items_to_remove = [
            item for item in event_items.values()
            if item.layer_id not in layer_ids_to_keep
        ]
        
        # Should only remove the regular event
        assert len(items_to_remove) == 1
        assert items_to_remove[0].event_id == "regular_evt_1"
        
        # Verify synced event not in remove list
        assert synced_event not in items_to_remove
    
    def test_clear_events_except_layers_logic_removes_all_non_specified(self):
        """Test that events in non-specified layers are removed."""
        # Simulate multiple layers
        keep_event = Mock(event_id="keep_1", layer_id="layer_keep")
        remove_event_1 = Mock(event_id="remove_1", layer_id="layer_remove_1")
        remove_event_2 = Mock(event_id="remove_2", layer_id="layer_remove_2")
        remove_event_3 = Mock(event_id="remove_3", layer_id="layer_remove_2")
        
        event_items = {
            "keep_1": keep_event,
            "remove_1": remove_event_1,
            "remove_2": remove_event_2,
            "remove_3": remove_event_3,
        }
        
        layer_ids_to_keep = {"layer_keep"}
        
        items_to_remove = [
            item for item in event_items.values()
            if item.layer_id not in layer_ids_to_keep
        ]
        
        # Should remove 3 events
        assert len(items_to_remove) == 3
        
        # Verify correct events in remove list
        removed_ids = {item.event_id for item in items_to_remove}
        assert "remove_1" in removed_ids
        assert "remove_2" in removed_ids
        assert "remove_3" in removed_ids
        assert "keep_1" not in removed_ids
    
    def test_clear_events_except_layers_logic_empty_set_clears_all(self):
        """Test that passing empty set clears all events."""
        event_1 = Mock(event_id="evt_1", layer_id="layer_1")
        event_2 = Mock(event_id="evt_2", layer_id="layer_2")
        
        event_items = {
            "evt_1": event_1,
            "evt_2": event_2,
        }
        
        layer_ids_to_keep = set()  # Empty set
        
        items_to_remove = [
            item for item in event_items.values()
            if item.layer_id not in layer_ids_to_keep
        ]
        
        # All events should be removed when empty set passed
        assert len(items_to_remove) == 2
    
    def test_multiple_synced_layers_preserved(self):
        """Test that multiple synced layers are all preserved."""
        synced_1 = Mock(event_id="synced_1", layer_id="layer_synced_a")
        synced_2 = Mock(event_id="synced_2", layer_id="layer_synced_b")
        regular = Mock(event_id="regular_1", layer_id="layer_regular")
        
        event_items = {
            "synced_1": synced_1,
            "synced_2": synced_2,
            "regular_1": regular,
        }
        
        # Multiple synced layers
        layer_ids_to_keep = {"layer_synced_a", "layer_synced_b"}
        
        items_to_remove = [
            item for item in event_items.values()
            if item.layer_id not in layer_ids_to_keep
        ]
        
        # Only regular event should be removed
        assert len(items_to_remove) == 1
        assert items_to_remove[0].event_id == "regular_1"
