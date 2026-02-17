"""
Integration Tests for ShowManager Layer Mappings Save/Load

Tests that layer mappings are properly saved to and loaded from block metadata.
"""

import pytest
import json
from typing import Dict

from src.application.settings.show_manager_settings import ShowManagerSettings, ShowManagerSettingsManager
from src.features.blocks.domain import Block


class MockCommandBus:
    """Mock command bus for testing."""
    def __init__(self):
        self.executed_commands = []
    
    def execute(self, command):
        """Execute command and store it."""
        # Execute the command's redo() method
        if hasattr(command, 'redo'):
            command.redo()
        self.executed_commands.append(command)
        return True


def create_mock_facade_with_command_bus():
    """Create a mock facade with command bus support."""
    class MockResult:
        def __init__(self, success=True, data=None):
            self.success = success
            self.data = data
    
    class MockBlock:
        def __init__(self, block_id):
            self.id = block_id
            self.type = "ShowManager"
            self.name = "TestShowManager"
            self.project_id = "test_project"
            self.metadata = {}
    
    class MockFacade:
        def __init__(self):
            self._blocks = {}
            self.command_bus = MockCommandBus()
            self.current_project_id = "test_project"
        
        def describe_block(self, block_id):
            if block_id not in self._blocks:
                self._blocks[block_id] = MockBlock(block_id)
            return MockResult(success=True, data=self._blocks[block_id])
        
        @property
        def block_service(self):
            """Return mock block service."""
            class MockBlockService:
                def __init__(self, facade):
                    self._facade = facade
                
                def update_block(self, project_id, block_id, block):
                    """Update block in mock storage."""
                    self._facade._blocks[block_id] = block
            
            return MockBlockService(self)
    
    return MockFacade()


class TestShowManagerSettingsSerialization:
    """Test ShowManagerSettings serialization/deserialization."""
    
    def test_layer_mappings_to_dict(self):
        """Test that layer_mappings are properly serialized."""
        settings = ShowManagerSettings()
        settings.layer_mappings = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
            "tc101_tg1_tr3": "layer_hats",
        }
        
        settings_dict = settings.to_dict()
        
        assert "layer_mappings" in settings_dict
        assert isinstance(settings_dict["layer_mappings"], dict)
        assert settings_dict["layer_mappings"] == {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
            "tc101_tg1_tr3": "layer_hats",
        }
    
    def test_layer_mappings_from_dict(self):
        """Test that layer_mappings are properly deserialized."""
        settings_dict = {
            "ma3_ip": "127.0.0.1",
            "ma3_port": 9001,
            "layer_mappings": {
                "tc101_tg1_tr1": "layer_kicks",
                "tc101_tg1_tr2": "layer_snares",
            },
        }
        
        settings = ShowManagerSettings.from_dict(settings_dict)
        
        assert isinstance(settings.layer_mappings, dict)
        assert settings.layer_mappings == {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
    
    def test_layer_mappings_empty_dict(self):
        """Test that empty layer_mappings are handled correctly."""
        settings_dict = {
            "ma3_ip": "127.0.0.1",
            "layer_mappings": {},
        }
        
        settings = ShowManagerSettings.from_dict(settings_dict)
        
        assert isinstance(settings.layer_mappings, dict)
        assert settings.layer_mappings == {}
    
    def test_layer_mappings_missing_key(self):
        """Test that missing layer_mappings key uses default."""
        settings_dict = {
            "ma3_ip": "127.0.0.1",
            # layer_mappings not present
        }
        
        settings = ShowManagerSettings.from_dict(settings_dict)
        
        assert isinstance(settings.layer_mappings, dict)
        assert settings.layer_mappings == {}  # Should use default_factory=dict
    
    def test_layer_mappings_json_serializable(self):
        """Test that layer_mappings can be JSON serialized."""
        settings = ShowManagerSettings()
        settings.layer_mappings = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
        
        settings_dict = settings.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(settings_dict)
        assert json_str is not None
        
        # Should be JSON deserializable
        loaded_dict = json.loads(json_str)
        assert loaded_dict["layer_mappings"] == {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }

    def test_listen_address_serialization(self):
        """Test that listen_address is serialized and restored."""
        settings = ShowManagerSettings()
        settings.listen_address = "0.0.0.0"
        settings_dict = settings.to_dict()
        assert settings_dict["listen_address"] == "0.0.0.0"
        restored = ShowManagerSettings.from_dict(settings_dict)
        assert restored.listen_address == "0.0.0.0"



class TestShowManagerSettingsManagerSaveLoad:
    """Test ShowManagerSettingsManager save/load through block metadata."""
    
    def test_layer_mappings_save_to_metadata(self):
        """Test that layer_mappings are saved to block metadata."""
        facade = create_mock_facade_with_command_bus()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Set layer mappings
        test_mappings = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
        manager.layer_mappings = test_mappings
        
        # Force save (bypass debounce)
        manager.force_save()
        
        # Verify command was executed
        assert len(facade.command_bus.executed_commands) > 0
        
        # Verify block metadata contains layer_mappings
        block_result = facade.describe_block("test_block")
        assert block_result.success
        block = block_result.data
        
        assert "layer_mappings" in block.metadata
        assert block.metadata["layer_mappings"] == test_mappings
    
    def test_layer_mappings_load_from_metadata(self):
        """Test that layer_mappings are loaded from block metadata."""
        facade = create_mock_facade_with_command_bus()
        
        # Set up block with metadata
        block_result = facade.describe_block("test_block")
        block = block_result.data
        block.metadata = {
            "ma3_ip": "127.0.0.1",
            "ma3_port": 9001,
            "layer_mappings": {
                "tc101_tg1_tr1": "layer_kicks",
                "tc101_tg1_tr2": "layer_snares",
            },
        }
        facade._blocks["test_block"] = block
        
        # Create manager (should load from metadata)
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Verify mappings were loaded
        assert manager.layer_mappings == {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
    
    def test_layer_mappings_update_persistence(self):
        """Test that updating layer_mappings persists correctly."""
        facade = create_mock_facade_with_command_bus()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Set initial mappings
        manager.set_layer_mapping("tc101_tg1_tr1", "layer_kicks")
        manager.force_save()
        
        # Update mappings
        manager.set_layer_mapping("tc101_tg1_tr2", "layer_snares")
        manager.force_save()
        
        # Verify both mappings are in metadata
        block_result = facade.describe_block("test_block")
        block = block_result.data
        
        assert "layer_mappings" in block.metadata
        mappings = block.metadata["layer_mappings"]
        assert mappings["tc101_tg1_tr1"] == "layer_kicks"
        assert mappings["tc101_tg1_tr2"] == "layer_snares"
    
    def test_layer_mappings_clear_persistence(self):
        """Test that clearing layer_mappings persists correctly."""
        facade = create_mock_facade_with_command_bus()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Set mappings
        manager.set_layer_mapping("tc101_tg1_tr1", "layer_kicks")
        manager.set_layer_mapping("tc101_tg1_tr2", "layer_snares")
        manager.force_save()
        
        # Clear mappings
        manager.clear_all_mappings()
        manager.force_save()
        
        # Verify mappings are cleared in metadata
        block_result = facade.describe_block("test_block")
        block = block_result.data
        
        assert "layer_mappings" in block.metadata
        assert block.metadata["layer_mappings"] == {}

    def test_listen_address_persistence(self):
        """Test that listen_address persists in block metadata."""
        facade = create_mock_facade_with_command_bus()
        manager = ShowManagerSettingsManager(facade, "test_block")
        manager.listen_address = "0.0.0.0"
        manager.force_save()
        block_result = facade.describe_block("test_block")
        block = block_result.data
        assert block.metadata.get("listen_address") == "0.0.0.0"



class TestBlockSerialization:
    """Test Block serialization with layer_mappings in metadata."""
    
    def test_block_to_dict_includes_layer_mappings(self):
        """Test that Block.to_dict() includes layer_mappings in metadata."""
        block = Block(
            id="test_block",
            project_id="test_project",
            name="TestShowManager",
            type="ShowManager",
        )
        
        block.metadata = {
            "ma3_ip": "127.0.0.1",
            "layer_mappings": {
                "tc101_tg1_tr1": "layer_kicks",
                "tc101_tg1_tr2": "layer_snares",
            },
        }
        
        block_dict = block.to_dict()
        
        assert "metadata" in block_dict
        assert "layer_mappings" in block_dict["metadata"]
        assert block_dict["metadata"]["layer_mappings"] == {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
    
    def test_block_from_dict_loads_layer_mappings(self):
        """Test that Block.from_dict() loads layer_mappings from metadata."""
        block_dict = {
            "id": "test_block",
            "project_id": "test_project",
            "name": "TestShowManager",
            "type": "ShowManager",
            "ports": {},
            "metadata": {
                "ma3_ip": "127.0.0.1",
                "layer_mappings": {
                    "tc101_tg1_tr1": "layer_kicks",
                    "tc101_tg1_tr2": "layer_snares",
                },
            },
        }
        
        block = Block.from_dict(block_dict)
        
        assert "layer_mappings" in block.metadata
        assert block.metadata["layer_mappings"] == {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
    
    def test_block_round_trip_layer_mappings(self):
        """Test that layer_mappings survive Block serialization round-trip."""
        original_mappings = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
            "tc101_tg1_tr3": "layer_hats",
        }
        
        block = Block(
            id="test_block",
            project_id="test_project",
            name="TestShowManager",
            type="ShowManager",
        )
        block.metadata = {
            "ma3_ip": "127.0.0.1",
            "layer_mappings": original_mappings,
        }
        
        # Serialize
        block_dict = block.to_dict()
        
        # Deserialize
        loaded_block = Block.from_dict(block_dict)
        
        # Verify mappings are preserved
        assert loaded_block.metadata["layer_mappings"] == original_mappings
    
    def test_block_json_round_trip_layer_mappings(self):
        """Test that layer_mappings survive JSON serialization round-trip."""
        original_mappings = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
        
        block = Block(
            id="test_block",
            project_id="test_project",
            name="TestShowManager",
            type="ShowManager",
        )
        block.metadata = {
            "layer_mappings": original_mappings,
        }
        
        # Serialize to dict
        block_dict = block.to_dict()
        
        # JSON serialize/deserialize
        json_str = json.dumps(block_dict)
        loaded_dict = json.loads(json_str)
        
        # Deserialize block
        loaded_block = Block.from_dict(loaded_dict)
        
        # Verify mappings are preserved
        assert loaded_block.metadata["layer_mappings"] == original_mappings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
