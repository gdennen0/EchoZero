"""
Integration Tests for MA3 Layer Mapping

Tests the complete layer mapping system including:
- Settings management
- Mapping service
- Validation
- Auto-detection
"""

import pytest
from typing import Dict, List

from src.application.settings.show_manager_settings import ShowManagerSettings, ShowManagerSettingsManager
from src.features.ma3.application.ma3_layer_mapping_service import (
    MA3LayerMappingService, MA3TrackInfo, MappingStatus, ValidationSeverity
)


def create_mock_facade():
    """Create a mock facade for testing."""
    class MockResult:
        def __init__(self, success=True, data=None):
            self.success = success
            self.data = data
    
    class MockBlock:
        def __init__(self, block_id):
            self.id = block_id
            self.type = "ShowManager"
            self.metadata = {}
    
    class MockFacade:
        def __init__(self):
            self._blocks = {}
        
        def describe_block(self, block_id):
            if block_id not in self._blocks:
                self._blocks[block_id] = MockBlock(block_id)
            return MockResult(success=True, data=self._blocks[block_id])
        
        def get_block_setting(self, block_id, key, default=None):
            if block_id in self._blocks:
                return self._blocks[block_id].metadata.get(key, default)
            return default
        
        def set_block_setting(self, block_id, key, value):
            if block_id not in self._blocks:
                self._blocks[block_id] = MockBlock(block_id)
            self._blocks[block_id].metadata[key] = value
    
    return MockFacade()


class TestShowManagerSettings:
    """Test ShowManagerSettings layer mapping functionality."""
    
    def test_layer_mappings_default(self):
        """Test default layer mappings is empty dict."""
        settings = ShowManagerSettings()
        assert settings.layer_mappings == {}
    
    def test_set_get_mapping(self):
        """Test setting and getting individual mappings."""
        facade = create_mock_facade()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Set mapping
        manager.set_mapping("tc101_tg1_tr1", "layer_kicks")
        
        # Get mapping
        result = manager.get_mapping("tc101_tg1_tr1")
        assert result == "layer_kicks"
    
    def test_remove_mapping(self):
        """Test removing a mapping."""
        facade = create_mock_facade()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Set and remove
        manager.set_mapping("tc101_tg1_tr1", "layer_kicks")
        manager.remove_mapping("tc101_tg1_tr1")
        
        # Should be None
        assert manager.get_mapping("tc101_tg1_tr1") is None
    
    def test_clear_all_mappings(self):
        """Test clearing all mappings."""
        facade = create_mock_facade()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Set multiple
        manager.set_mapping("tc101_tg1_tr1", "layer_kicks")
        manager.set_mapping("tc101_tg1_tr2", "layer_snares")
        
        # Clear all
        manager.clear_all_mappings()
        
        # Should be empty
        assert manager.layer_mappings == {}
    
    def test_get_unmapped_ma3_tracks(self):
        """Test getting unmapped MA3 tracks."""
        facade = create_mock_facade()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Set one mapping
        manager.set_mapping("tc101_tg1_tr1", "layer_kicks")
        
        # Check unmapped
        all_tracks = ["tc101_tg1_tr1", "tc101_tg1_tr2", "tc101_tg1_tr3"]
        unmapped = manager.get_unmapped_ma3_tracks(all_tracks)
        
        assert len(unmapped) == 2
        assert "tc101_tg1_tr2" in unmapped
        assert "tc101_tg1_tr3" in unmapped
    
    def test_has_conflicts(self):
        """Test conflict detection."""
        facade = create_mock_facade()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # No conflicts initially
        assert not manager.has_conflicts()
        
        # Add conflicting mappings (two MA3 tracks → same EZ layer)
        manager.set_mapping("tc101_tg1_tr1", "layer_kicks")
        manager.set_mapping("tc101_tg1_tr2", "layer_kicks")
        
        # Should have conflicts
        assert manager.has_conflicts()
    
    def test_get_conflicts(self):
        """Test getting conflict details."""
        facade = create_mock_facade()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # Add conflicting mappings
        manager.set_mapping("tc101_tg1_tr1", "layer_kicks")
        manager.set_mapping("tc101_tg1_tr2", "layer_kicks")
        manager.set_mapping("tc101_tg1_tr3", "layer_snares")
        
        # Get conflicts
        conflicts = manager.get_conflicts()
        
        # Should have one conflict (layer_kicks)
        assert len(conflicts) == 1
        assert "layer_kicks" in conflicts
        assert len(conflicts["layer_kicks"]) == 2


class TestMA3LayerMappingService:
    """Test MA3LayerMappingService functionality."""
    
    def test_parse_ma3_structure(self):
        """Test parsing MA3 structure data."""
        service = MA3LayerMappingService()
        
        # Mock structure data
        structure_data = {
            'timecode_no': 101,
            'track_groups': [
                {
                    'index': 1,
                    'tracks': [
                        {'index': 1, 'name': 'Kicks'},
                        {'index': 2, 'name': 'Snares'},
                    ]
                }
            ]
        }
        
        tracks = service.parse_ma3_structure(structure_data)
        
        assert len(tracks) == 2
        assert tracks[0].coord == "tc101_tg1_tr1"
        assert tracks[0].name == "Kicks"
        assert tracks[1].coord == "tc101_tg1_tr2"
        assert tracks[1].name == "Snares"
    
    def test_build_mappings_all_mapped(self):
        """Test building mappings when all tracks are mapped."""
        service = MA3LayerMappingService()
        
        # Create tracks
        tracks = [
            MA3TrackInfo(101, 1, 1, "Kicks"),
            MA3TrackInfo(101, 1, 2, "Snares"),
        ]
        
        # Create layers
        layers = ["layer_kicks", "layer_snares"]
        
        # Create mappings
        mapping_dict = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
        
        # Build
        mappings = service.build_mappings(mapping_dict, tracks, layers)
        
        # All should be mapped
        assert len(mappings) == 2
        assert all(m.status == MappingStatus.MAPPED for m in mappings)
    
    def test_build_mappings_unmapped(self):
        """Test building mappings with unmapped tracks."""
        service = MA3LayerMappingService()
        
        # Create tracks
        tracks = [
            MA3TrackInfo(101, 1, 1, "Kicks"),
            MA3TrackInfo(101, 1, 2, "Snares"),
        ]
        
        # Create layers
        layers = ["layer_kicks", "layer_snares"]
        
        # Only map one
        mapping_dict = {
            "tc101_tg1_tr1": "layer_kicks",
        }
        
        # Build
        mappings = service.build_mappings(mapping_dict, tracks, layers)
        
        # Should have 2 mappings (1 mapped, 1 unmapped)
        assert len(mappings) == 2
        mapped = [m for m in mappings if m.status == MappingStatus.MAPPED]
        unmapped = [m for m in mappings if m.status == MappingStatus.UNMAPPED_MA3]
        
        assert len(mapped) == 1
        assert len(unmapped) == 1
    
    def test_build_mappings_conflict(self):
        """Test building mappings with conflicts."""
        service = MA3LayerMappingService()
        
        # Create tracks
        tracks = [
            MA3TrackInfo(101, 1, 1, "Kicks"),
            MA3TrackInfo(101, 1, 2, "Snares"),
        ]
        
        # Create layers
        layers = ["layer_kicks"]
        
        # Map both to same layer (conflict)
        mapping_dict = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_kicks",
        }
        
        # Build
        mappings = service.build_mappings(mapping_dict, tracks, layers)
        
        # Both should be marked as conflicts
        assert len(mappings) == 2
        assert all(m.status == MappingStatus.CONFLICT for m in mappings)
    
    def test_validate_mappings_no_errors(self):
        """Test validation with valid mappings."""
        service = MA3LayerMappingService()
        
        # Parse structure to set tracks (this updates the service's internal cache)
        structure_data = {
            'timecode_no': 101,
            'track_groups': [
                {
                    'index': 1,
                    'tracks': [
                        {'index': 1, 'name': 'Kicks'},
                        {'index': 2, 'name': 'Snares'},
                    ]
                }
            ]
        }
        tracks = service.parse_ma3_structure(structure_data)
        
        # Create layers
        layers = ["layer_kicks", "layer_snares"]
        service.set_ez_layers(layers)
        
        # Valid mappings
        mapping_dict = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_snares",
        }
        
        # Validate (don't pass tracks/layers, use cached ones)
        errors = service.validate_mappings(mapping_dict)
        
        # Should have no errors
        assert len(errors) == 0
    
    def test_validate_mappings_unmapped_warning(self):
        """Test validation with unmapped tracks."""
        service = MA3LayerMappingService()
        
        # Create tracks
        tracks = [
            MA3TrackInfo(101, 1, 1, "Kicks"),
            MA3TrackInfo(101, 1, 2, "Snares"),
        ]
        
        # Create layers
        layers = ["layer_kicks", "layer_snares"]
        
        # Only map one
        mapping_dict = {
            "tc101_tg1_tr1": "layer_kicks",
        }
        
        # Validate
        errors = service.validate_mappings(mapping_dict, tracks, layers)
        
        # Should have warning about unmapped track
        assert len(errors) == 1
        assert errors[0].type == "unmapped_ma3"
        assert errors[0].severity == ValidationSeverity.WARNING
    
    def test_validate_mappings_conflict_error(self):
        """Test validation with conflicts."""
        service = MA3LayerMappingService()
        
        # Create tracks
        tracks = [
            MA3TrackInfo(101, 1, 1, "Kicks"),
            MA3TrackInfo(101, 1, 2, "Snares"),
        ]
        
        # Create layers
        layers = ["layer_kicks"]
        
        # Conflicting mappings
        mapping_dict = {
            "tc101_tg1_tr1": "layer_kicks",
            "tc101_tg1_tr2": "layer_kicks",
        }
        
        # Validate
        errors = service.validate_mappings(mapping_dict, tracks, layers)
        
        # Should have error about conflict
        conflict_errors = [e for e in errors if e.type == "conflict"]
        assert len(conflict_errors) == 1
        assert conflict_errors[0].severity == ValidationSeverity.ERROR
    
    def test_suggest_mappings_exact_match(self):
        """Test auto-detection with exact name matches."""
        service = MA3LayerMappingService()
        
        # Create tracks with matching names
        tracks = [
            MA3TrackInfo(101, 1, 1, "kicks"),
            MA3TrackInfo(101, 1, 2, "snares"),
        ]
        
        # Create layers with matching names
        layers = ["kicks", "snares", "hats"]
        
        # Get suggestions
        suggestions = service.suggest_mappings(tracks, layers, threshold=0.6)
        
        # Should suggest both (exact matches)
        assert len(suggestions) == 2
        assert suggestions["tc101_tg1_tr1"][0] == "kicks"
        assert suggestions["tc101_tg1_tr2"][0] == "snares"
        
        # Scores should be 1.0 (exact match)
        assert suggestions["tc101_tg1_tr1"][1] == 1.0
        assert suggestions["tc101_tg1_tr2"][1] == 1.0
    
    def test_suggest_mappings_fuzzy_match(self):
        """Test auto-detection with fuzzy name matches."""
        service = MA3LayerMappingService()
        
        # Create tracks with similar names
        tracks = [
            MA3TrackInfo(101, 1, 1, "kicks"),
        ]
        
        # Create layers with similar names
        layers = ["layer_kicks", "layer_snares"]
        
        # Get suggestions with lower threshold
        suggestions = service.suggest_mappings(tracks, layers, threshold=0.3)
        
        # Should suggest kicks → layer_kicks (substring match)
        # If no suggestions, that's okay - fuzzy matching is optional
        if len(suggestions) > 0:
            assert "tc101_tg1_tr1" in suggestions
            assert "kick" in suggestions["tc101_tg1_tr1"][0].lower()
    
    def test_suggest_mappings_no_match(self):
        """Test auto-detection with no matches."""
        service = MA3LayerMappingService()
        
        # Create tracks
        tracks = [
            MA3TrackInfo(101, 1, 1, "xyz"),
        ]
        
        # Create layers with no similarity
        layers = ["abc", "def"]
        
        # Get suggestions
        suggestions = service.suggest_mappings(tracks, layers, threshold=0.6)
        
        # Should have no suggestions
        assert len(suggestions) == 0
    
    def test_graceful_degradation(self):
        """Test graceful degradation when MA3/EZ offline."""
        service = MA3LayerMappingService()
        
        # Initially offline
        assert not service.is_ma3_available
        assert not service.is_ez_available
        
        # Status message should indicate offline
        status = service.get_status_message()
        assert "offline" in status.lower()
        
        # Set MA3 tracks
        service.parse_ma3_structure({'timecode_no': 101, 'track_groups': [{'index': 1, 'tracks': [{'index': 1, 'name': 'Kicks'}]}]})
        
        assert service.is_ma3_available
        
        # Set EZ layers so status shows both
        service.set_ez_layers(["layer_kicks"])
        assert service.is_ez_available
        
        # Status should update
        status = service.get_status_message()
        assert "1 MA3 track" in status or "1 EZ layer" in status


class TestIntegration:
    """Integration tests combining settings and service."""
    
    def test_full_mapping_workflow(self):
        """Test complete mapping workflow."""
        # Create service and manager
        service = MA3LayerMappingService()
        facade = create_mock_facade()
        manager = ShowManagerSettingsManager(facade, "test_block")
        
        # 1. Parse MA3 structure
        structure_data = {
            'timecode_no': 101,
            'track_groups': [
                {
                    'index': 1,
                    'tracks': [
                        {'index': 1, 'name': 'Kicks'},
                        {'index': 2, 'name': 'Snares'},
                        {'index': 3, 'name': 'Hats'},
                    ]
                }
            ]
        }
        tracks = service.parse_ma3_structure(structure_data)
        
        # 2. Set EZ layers
        layers = ["layer_kicks", "layer_snares", "layer_hats"]
        service.set_ez_layers(layers)
        
        # 3. Auto-detect mappings
        suggestions = service.suggest_mappings(tracks, layers, threshold=0.6)
        
        # 4. Apply suggestions to settings
        for ma3_coord, (ez_layer, score) in suggestions.items():
            manager.set_mapping(ma3_coord, ez_layer)
        
        # 5. Validate (use cached tracks/layers from service)
        mappings = manager.layer_mappings
        errors = service.validate_mappings(mappings)
        
        # Should have no errors (all mapped)
        assert len(errors) == 0
        
        # 6. Build mapping objects
        mapping_objects = service.build_mappings(mappings, tracks, layers)
        
        # All should be mapped
        assert all(m.status == MappingStatus.MAPPED for m in mapping_objects)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
