"""
Tests for the ModuleRegistry pattern.

Tests registration, lookup, decorators, and global registries.
"""
import pytest
from typing import List

from src.shared.application.registry.module_registry import (
    ModuleRegistry,
    ComponentMetadata,
    register_component,
    get_registry,
    list_registries,
    clear_all_registries,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class TestComponent:
    """Base test component."""
    pass


class ComponentA(TestComponent):
    """Test component A."""
    pass


class ComponentB(TestComponent):
    """Test component B."""
    pass


class ComponentC(TestComponent):
    """Test component C."""
    pass


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear all registries before and after each test."""
    clear_all_registries()
    yield
    clear_all_registries()


# =============================================================================
# ComponentMetadata Tests
# =============================================================================

class TestComponentMetadata:
    """Tests for ComponentMetadata dataclass."""
    
    def test_default_values(self):
        """Test metadata has sensible defaults."""
        meta = ComponentMetadata(key="test", component_class=ComponentA)
        assert meta.key == "test"
        assert meta.component_class is ComponentA
        assert meta.registry_name == ""
        assert meta.description == ""
        assert meta.version == 1
        assert meta.tags == []
        assert meta.extra == {}
    
    def test_with_all_values(self):
        """Test metadata with all values set."""
        meta = ComponentMetadata(
            key="test",
            component_class=ComponentA,
            registry_name="test_registry",
            description="Test description",
            version=2,
            tags=["audio", "processing"],
            extra={"custom": "value"},
        )
        assert meta.description == "Test description"
        assert meta.version == 2
        assert "audio" in meta.tags
        assert meta.extra["custom"] == "value"
    
    def test_none_tags_becomes_empty_list(self):
        """Test that None tags becomes empty list."""
        meta = ComponentMetadata(
            key="test",
            component_class=ComponentA,
            tags=None,
        )
        assert meta.tags == []


# =============================================================================
# ModuleRegistry Tests
# =============================================================================

class TestModuleRegistry:
    """Tests for ModuleRegistry class."""
    
    def test_create_registry(self):
        """Test creating a registry."""
        registry = ModuleRegistry("TestRegistry")
        assert registry.name == "TestRegistry"
        assert registry.count() == 0
    
    def test_register_class(self):
        """Test registering a class directly."""
        registry = ModuleRegistry("Test")
        registry.register_class("comp_a", ComponentA)
        
        assert registry.is_registered("comp_a")
        assert registry.get("comp_a") is ComponentA
    
    def test_register_class_with_metadata(self):
        """Test registering with full metadata."""
        registry = ModuleRegistry("Test")
        registry.register_class(
            "comp_a",
            ComponentA,
            description="Component A",
            version=2,
            tags=["test"],
            extra={"custom": 123},
        )
        
        meta = registry.get_metadata("comp_a")
        assert meta is not None
        assert meta.description == "Component A"
        assert meta.version == 2
        assert "test" in meta.tags
        assert meta.extra["custom"] == 123
    
    def test_register_decorator(self):
        """Test registering via decorator."""
        registry = ModuleRegistry("Test")
        
        @registry.register("decorated", description="Decorated component")
        class DecoratedComponent(TestComponent):
            pass
        
        assert registry.is_registered("decorated")
        assert registry.get("decorated") is DecoratedComponent
        
        meta = registry.get_metadata("decorated")
        assert meta.description == "Decorated component"
    
    def test_get_nonexistent(self):
        """Test getting non-existent key returns None."""
        registry = ModuleRegistry("Test")
        assert registry.get("nonexistent") is None
    
    def test_get_metadata_nonexistent(self):
        """Test getting metadata for non-existent key returns None."""
        registry = ModuleRegistry("Test")
        assert registry.get_metadata("nonexistent") is None
    
    def test_duplicate_registration_same_class(self):
        """Test re-registering same class is allowed."""
        registry = ModuleRegistry("Test")
        registry.register_class("comp_a", ComponentA)
        # Should not raise
        registry.register_class("comp_a", ComponentA)
        
        assert registry.get("comp_a") is ComponentA
    
    def test_duplicate_registration_different_class(self):
        """Test registering different class with same key raises."""
        registry = ModuleRegistry("Test")
        registry.register_class("comp", ComponentA)
        
        with pytest.raises(ValueError) as exc:
            registry.register_class("comp", ComponentB)
        
        assert "already registered" in str(exc.value)
    
    def test_list_all(self):
        """Test listing all registered components."""
        registry = ModuleRegistry("Test")
        registry.register_class("a", ComponentA)
        registry.register_class("b", ComponentB)
        registry.register_class("c", ComponentC)
        
        results = registry.list_all()
        assert len(results) == 3
        
        keys = [r[0] for r in results]
        assert "a" in keys
        assert "b" in keys
        assert "c" in keys
    
    def test_list_keys(self):
        """Test listing all keys."""
        registry = ModuleRegistry("Test")
        registry.register_class("a", ComponentA)
        registry.register_class("b", ComponentB)
        
        keys = registry.list_keys()
        assert "a" in keys
        assert "b" in keys
    
    def test_is_registered(self):
        """Test checking if key is registered."""
        registry = ModuleRegistry("Test")
        assert registry.is_registered("a") is False
        
        registry.register_class("a", ComponentA)
        assert registry.is_registered("a") is True
    
    def test_unregister(self):
        """Test unregistering a component."""
        registry = ModuleRegistry("Test")
        registry.register_class("a", ComponentA)
        assert registry.is_registered("a")
        
        result = registry.unregister("a")
        assert result is True
        assert registry.is_registered("a") is False
    
    def test_unregister_nonexistent(self):
        """Test unregistering non-existent key returns False."""
        registry = ModuleRegistry("Test")
        result = registry.unregister("nonexistent")
        assert result is False
    
    def test_clear(self):
        """Test clearing all registrations."""
        registry = ModuleRegistry("Test")
        registry.register_class("a", ComponentA)
        registry.register_class("b", ComponentB)
        
        registry.clear()
        
        assert registry.count() == 0
        assert registry.is_registered("a") is False
    
    def test_count(self):
        """Test counting registered components."""
        registry = ModuleRegistry("Test")
        assert registry.count() == 0
        
        registry.register_class("a", ComponentA)
        assert registry.count() == 1
        
        registry.register_class("b", ComponentB)
        assert registry.count() == 2
    
    def test_get_by_tag(self):
        """Test filtering by tag."""
        registry = ModuleRegistry("Test")
        registry.register_class("a", ComponentA, tags=["audio", "input"])
        registry.register_class("b", ComponentB, tags=["audio", "output"])
        registry.register_class("c", ComponentC, tags=["video"])
        
        audio_components = registry.get_by_tag("audio")
        assert len(audio_components) == 2
        
        keys = [c[0] for c in audio_components]
        assert "a" in keys
        assert "b" in keys
        assert "c" not in keys
    
    def test_search(self):
        """Test searching components."""
        registry = ModuleRegistry("Test")
        registry.register_class("LoadAudio", ComponentA, description="Load audio files")
        registry.register_class("ExportAudio", ComponentB, description="Export audio files")
        registry.register_class("DetectOnsets", ComponentC, description="Detect onsets", tags=["detection"])
        
        # Search by key
        results = registry.search("Audio")
        assert len(results) == 2
        
        # Search by description
        results = registry.search("Export")
        assert len(results) == 1
        assert results[0][0] == "ExportAudio"
        
        # Search by tag
        results = registry.search("detection")
        assert len(results) == 1
        assert results[0][0] == "DetectOnsets"


# =============================================================================
# Global Registry Tests
# =============================================================================

class TestGlobalRegistry:
    """Tests for global registry functions."""
    
    def test_get_registry_creates(self):
        """Test get_registry creates new registry."""
        registry = get_registry("test_registry")
        assert registry is not None
        assert registry.name == "test_registry"
    
    def test_get_registry_returns_same(self):
        """Test get_registry returns same instance."""
        registry1 = get_registry("test_registry")
        registry2 = get_registry("test_registry")
        assert registry1 is registry2
    
    def test_list_registries(self):
        """Test listing all registries."""
        get_registry("registry_a")
        get_registry("registry_b")
        
        names = list_registries()
        assert "registry_a" in names
        assert "registry_b" in names
    
    def test_clear_all_registries(self):
        """Test clearing all registries."""
        registry = get_registry("test")
        registry.register_class("a", ComponentA)
        
        clear_all_registries()
        
        # Should be cleared and removed
        assert list_registries() == []


class TestRegisterComponentDecorator:
    """Tests for register_component decorator."""
    
    def test_decorator_registers(self):
        """Test decorator registers in global registry."""
        @register_component("test_components", "decorated")
        class DecoratedComponent(TestComponent):
            pass
        
        registry = get_registry("test_components")
        assert registry.get("decorated") is DecoratedComponent
    
    def test_decorator_with_metadata(self):
        """Test decorator with all metadata."""
        @register_component(
            "test_components",
            "decorated",
            description="Test component",
            version=2,
            tags=["test"],
        )
        class DecoratedComponent(TestComponent):
            pass
        
        registry = get_registry("test_components")
        meta = registry.get_metadata("decorated")
        assert meta.description == "Test component"
        assert meta.version == 2
    
    def test_decorator_returns_class(self):
        """Test decorator returns original class."""
        @register_component("test_components", "decorated")
        class DecoratedComponent(TestComponent):
            value = 42
        
        # Should be able to use class normally
        assert DecoratedComponent.value == 42
        instance = DecoratedComponent()
        assert isinstance(instance, DecoratedComponent)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for real-world usage patterns."""
    
    def test_block_processor_pattern(self):
        """Test pattern similar to block processor registration."""
        processors = get_registry("block_processors")
        
        @processors.register("LoadAudio", description="Loads audio files", tags=["audio"])
        class LoadAudioProcessor:
            def process(self):
                return "audio"
        
        @processors.register("DetectOnsets", description="Detects onsets", tags=["analysis"])
        class DetectOnsetsProcessor:
            def process(self):
                return "onsets"
        
        # Look up and use
        cls = processors.get("LoadAudio")
        assert cls is LoadAudioProcessor
        
        instance = cls()
        assert instance.process() == "audio"
        
        # List all
        all_processors = processors.list_all()
        assert len(all_processors) == 2
    
    def test_block_panel_pattern(self):
        """Test pattern similar to block panel registration."""
        panels = get_registry("block_panels")
        
        @panels.register("Separator", description="Separator panel")
        class SeparatorPanel:
            def __init__(self):
                self.block_type = "Separator"
        
        @panels.register("LoadAudio", description="Load audio panel")
        class LoadAudioPanel:
            def __init__(self):
                self.block_type = "LoadAudio"
        
        # Get panel for block type
        panel_class = panels.get("Separator")
        panel = panel_class()
        assert panel.block_type == "Separator"
    
    def test_multiple_registries(self):
        """Test using multiple independent registries."""
        processors = get_registry("processors")
        panels = get_registry("panels")
        settings = get_registry("settings")
        
        processors.register_class("LoadAudio", ComponentA)
        panels.register_class("LoadAudio", ComponentB)
        settings.register_class("LoadAudio", ComponentC)
        
        # Each registry has its own component
        assert processors.get("LoadAudio") is ComponentA
        assert panels.get("LoadAudio") is ComponentB
        assert settings.get("LoadAudio") is ComponentC
