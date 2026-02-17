"""
Tests for the SettingsRegistry.

Tests registration, lookup, decorators, and namespaces.
"""
import pytest
from dataclasses import dataclass

from src.shared.application.settings.settings_registry import (
    SettingsRegistry,
    SettingsMetadata,
    register_settings,
    register_block_settings,
)
from src.application.settings.base_settings import BaseSettings


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the registry before and after each test."""
    SettingsRegistry.clear()
    yield
    SettingsRegistry.clear()


class TestSettingsRegistry:
    """Tests for SettingsRegistry class."""
    
    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        @dataclass
        class TestSettings(BaseSettings):
            value: str = "default"
        
        SettingsRegistry.register("test", TestSettings)
        
        result = SettingsRegistry.get("test")
        assert result is TestSettings
    
    def test_register_with_namespace(self):
        """Test registration with custom namespace."""
        @dataclass
        class TestSettings(BaseSettings):
            value: str = "default"
        
        SettingsRegistry.register("test", TestSettings, namespace="custom")
        
        # Should not find in default namespace
        assert SettingsRegistry.get("test") is None
        
        # Should find in custom namespace
        result = SettingsRegistry.get("test", namespace="custom")
        assert result is TestSettings
    
    def test_get_block_settings(self):
        """Test block settings convenience method."""
        @dataclass
        class LoadAudioSettings(BaseSettings):
            path: str = ""
        
        SettingsRegistry.register("LoadAudio", LoadAudioSettings, namespace="block")
        
        result = SettingsRegistry.get_block_settings("LoadAudio")
        assert result is LoadAudioSettings
    
    def test_get_nonexistent(self):
        """Test getting non-existent settings returns None."""
        result = SettingsRegistry.get("nonexistent")
        assert result is None
    
    def test_get_metadata(self):
        """Test retrieving full metadata."""
        @dataclass
        class TestSettings(BaseSettings):
            value: str = "default"
        
        SettingsRegistry.register(
            "test",
            TestSettings,
            description="Test description",
            version=2,
            tags=["audio", "processing"],
        )
        
        metadata = SettingsRegistry.get_metadata("test")
        assert metadata is not None
        assert metadata.key == "test"
        assert metadata.settings_class is TestSettings
        assert metadata.description == "Test description"
        assert metadata.version == 2
        assert "audio" in metadata.tags
    
    def test_duplicate_registration_same_class(self):
        """Test that re-registering the same class is allowed."""
        @dataclass
        class TestSettings(BaseSettings):
            value: str = "default"
        
        SettingsRegistry.register("test", TestSettings)
        # Should not raise
        SettingsRegistry.register("test", TestSettings)
        
        result = SettingsRegistry.get("test")
        assert result is TestSettings
    
    def test_duplicate_registration_different_class(self):
        """Test that registering different class with same key raises."""
        @dataclass
        class TestSettings1(BaseSettings):
            value: str = "default"
        
        @dataclass
        class TestSettings2(BaseSettings):
            other: int = 0
        
        SettingsRegistry.register("test", TestSettings1)
        
        with pytest.raises(ValueError) as exc:
            SettingsRegistry.register("test", TestSettings2)
        
        assert "already registered" in str(exc.value)
    
    def test_list_all(self):
        """Test listing all settings."""
        @dataclass
        class Settings1(BaseSettings):
            pass
        
        @dataclass
        class Settings2(BaseSettings):
            pass
        
        SettingsRegistry.register("one", Settings1)
        SettingsRegistry.register("two", Settings2)
        
        results = SettingsRegistry.list_all()
        assert len(results) == 2
        
        keys = [r[0] for r in results]
        assert "one" in keys
        assert "two" in keys
    
    def test_list_all_with_namespace(self):
        """Test listing settings in a specific namespace."""
        @dataclass
        class GeneralSettings(BaseSettings):
            pass
        
        @dataclass
        class BlockSettings(BaseSettings):
            pass
        
        SettingsRegistry.register("general", GeneralSettings, namespace="general")
        SettingsRegistry.register("block", BlockSettings, namespace="block")
        
        # List only block namespace
        results = SettingsRegistry.list_all(namespace="block")
        assert len(results) == 1
        assert results[0][0] == "block"
    
    def test_list_block_settings(self):
        """Test convenience method for listing block settings."""
        @dataclass
        class LoadAudio(BaseSettings):
            pass
        
        @dataclass
        class DetectOnsets(BaseSettings):
            pass
        
        SettingsRegistry.register("LoadAudio", LoadAudio, namespace="block")
        SettingsRegistry.register("DetectOnsets", DetectOnsets, namespace="block")
        
        results = SettingsRegistry.list_block_settings()
        assert len(results) == 2
        
        block_types = [r[0] for r in results]
        assert "LoadAudio" in block_types
        assert "DetectOnsets" in block_types
    
    def test_list_namespaces(self):
        """Test listing available namespaces."""
        @dataclass
        class Settings1(BaseSettings):
            pass
        
        @dataclass
        class Settings2(BaseSettings):
            pass
        
        SettingsRegistry.register("one", Settings1, namespace="ns1")
        SettingsRegistry.register("two", Settings2, namespace="ns2")
        
        namespaces = SettingsRegistry.list_namespaces()
        assert "ns1" in namespaces
        assert "ns2" in namespaces
    
    def test_is_registered(self):
        """Test checking if a key is registered."""
        @dataclass
        class TestSettings(BaseSettings):
            pass
        
        assert SettingsRegistry.is_registered("test") is False
        
        SettingsRegistry.register("test", TestSettings)
        
        assert SettingsRegistry.is_registered("test") is True
        assert SettingsRegistry.is_registered("test", namespace="other") is False
    
    def test_unregister(self):
        """Test unregistering a settings class."""
        @dataclass
        class TestSettings(BaseSettings):
            pass
        
        SettingsRegistry.register("test", TestSettings)
        assert SettingsRegistry.get("test") is TestSettings
        
        result = SettingsRegistry.unregister("test")
        assert result is True
        assert SettingsRegistry.get("test") is None
    
    def test_unregister_nonexistent(self):
        """Test unregistering non-existent key returns False."""
        result = SettingsRegistry.unregister("nonexistent")
        assert result is False
    
    def test_clear_all(self):
        """Test clearing all registrations."""
        @dataclass
        class Settings1(BaseSettings):
            pass
        
        @dataclass
        class Settings2(BaseSettings):
            pass
        
        SettingsRegistry.register("one", Settings1, namespace="ns1")
        SettingsRegistry.register("two", Settings2, namespace="ns2")
        
        SettingsRegistry.clear()
        
        assert SettingsRegistry.list_namespaces() == []
    
    def test_clear_specific_namespace(self):
        """Test clearing a specific namespace."""
        @dataclass
        class Settings1(BaseSettings):
            pass
        
        @dataclass
        class Settings2(BaseSettings):
            pass
        
        SettingsRegistry.register("one", Settings1, namespace="ns1")
        SettingsRegistry.register("two", Settings2, namespace="ns2")
        
        SettingsRegistry.clear(namespace="ns1")
        
        assert SettingsRegistry.get("one", namespace="ns1") is None
        assert SettingsRegistry.get("two", namespace="ns2") is Settings2
    
    def test_get_by_tag(self):
        """Test filtering settings by tag."""
        @dataclass
        class AudioSettings(BaseSettings):
            pass
        
        @dataclass
        class VideoSettings(BaseSettings):
            pass
        
        @dataclass
        class GeneralSettings(BaseSettings):
            pass
        
        SettingsRegistry.register("audio", AudioSettings, tags=["media", "audio"])
        SettingsRegistry.register("video", VideoSettings, tags=["media", "video"])
        SettingsRegistry.register("general", GeneralSettings, tags=["app"])
        
        # Get all media settings
        media_settings = SettingsRegistry.get_by_tag("media")
        assert len(media_settings) == 2
        
        keys = [s[0] for s in media_settings]
        assert "audio" in keys
        assert "video" in keys
        assert "general" not in keys


class TestRegisterSettingsDecorator:
    """Tests for @register_settings decorator."""
    
    def test_decorator_registers_class(self):
        """Test that decorator registers the class."""
        @register_settings("decorated")
        @dataclass
        class DecoratedSettings(BaseSettings):
            value: str = "default"
        
        result = SettingsRegistry.get("decorated")
        assert result is DecoratedSettings
    
    def test_decorator_with_options(self):
        """Test decorator with all options."""
        @register_settings(
            "decorated",
            namespace="custom",
            description="Test settings",
            version=3,
            tags=["test"],
        )
        @dataclass
        class DecoratedSettings(BaseSettings):
            value: str = "default"
        
        metadata = SettingsRegistry.get_metadata("decorated", namespace="custom")
        assert metadata is not None
        assert metadata.description == "Test settings"
        assert metadata.version == 3
        assert "test" in metadata.tags
    
    def test_decorator_returns_original_class(self):
        """Test that decorator returns the original class unchanged."""
        @register_settings("decorated")
        @dataclass
        class DecoratedSettings(BaseSettings):
            value: str = "default"
        
        # Should be able to instantiate normally
        instance = DecoratedSettings(value="test")
        assert instance.value == "test"


class TestRegisterBlockSettingsDecorator:
    """Tests for @register_block_settings decorator."""
    
    def test_decorator_registers_in_block_namespace(self):
        """Test that decorator registers in 'block' namespace."""
        @register_block_settings("TestBlock")
        @dataclass
        class TestBlockSettings(BaseSettings):
            value: str = "default"
        
        # Should be in block namespace
        result = SettingsRegistry.get_block_settings("TestBlock")
        assert result is TestBlockSettings
        
        # Should not be in general namespace
        assert SettingsRegistry.get("TestBlock") is None
    
    def test_decorator_with_options(self):
        """Test decorator with options."""
        @register_block_settings(
            "TestBlock",
            description="Test block settings",
            version=2,
            tags=["audio"],
        )
        @dataclass
        class TestBlockSettings(BaseSettings):
            value: str = "default"
        
        metadata = SettingsRegistry.get_metadata("TestBlock", namespace="block")
        assert metadata is not None
        assert metadata.description == "Test block settings"
        assert metadata.version == 2


class TestSettingsMetadata:
    """Tests for SettingsMetadata dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        @dataclass
        class TestSettings(BaseSettings):
            pass
        
        metadata = SettingsMetadata(
            key="test",
            settings_class=TestSettings,
        )
        
        assert metadata.key == "test"
        assert metadata.namespace == "general"
        assert metadata.description == ""
        assert metadata.version == 1
        assert metadata.tags == []
    
    def test_tags_default_to_empty_list(self):
        """Test that tags defaults to empty list, not None."""
        @dataclass
        class TestSettings(BaseSettings):
            pass
        
        metadata = SettingsMetadata(
            key="test",
            settings_class=TestSettings,
            tags=None,
        )
        
        # Should be empty list, not None
        assert metadata.tags == []


class TestIntegration:
    """Integration tests for real-world usage patterns."""
    
    def test_multiple_block_settings(self):
        """Test registering multiple block settings."""
        @register_block_settings("LoadAudio", description="Load audio files")
        @dataclass
        class LoadAudioSettings(BaseSettings):
            audio_path: str = ""
        
        @register_block_settings("DetectOnsets", description="Detect audio onsets")
        @dataclass
        class DetectOnsetsSettings(BaseSettings):
            threshold: float = 0.5
        
        @register_block_settings("Separator", description="Separate audio sources")
        @dataclass
        class SeparatorSettings(BaseSettings):
            model: str = "htdemucs"
        
        # All should be registered
        assert SettingsRegistry.get_block_settings("LoadAudio") is LoadAudioSettings
        assert SettingsRegistry.get_block_settings("DetectOnsets") is DetectOnsetsSettings
        assert SettingsRegistry.get_block_settings("Separator") is SeparatorSettings
        
        # Should have 3 block settings
        block_settings = SettingsRegistry.list_block_settings()
        assert len(block_settings) == 3
    
    def test_mixed_namespaces(self):
        """Test using multiple namespaces together."""
        @register_settings("app", description="App preferences")
        @dataclass
        class AppSettings(BaseSettings):
            theme: str = "dark"
        
        @register_settings("timeline", namespace="ui", description="Timeline UI")
        @dataclass
        class TimelineSettings(BaseSettings):
            zoom: float = 1.0
        
        @register_block_settings("Editor", description="Editor block")
        @dataclass
        class EditorSettings(BaseSettings):
            auto_save: bool = True
        
        # Check all namespaces
        namespaces = SettingsRegistry.list_namespaces()
        assert "general" in namespaces
        assert "ui" in namespaces
        assert "block" in namespaces
        
        # Check retrieval from each
        assert SettingsRegistry.get("app") is AppSettings
        assert SettingsRegistry.get("timeline", namespace="ui") is TimelineSettings
        assert SettingsRegistry.get_block_settings("Editor") is EditorSettings
