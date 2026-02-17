"""
Settings Registry

Provides auto-discovery and registration of settings classes.
This centralizes settings management and eliminates manual imports.

Usage:
    # Register a settings class
    @register_settings("my_component")
    @dataclass
    class MySettings(BaseSettings):
        theme: str = "dark"
    
    # Or register for a block type
    @register_block_settings("LoadAudio")
    @dataclass
    class LoadAudioSettings(BaseSettings):
        audio_path: str = ""
    
    # Look up settings
    settings_class = SettingsRegistry.get("my_component")
    block_settings = SettingsRegistry.get_block_settings("LoadAudio")
    
    # List all registered settings
    for key, cls, metadata in SettingsRegistry.list_all():
        print(f"{key}: {cls.__name__}")

Features:
- Decorator-based registration (auto-registers on import)
- Separate namespaces for general settings and block settings
- Metadata support for additional information
- Thread-safe singleton pattern
"""
from dataclasses import dataclass
from typing import Dict, Optional, Type, List, Tuple, Any, TypeVar
import threading

# Type variable for settings classes
T = TypeVar('T')


@dataclass
class SettingsMetadata:
    """
    Metadata for a registered settings class.
    
    Attributes:
        key: Registration key (e.g., "timeline", "LoadAudio")
        settings_class: The settings dataclass
        namespace: Category namespace (e.g., "general", "block")
        description: Optional description
        version: Optional version for migration
        tags: Optional tags for categorization
    """
    key: str
    settings_class: Type
    namespace: str = "general"
    description: str = ""
    version: int = 1
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SettingsRegistry:
    """
    Central registry for all settings classes.
    
    This is a singleton that stores all registered settings classes.
    Settings can be registered via decorators or direct calls.
    
    Namespaces:
    - "general": General application settings (timeline, preferences, etc.)
    - "block": Block-specific settings (LoadAudio, DetectOnsets, etc.)
    - Custom namespaces are also supported
    
    Thread Safety:
    - All operations are thread-safe via a lock
    - Safe to register from multiple modules during import
    
    Example:
        # Register
        SettingsRegistry.register("timeline", TimelineSettings)
        SettingsRegistry.register("LoadAudio", LoadAudioSettings, namespace="block")
        
        # Look up
        cls = SettingsRegistry.get("timeline")
        cls = SettingsRegistry.get("LoadAudio", namespace="block")
        
        # List
        for key, cls, meta in SettingsRegistry.list_all():
            print(f"{key}: {cls}")
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Storage: namespace -> key -> SettingsMetadata
    _registries: Dict[str, Dict[str, SettingsMetadata]] = {}
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._registries = {}
        return cls._instance
    
    @classmethod
    def register(
        cls,
        key: str,
        settings_class: Type,
        namespace: str = "general",
        description: str = "",
        version: int = 1,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a settings class.
        
        Args:
            key: Unique key for this settings class (e.g., "timeline", "LoadAudio")
            settings_class: The settings dataclass to register
            namespace: Category namespace (default: "general")
            description: Optional description of the settings
            version: Version number for migration support
            tags: Optional tags for categorization
        
        Raises:
            ValueError: If key is already registered in the namespace
        """
        with cls._lock:
            if namespace not in cls._registries:
                cls._registries[namespace] = {}
            
            if key in cls._registries[namespace]:
                existing = cls._registries[namespace][key]
                # Allow re-registration of the same class (happens during reloads)
                if existing.settings_class is not settings_class:
                    raise ValueError(
                        f"Settings key '{key}' already registered in namespace '{namespace}' "
                        f"with class {existing.settings_class.__name__}. "
                        f"Cannot register {settings_class.__name__}."
                    )
                return  # Same class, nothing to do
            
            metadata = SettingsMetadata(
                key=key,
                settings_class=settings_class,
                namespace=namespace,
                description=description,
                version=version,
                tags=tags or [],
            )
            
            cls._registries[namespace][key] = metadata
    
    @classmethod
    def get(cls, key: str, namespace: str = "general") -> Optional[Type]:
        """
        Get a settings class by key.
        
        Args:
            key: The registration key
            namespace: The namespace to look in (default: "general")
            
        Returns:
            The settings class, or None if not found
        """
        with cls._lock:
            if namespace not in cls._registries:
                return None
            metadata = cls._registries[namespace].get(key)
            return metadata.settings_class if metadata else None
    
    @classmethod
    def get_metadata(cls, key: str, namespace: str = "general") -> Optional[SettingsMetadata]:
        """
        Get full metadata for a settings class.
        
        Args:
            key: The registration key
            namespace: The namespace to look in (default: "general")
            
        Returns:
            SettingsMetadata, or None if not found
        """
        with cls._lock:
            if namespace not in cls._registries:
                return None
            return cls._registries[namespace].get(key)
    
    @classmethod
    def get_block_settings(cls, block_type: str) -> Optional[Type]:
        """
        Get settings class for a block type.
        
        Convenience method for looking up block settings.
        
        Args:
            block_type: The block type name (e.g., "LoadAudio", "DetectOnsets")
            
        Returns:
            The settings class, or None if not found
        """
        return cls.get(block_type, namespace="block")
    
    @classmethod
    def list_all(cls, namespace: Optional[str] = None) -> List[Tuple[str, Type, SettingsMetadata]]:
        """
        List all registered settings.
        
        Args:
            namespace: If provided, only list settings in this namespace.
                      If None, list all settings from all namespaces.
        
        Returns:
            List of (key, settings_class, metadata) tuples
        """
        with cls._lock:
            results = []
            
            namespaces = [namespace] if namespace else list(cls._registries.keys())
            
            for ns in namespaces:
                if ns in cls._registries:
                    for key, metadata in cls._registries[ns].items():
                        results.append((key, metadata.settings_class, metadata))
            
            return results
    
    @classmethod
    def list_block_settings(cls) -> List[Tuple[str, Type, SettingsMetadata]]:
        """
        List all registered block settings.
        
        Returns:
            List of (block_type, settings_class, metadata) tuples
        """
        return cls.list_all(namespace="block")
    
    @classmethod
    def list_namespaces(cls) -> List[str]:
        """
        List all registered namespaces.
        
        Returns:
            List of namespace names
        """
        with cls._lock:
            return list(cls._registries.keys())
    
    @classmethod
    def is_registered(cls, key: str, namespace: str = "general") -> bool:
        """
        Check if a key is registered.
        
        Args:
            key: The key to check
            namespace: The namespace to check in
            
        Returns:
            True if registered, False otherwise
        """
        with cls._lock:
            if namespace not in cls._registries:
                return False
            return key in cls._registries[namespace]
    
    @classmethod
    def unregister(cls, key: str, namespace: str = "general") -> bool:
        """
        Unregister a settings class.
        
        Primarily used for testing.
        
        Args:
            key: The key to unregister
            namespace: The namespace to unregister from
            
        Returns:
            True if unregistered, False if not found
        """
        with cls._lock:
            if namespace not in cls._registries:
                return False
            if key in cls._registries[namespace]:
                del cls._registries[namespace][key]
                return True
            return False
    
    @classmethod
    def clear(cls, namespace: Optional[str] = None) -> None:
        """
        Clear all registrations.
        
        Primarily used for testing.
        
        Args:
            namespace: If provided, only clear this namespace.
                      If None, clear all namespaces.
        """
        with cls._lock:
            if namespace:
                if namespace in cls._registries:
                    cls._registries[namespace] = {}
            else:
                cls._registries = {}
    
    @classmethod
    def get_by_tag(cls, tag: str, namespace: Optional[str] = None) -> List[Tuple[str, Type, SettingsMetadata]]:
        """
        Get all settings with a specific tag.
        
        Args:
            tag: The tag to filter by
            namespace: If provided, only search this namespace
            
        Returns:
            List of (key, settings_class, metadata) tuples
        """
        results = []
        for key, settings_class, metadata in cls.list_all(namespace):
            if tag in metadata.tags:
                results.append((key, settings_class, metadata))
        return results


# =============================================================================
# Decorator Functions
# =============================================================================

def register_settings(
    key: str,
    namespace: str = "general",
    description: str = "",
    version: int = 1,
    tags: Optional[List[str]] = None,
):
    """
    Decorator to register a settings class.
    
    Use this decorator on a settings dataclass to auto-register it
    when the module is imported.
    
    Args:
        key: Unique key for this settings class
        namespace: Category namespace (default: "general")
        description: Optional description
        version: Version number for migration
        tags: Optional tags for categorization
    
    Example:
        @register_settings("timeline", description="Timeline display settings")
        @dataclass
        class TimelineSettings(BaseSettings):
            zoom_level: float = 1.0
            show_grid: bool = True
    """
    def decorator(cls):
        SettingsRegistry.register(
            key=key,
            settings_class=cls,
            namespace=namespace,
            description=description,
            version=version,
            tags=tags,
        )
        return cls
    return decorator


def register_block_settings(
    block_type: str,
    description: str = "",
    version: int = 1,
    tags: Optional[List[str]] = None,
):
    """
    Decorator to register block-specific settings.
    
    Convenience decorator that registers in the "block" namespace.
    
    Args:
        block_type: The block type name (e.g., "LoadAudio", "DetectOnsets")
        description: Optional description
        version: Version number for migration
        tags: Optional tags for categorization
    
    Example:
        @register_block_settings("LoadAudio", description="Audio loading settings")
        @dataclass
        class LoadAudioSettings(BaseSettings):
            audio_path: str = ""
            sample_rate: int = 44100
    """
    return register_settings(
        key=block_type,
        namespace="block",
        description=description,
        version=version,
        tags=tags,
    )
