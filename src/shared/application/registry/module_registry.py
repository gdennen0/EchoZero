"""
Module Registry Pattern

Provides a unified, generic registry pattern for component registration.
This consolidates the various registration patterns in the codebase:
- Block processors
- Block panels
- Settings classes
- Commands
- etc.

Usage:
    # Create a registry for a specific component type
    processor_registry = ModuleRegistry("BlockProcessor")
    
    # Register via decorator
    @processor_registry.register("LoadAudio")
    class LoadAudioProcessor(BlockProcessor):
        pass
    
    # Or register directly
    processor_registry.register_class("DetectOnsets", DetectOnsetsProcessor)
    
    # Look up
    cls = processor_registry.get("LoadAudio")
    
    # List all
    for key, cls, meta in processor_registry.list_all():
        print(f"{key}: {cls.__name__}")
    
    # Use global registries
    from src.shared.application.registry import get_registry
    panels = get_registry("block_panels")
    panels.register_class("Separator", SeparatorPanel)

Features:
- Type-safe generic registry
- Decorator-based or direct registration
- Namespaces for grouping
- Metadata support (description, version, tags)
- Thread-safe operations
- Global registry factory
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, List, Tuple, Any, TypeVar, Generic, Callable
import threading

from src.utils.message import Log


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar('T')  # Component type


# =============================================================================
# Component Metadata
# =============================================================================

@dataclass
class ComponentMetadata:
    """
    Metadata for a registered component.
    
    Attributes:
        key: Registration key (e.g., "LoadAudio", "Separator")
        component_class: The registered class
        registry_name: Name of the registry this belongs to
        description: Optional description
        version: Optional version for migration
        tags: Optional tags for categorization
        extra: Extra metadata (flexible dict)
    """
    key: str
    component_class: Type
    registry_name: str = ""
    description: str = ""
    version: int = 1
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.extra is None:
            self.extra = {}


# =============================================================================
# Module Registry
# =============================================================================

class ModuleRegistry(Generic[T]):
    """
    Generic registry for component registration.
    
    Provides a unified pattern for registering and looking up components
    by key. Can be used for any component type: processors, panels, 
    settings, commands, etc.
    
    Thread-safe and supports decorator-based registration.
    
    Example:
        # Create registry
        panel_registry = ModuleRegistry[BlockPanelBase]("BlockPanel")
        
        # Register via decorator
        @panel_registry.register("Separator", description="Separator panel")
        class SeparatorPanel(BlockPanelBase):
            pass
        
        # Look up
        panel_class = panel_registry.get("Separator")
        
        # List all
        for key, cls, meta in panel_registry.list_all():
            print(f"{key}: {cls.__name__}")
    
    Attributes:
        name: Registry name for logging and identification
    """
    
    def __init__(self, name: str):
        """
        Initialize registry.
        
        Args:
            name: Registry name (e.g., "BlockProcessor", "BlockPanel")
        """
        self._name = name
        self._components: Dict[str, ComponentMetadata] = {}
        self._lock = threading.Lock()
        Log.debug(f"ModuleRegistry: Created '{name}' registry")
    
    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name
    
    def register(
        self,
        key: str,
        description: str = "",
        version: int = 1,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a component class.
        
        Args:
            key: Unique key for this component
            description: Optional description
            version: Version number for migration
            tags: Optional tags for categorization
            extra: Extra metadata
        
        Returns:
            Decorator function
        
        Example:
            @registry.register("LoadAudio", description="Loads audio files")
            class LoadAudioProcessor(BlockProcessor):
                pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self.register_class(
                key=key,
                component_class=cls,
                description=description,
                version=version,
                tags=tags,
                extra=extra,
            )
            return cls
        return decorator
    
    def register_class(
        self,
        key: str,
        component_class: Type[T],
        description: str = "",
        version: int = 1,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a component class directly.
        
        Args:
            key: Unique key for this component
            component_class: The class to register
            description: Optional description
            version: Version number
            tags: Optional tags
            extra: Extra metadata
        
        Raises:
            ValueError: If key is already registered with a different class
        """
        with self._lock:
            if key in self._components:
                existing = self._components[key]
                # Allow re-registration of same class (happens during reloads)
                if existing.component_class is not component_class:
                    raise ValueError(
                        f"ModuleRegistry '{self._name}': Key '{key}' already registered "
                        f"with class {existing.component_class.__name__}. "
                        f"Cannot register {component_class.__name__}."
                    )
                return  # Same class, nothing to do
            
            metadata = ComponentMetadata(
                key=key,
                component_class=component_class,
                registry_name=self._name,
                description=description,
                version=version,
                tags=tags or [],
                extra=extra or {},
            )
            
            self._components[key] = metadata
            Log.debug(f"ModuleRegistry '{self._name}': Registered '{key}' -> {component_class.__name__}")
    
    def get(self, key: str) -> Optional[Type[T]]:
        """
        Get a component class by key.
        
        Args:
            key: The registration key
            
        Returns:
            The component class, or None if not found
        """
        with self._lock:
            metadata = self._components.get(key)
            return metadata.component_class if metadata else None
    
    def get_metadata(self, key: str) -> Optional[ComponentMetadata]:
        """
        Get full metadata for a component.
        
        Args:
            key: The registration key
            
        Returns:
            ComponentMetadata, or None if not found
        """
        with self._lock:
            return self._components.get(key)
    
    def list_all(self) -> List[Tuple[str, Type[T], ComponentMetadata]]:
        """
        List all registered components.
        
        Returns:
            List of (key, component_class, metadata) tuples
        """
        with self._lock:
            return [
                (key, meta.component_class, meta)
                for key, meta in self._components.items()
            ]
    
    def list_keys(self) -> List[str]:
        """
        List all registered keys.
        
        Returns:
            List of keys
        """
        with self._lock:
            return list(self._components.keys())
    
    def is_registered(self, key: str) -> bool:
        """
        Check if a key is registered.
        
        Args:
            key: The key to check
            
        Returns:
            True if registered
        """
        with self._lock:
            return key in self._components
    
    def unregister(self, key: str) -> bool:
        """
        Unregister a component.
        
        Args:
            key: The key to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if key in self._components:
                del self._components[key]
                Log.debug(f"ModuleRegistry '{self._name}': Unregistered '{key}'")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._components.clear()
            Log.debug(f"ModuleRegistry '{self._name}': Cleared all registrations")
    
    def count(self) -> int:
        """Get count of registered components."""
        with self._lock:
            return len(self._components)
    
    def get_by_tag(self, tag: str) -> List[Tuple[str, Type[T], ComponentMetadata]]:
        """
        Get all components with a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of (key, component_class, metadata) tuples
        """
        results = []
        with self._lock:
            for key, meta in self._components.items():
                if tag in meta.tags:
                    results.append((key, meta.component_class, meta))
        return results
    
    def search(self, query: str) -> List[Tuple[str, Type[T], ComponentMetadata]]:
        """
        Search components by key, description, or tags.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching (key, component_class, metadata) tuples
        """
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for key, meta in self._components.items():
                # Search in key
                if query_lower in key.lower():
                    results.append((key, meta.component_class, meta))
                    continue
                
                # Search in description
                if query_lower in meta.description.lower():
                    results.append((key, meta.component_class, meta))
                    continue
                
                # Search in tags
                if any(query_lower in tag.lower() for tag in meta.tags):
                    results.append((key, meta.component_class, meta))
                    continue
                
                # Search in class name
                if query_lower in meta.component_class.__name__.lower():
                    results.append((key, meta.component_class, meta))
                    continue
        
        return results


# =============================================================================
# Global Registry Factory
# =============================================================================

# Global storage for named registries
_global_registries: Dict[str, ModuleRegistry] = {}
_global_lock = threading.Lock()


def get_registry(name: str) -> ModuleRegistry:
    """
    Get or create a global registry by name.
    
    This provides a central access point for registries, allowing
    different parts of the codebase to share the same registry.
    
    Args:
        name: Registry name (e.g., "block_processors", "block_panels")
        
    Returns:
        ModuleRegistry instance
    
    Example:
        # In processor module
        processors = get_registry("block_processors")
        processors.register_class("LoadAudio", LoadAudioProcessor)
        
        # In execution engine
        processors = get_registry("block_processors")
        cls = processors.get("LoadAudio")
    """
    with _global_lock:
        if name not in _global_registries:
            _global_registries[name] = ModuleRegistry(name)
            Log.debug(f"ModuleRegistry: Created global registry '{name}'")
        return _global_registries[name]


def list_registries() -> List[str]:
    """
    List all global registry names.
    
    Returns:
        List of registry names
    """
    with _global_lock:
        return list(_global_registries.keys())


def clear_all_registries() -> None:
    """
    Clear all global registries.
    
    Primarily used for testing.
    """
    with _global_lock:
        for registry in _global_registries.values():
            registry.clear()
        _global_registries.clear()
        Log.debug("ModuleRegistry: Cleared all global registries")


# =============================================================================
# Convenience Decorator Factory
# =============================================================================

def register_component(
    registry_name: str,
    key: str,
    description: str = "",
    version: int = 1,
    tags: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a component in a global registry.
    
    Convenience function that gets the registry and registers in one step.
    
    Args:
        registry_name: Name of the registry to use
        key: Unique key for this component
        description: Optional description
        version: Version number
        tags: Optional tags
        extra: Extra metadata
    
    Returns:
        Decorator function
    
    Example:
        @register_component("block_panels", "Separator", description="Separator panel")
        class SeparatorPanel(BlockPanelBase):
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        registry = get_registry(registry_name)
        registry.register_class(
            key=key,
            component_class=cls,
            description=description,
            version=version,
            tags=tags,
            extra=extra,
        )
        return cls
    return decorator
