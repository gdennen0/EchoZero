"""
Layer Handler Registry
======================

Registry for managing layer handlers.

Provides a single point to get the appropriate handler for any layer type.
New layer types can be added by registering additional handlers.
"""

from typing import List, Optional, TYPE_CHECKING

from .base import LayerHandler
from .regular import RegularLayerHandler
from .sync import SyncLayerHandler

if TYPE_CHECKING:
    from ..types import TimelineLayer
    from ..core.scene import TimelineScene
    from ..events.layer_manager import LayerManager


class LayerHandlerRegistry:
    """
    Registry for layer handlers.
    
    Maintains a list of handlers in priority order. When asked for a handler
    for a layer, returns the first handler that can handle that layer type.
    
    Default handlers:
    1. SyncLayerHandler (for is_synced=True layers)
    2. RegularLayerHandler (fallback for all other layers)
    """
    
    def __init__(
        self,
        scene: "TimelineScene",
        layer_manager: "LayerManager"
    ):
        """
        Initialize registry with dependencies.
        
        Creates default handlers and registers them.
        
        Args:
            scene: TimelineScene for event operations
            layer_manager: LayerManager for layer operations
        """
        self._scene = scene
        self._layer_manager = layer_manager
        self._handlers: List[LayerHandler] = []
        
        # Register default handlers in priority order
        # SyncLayerHandler checks for is_synced=True, so it must come first
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register the default handler implementations."""
        # Sync handler first (specific check)
        self._handlers.append(
            SyncLayerHandler(self._scene, self._layer_manager)
        )
        # Regular handler last (fallback)
        self._handlers.append(
            RegularLayerHandler(self._scene, self._layer_manager)
        )
    
    def register_handler(self, handler: LayerHandler, priority: int = -1) -> None:
        """
        Register a custom handler.
        
        Args:
            handler: Handler instance to register
            priority: Position in handler list (0 = first, -1 = before fallback)
        """
        if priority < 0:
            # Insert before the last handler (RegularLayerHandler fallback)
            self._handlers.insert(len(self._handlers) - 1, handler)
        else:
            self._handlers.insert(priority, handler)
    
    def get_handler(self, layer: "TimelineLayer") -> LayerHandler:
        """
        Get the appropriate handler for a layer.
        
        Iterates through registered handlers in priority order and returns
        the first one that can handle this layer type.
        
        Args:
            layer: Layer to get handler for
            
        Returns:
            Handler instance for this layer type
            
        Raises:
            ValueError: If no handler can handle this layer (shouldn't happen
                       with RegularLayerHandler as fallback)
        """
        for handler in self._handlers:
            if handler.can_handle(layer):
                return handler
        
        # This should never happen if RegularLayerHandler is registered
        raise ValueError(f"No handler found for layer '{layer.name}'")
    
    def get_all_handlers(self) -> List[LayerHandler]:
        """
        Get all registered handlers.
        
        Returns:
            List of all handler instances
        """
        return list(self._handlers)


# Module-level convenience function
_registry: Optional[LayerHandlerRegistry] = None


def get_handler_for_layer(
    layer: "TimelineLayer",
    scene: "TimelineScene",
    layer_manager: "LayerManager"
) -> LayerHandler:
    """
    Convenience function to get handler for a layer.
    
    Creates a registry instance if needed and returns the appropriate handler.
    
    Args:
        layer: Layer to get handler for
        scene: TimelineScene for event operations
        layer_manager: LayerManager for layer operations
        
    Returns:
        Handler instance for this layer type
    """
    global _registry
    
    # Create registry if needed (or if dependencies changed)
    if _registry is None or _registry._scene != scene or _registry._layer_manager != layer_manager:
        _registry = LayerHandlerRegistry(scene, layer_manager)
    
    return _registry.get_handler(layer)
