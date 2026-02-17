"""
Layer Handler Base Class
========================

Abstract base class defining the interface for layer-type-specific behavior.

All layer types (regular, sync, etc.) implement this interface to provide
standardized operations while allowing type-specific behavior.

This is the Strategy pattern - the TimelineWidget uses handlers to delegate
layer-specific operations without knowing the concrete implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TimelineLayer, TimelineEvent
    from ..core.scene import TimelineScene
    from ..events.layer_manager import LayerManager


class LayerHandler(ABC):
    """
    Strategy interface for layer-type-specific behavior.
    
    Implementations handle:
    - Loading events into a layer
    - Reloading a layer from its data source
    - Responding to event changes (moves, edits)
    - Validating layer state
    
    Each handler has access to the scene and layer_manager for performing
    operations, but the specific behavior varies by layer type.
    """
    
    def __init__(
        self,
        scene: "TimelineScene",
        layer_manager: "LayerManager"
    ):
        """
        Initialize handler with required dependencies.
        
        Args:
            scene: TimelineScene for event operations
            layer_manager: LayerManager for layer operations
        """
        self._scene = scene
        self._layer_manager = layer_manager
    
    @abstractmethod
    def can_handle(self, layer: "TimelineLayer") -> bool:
        """
        Check if this handler can manage this layer type.
        
        Args:
            layer: The layer to check
            
        Returns:
            True if this handler should manage this layer
        """
        pass
    
    @abstractmethod
    def load_events(
        self,
        layer: "TimelineLayer",
        events: List["TimelineEvent"]
    ) -> int:
        """
        Load events into the layer.
        
        Args:
            layer: Target layer
            events: Events to load
            
        Returns:
            Number of events loaded
        """
        pass
    
    @abstractmethod
    def reload(self, layer: "TimelineLayer") -> bool:
        """
        Reload layer from its data source.
        
        For regular layers, this reloads from EventDataItem.
        For sync layers, this may fetch from MA3.
        
        Args:
            layer: Layer to reload
            
        Returns:
            True if reload succeeded
        """
        pass
    
    @abstractmethod
    def clear_events(self, layer: "TimelineLayer") -> int:
        """
        Clear all events from the layer.
        
        Args:
            layer: Layer to clear
            
        Returns:
            Number of events cleared
        """
        pass
    
    def on_layer_created(self, layer: "TimelineLayer") -> None:
        """
        Called after a layer is created.
        
        Override to perform handler-specific setup.
        
        Args:
            layer: The newly created layer
        """
        pass
    
    def on_event_moved(
        self,
        layer: "TimelineLayer",
        event_id: str,
        old_time: float,
        new_time: float
    ) -> None:
        """
        Called when an event is moved within the layer.
        
        Override to handle event moves (e.g., sync to MA3).
        
        Args:
            layer: Layer containing the event
            event_id: ID of moved event
            old_time: Previous time
            new_time: New time
        """
        pass
    
    def on_event_added(
        self,
        layer: "TimelineLayer",
        event: "TimelineEvent"
    ) -> None:
        """
        Called when an event is added to the layer.
        
        Override to handle event additions (e.g., sync to MA3).
        
        Args:
            layer: Layer receiving the event
            event: The added event
        """
        pass
    
    def on_event_deleted(
        self,
        layer: "TimelineLayer",
        event_id: str
    ) -> None:
        """
        Called when an event is deleted from the layer.
        
        Override to handle event deletions (e.g., sync to MA3).
        
        Args:
            layer: Layer that contained the event
            event_id: ID of deleted event
        """
        pass
    
    def validate(self, layer: "TimelineLayer") -> List[str]:
        """
        Validate layer state.
        
        Args:
            layer: Layer to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        return []
    
    def get_layer_info(self, layer: "TimelineLayer") -> Dict[str, Any]:
        """
        Get handler-specific information about the layer.
        
        Args:
            layer: Layer to get info for
            
        Returns:
            Dict with layer info (type, status, etc.)
        """
        return {
            'handler_type': self.__class__.__name__,
            'layer_id': layer.id,
            'layer_name': layer.name,
        }
