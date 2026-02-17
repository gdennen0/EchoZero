"""
Regular Layer Handler
=====================

Handler for regular (non-synced) layers.

Regular layers are the default layer type. They:
- Load events from EventDataItems
- Don't sync to external systems
- Support full editing capabilities
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .base import LayerHandler

if TYPE_CHECKING:
    from ..types import TimelineLayer, TimelineEvent
    from ..core.scene import TimelineScene
    from ..events.layer_manager import LayerManager


class RegularLayerHandler(LayerHandler):
    """
    Handler for regular (non-synced) layers.
    
    This is the default handler for layers that don't have special
    sync requirements. Events are loaded directly and edits are
    applied locally without external synchronization.
    """
    
    def can_handle(self, layer: "TimelineLayer") -> bool:
        """
        Handle any layer that is NOT synced.
        
        Args:
            layer: The layer to check
            
        Returns:
            True if layer.is_synced is False
        """
        return not getattr(layer, 'is_synced', False)
    
    def load_events(
        self,
        layer: "TimelineLayer",
        events: List["TimelineEvent"]
    ) -> int:
        """
        Load events into the layer.
        
        For regular layers, this is a straightforward load -
        add each event to the scene.
        
        Args:
            layer: Target layer
            events: Events to load
            
        Returns:
            Number of events loaded
        """
        from src.utils.message import Log
        
        loaded = 0
        for event in events:
            try:
                # Ensure event is assigned to this layer
                event.layer_id = layer.id
                
                # Add to scene
                self._scene.add_event(
                    event_id=event.id,
                    start_time=event.time,
                    duration=event.duration,
                    classification=event.classification,
                    layer_id=layer.id,
                    audio_id=event.audio_id,
                    audio_name=event.audio_name,
                    user_data=event.user_data,
                )
                loaded += 1
            except Exception as e:
                Log.warning(f"RegularLayerHandler: Failed to load event {event.id}: {e}")
        
        Log.debug(f"RegularLayerHandler: Loaded {loaded}/{len(events)} events into layer '{layer.name}'")
        return loaded
    
    def reload(self, layer: "TimelineLayer") -> bool:
        """
        Reload layer from its data source.
        
        For regular layers, this clears existing events and reloads
        from the cached EventDataItem data. The actual data fetching
        is handled by EditorPanel; this handler just manages the visual.
        
        Args:
            layer: Layer to reload
            
        Returns:
            True if reload succeeded
        """
        from src.utils.message import Log
        
        try:
            # Clear existing events in this layer
            cleared = self.clear_events(layer)
            Log.debug(f"RegularLayerHandler: Cleared {cleared} events from layer '{layer.name}' for reload")
            
            # Note: Actual event reloading is handled by EditorPanel calling load_events()
            # This method just prepares the layer for reload
            return True
        except Exception as e:
            Log.warning(f"RegularLayerHandler: Reload failed for layer '{layer.name}': {e}")
            return False
    
    def clear_events(self, layer: "TimelineLayer") -> int:
        """
        Clear all events from the layer.
        
        Args:
            layer: Layer to clear
            
        Returns:
            Number of events cleared
        """
        from src.utils.message import Log
        
        # Get all events in this layer
        events_in_layer = self._scene.get_events_in_layer(layer.id)
        cleared = 0
        
        for event in events_in_layer:
            try:
                self._scene.remove_event(event.id)
                cleared += 1
            except Exception as e:
                Log.warning(f"RegularLayerHandler: Failed to clear event {event.id}: {e}")
        
        return cleared
    
    def validate(self, layer: "TimelineLayer") -> List[str]:
        """
        Validate layer state.
        
        For regular layers, validation checks:
        - Layer exists in LayerManager
        - No duplicate event IDs
        
        Args:
            layer: Layer to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check layer exists
        if not self._layer_manager.has_layer(layer.id):
            issues.append(f"Layer '{layer.name}' not found in LayerManager")
        
        # Check for duplicate events (by ID)
        events = self._scene.get_events_in_layer(layer.id)
        event_ids = [e.id for e in events]
        if len(event_ids) != len(set(event_ids)):
            issues.append(f"Layer '{layer.name}' contains duplicate event IDs")
        
        return issues
    
    def get_layer_info(self, layer: "TimelineLayer") -> Dict[str, Any]:
        """
        Get information about the layer.
        
        Args:
            layer: Layer to get info for
            
        Returns:
            Dict with layer info
        """
        events = self._scene.get_events_in_layer(layer.id)
        return {
            'handler_type': 'RegularLayerHandler',
            'layer_id': layer.id,
            'layer_name': layer.name,
            'layer_type': 'regular',
            'event_count': len(events),
            'is_synced': False,
        }
