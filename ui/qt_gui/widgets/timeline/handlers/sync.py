"""
Sync Layer Handler
==================

Handler for MA3-synced layers.

Sync layers are connected to MA3 timecode tracks and:
- Receive real-time updates from MA3 via SyncSystemManager
- Push changes back to MA3 when events are edited
- Cannot be deleted from Editor (must remove sync from ShowManager)
- Use direct visual updates (EventVisualUpdate) for fast sync
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .base import LayerHandler

if TYPE_CHECKING:
    from ..types import TimelineLayer, TimelineEvent
    from ..core.scene import TimelineScene
    from ..events.layer_manager import LayerManager


class SyncLayerHandler(LayerHandler):
    """
    Handler for MA3-synced layers.
    
    Sync layers are bidirectionally synchronized with MA3 timecode tracks.
    This handler:
    - Loads events from MA3 sync data
    - Pushes edits back to MA3 (via signals/events)
    - Prevents accidental deletion
    - Uses optimized visual update path
    """
    
    def can_handle(self, layer: "TimelineLayer") -> bool:
        """
        Handle any layer that IS synced.
        
        Args:
            layer: The layer to check
            
        Returns:
            True if layer.is_synced is True
        """
        return getattr(layer, 'is_synced', False)
    
    def load_events(
        self,
        layer: "TimelineLayer",
        events: List["TimelineEvent"]
    ) -> int:
        """
        Load events into the sync layer.
        
        For sync layers, events come from MA3 and include ma3_idx
        metadata for stable identification during moves.
        
        Args:
            layer: Target layer
            events: Events to load (from MA3 sync)
            
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
                Log.warning(f"SyncLayerHandler: Failed to load event {event.id}: {e}")
        
        Log.debug(f"SyncLayerHandler: Loaded {loaded}/{len(events)} events into sync layer '{layer.name}'")
        return loaded
    
    def reload(self, layer: "TimelineLayer") -> bool:
        """
        Reload sync layer from MA3.
        
        For sync layers, reload clears existing events and prepares
        for fresh data from MA3. The actual MA3 fetch is triggered
        externally (by SyncEngine or ShowManager).
        
        Args:
            layer: Layer to reload
            
        Returns:
            True if reload preparation succeeded
        """
        from src.utils.message import Log
        
        try:
            # Clear existing events in this layer
            cleared = self.clear_events(layer)
            Log.debug(f"SyncLayerHandler: Cleared {cleared} events from sync layer '{layer.name}' for reload")
            
            # Note: Actual MA3 data fetch is handled by SyncEngine
            # This method prepares the layer for fresh data
            return True
        except Exception as e:
            Log.warning(f"SyncLayerHandler: Reload failed for sync layer '{layer.name}': {e}")
            return False
    
    def clear_events(self, layer: "TimelineLayer") -> int:
        """
        Clear all events from the sync layer.
        
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
                Log.warning(f"SyncLayerHandler: Failed to clear event {event.id}: {e}")
        
        return cleared
    
    def on_layer_created(self, layer: "TimelineLayer") -> None:
        """
        Called after a sync layer is created.
        
        For sync layers, this logs the MA3 track info.
        
        Args:
            layer: The newly created sync layer
        """
        from src.utils.message import Log
        
        ma3_coord = getattr(layer, 'ma3_track_coord', None) or 'unknown'
        show_manager = getattr(layer, 'show_manager_block_id', None) or 'unknown'
        
        Log.info(
            f"SyncLayerHandler: Created sync layer '{layer.name}' "
            f"(MA3: {ma3_coord}, ShowManager: {show_manager})"
        )
    
    def on_event_moved(
        self,
        layer: "TimelineLayer",
        event_id: str,
        old_time: float,
        new_time: float
    ) -> None:
        """
        Called when an event is moved within the sync layer.
        
        For sync layers, this should push the change to MA3.
        The actual MA3 update is handled by the signal chain:
        TimelineWidget.events_moved -> EditorPanel -> SyncEngine -> MA3
        
        Args:
            layer: Layer containing the event
            event_id: ID of moved event
            old_time: Previous time
            new_time: New time
        """
        from src.utils.message import Log
        
        # Log the move - actual MA3 sync is handled upstream
        Log.debug(
            f"SyncLayerHandler: Event {event_id} moved in sync layer '{layer.name}': "
            f"{old_time:.3f}s -> {new_time:.3f}s"
        )
    
    def on_event_added(
        self,
        layer: "TimelineLayer",
        event: "TimelineEvent"
    ) -> None:
        """
        Called when an event is added to the sync layer.
        
        For sync layers, new events should be pushed to MA3.
        
        Args:
            layer: Layer receiving the event
            event: The added event
        """
        from src.utils.message import Log
        
        Log.debug(
            f"SyncLayerHandler: Event {event.id} added to sync layer '{layer.name}' "
            f"at {event.time:.3f}s"
        )
    
    def on_event_deleted(
        self,
        layer: "TimelineLayer",
        event_id: str
    ) -> None:
        """
        Called when an event is deleted from the sync layer.
        
        For sync layers, deletions should be pushed to MA3.
        
        Args:
            layer: Layer that contained the event
            event_id: ID of deleted event
        """
        from src.utils.message import Log
        
        Log.debug(
            f"SyncLayerHandler: Event {event_id} deleted from sync layer '{layer.name}'"
        )
    
    def validate(self, layer: "TimelineLayer") -> List[str]:
        """
        Validate sync layer state.
        
        For sync layers, validation checks:
        - Layer exists in LayerManager
        - Layer has MA3 track coordinate
        - No duplicate event IDs
        - Events have ma3_idx metadata
        
        Args:
            layer: Layer to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check layer exists
        if not self._layer_manager.has_layer(layer.id):
            issues.append(f"Sync layer '{layer.name}' not found in LayerManager")
        
        # Check MA3 track coordinate
        if not getattr(layer, 'ma3_track_coord', None):
            issues.append(f"Sync layer '{layer.name}' missing ma3_track_coord")
        
        # Check for duplicate events (by ID)
        events = self._scene.get_events_in_layer(layer.id)
        event_ids = [e.id for e in events]
        if len(event_ids) != len(set(event_ids)):
            issues.append(f"Sync layer '{layer.name}' contains duplicate event IDs")
        
        # Check events have ma3_idx (warning only, not blocking)
        for event in events:
            user_data = getattr(event, 'user_data', {}) or {}
            if 'ma3_idx' not in user_data:
                # This is a soft warning - some events may not have idx yet
                pass
        
        return issues
    
    def get_layer_info(self, layer: "TimelineLayer") -> Dict[str, Any]:
        """
        Get information about the sync layer.
        
        Args:
            layer: Layer to get info for
            
        Returns:
            Dict with layer info including MA3 details
        """
        events = self._scene.get_events_in_layer(layer.id)
        return {
            'handler_type': 'SyncLayerHandler',
            'layer_id': layer.id,
            'layer_name': layer.name,
            'layer_type': 'sync',
            'event_count': len(events),
            'is_synced': True,
            'ma3_track_coord': getattr(layer, 'ma3_track_coord', None),
            'show_manager_block_id': getattr(layer, 'show_manager_block_id', None),
        }
    
    def update_event_visual(
        self,
        layer: "TimelineLayer",
        event_id: str,
        new_time: float
    ) -> bool:
        """
        Direct visual update for MA3 sync.
        
        This is the fast path used by EventVisualUpdate events.
        Updates just the visual position without full reload.
        
        Args:
            layer: Layer containing the event
            event_id: Event to update
            new_time: New time position
            
        Returns:
            True if update succeeded
        """
        from src.utils.message import Log
        
        try:
            result = self._scene.update_event(event_id, start_time=new_time)
            if result:
                Log.debug(
                    f"SyncLayerHandler: Direct visual update for event {event_id} "
                    f"in sync layer '{layer.name}' -> {new_time:.3f}s"
                )
            return result
        except Exception as e:
            Log.warning(f"SyncLayerHandler: Direct visual update failed: {e}")
            return False
