"""
Sync Layer Command

Syncs a single layer/track between Editor and MA3.
Uses EditorAPI for Editor operations and MA3CommunicationService for MA3.
"""
from typing import TYPE_CHECKING, Optional, List, Dict, Any

from src.application.commands.base_command import EchoZeroCommand
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class SyncLayerCommand(EchoZeroCommand):
    """
    Sync a single layer/track between Editor and MA3 (undoable).
    
    Direction determines sync behavior:
    - "editor_to_ma3": Read events from Editor layer, send to MA3 track
    - "ma3_to_editor": Read events from MA3 track, add to Editor layer
    
    Uses EditorAPI for all Editor operations to ensure proper signal emission.
    
    Args:
        facade: ApplicationFacade instance
        show_manager_block_id: ShowManager block ID
        editor_block_id: Editor block ID
        editor_layer_name: Editor layer name
        ma3_track_coord: MA3 track coordinate (e.g., "tc101_tg1_tr1")
        direction: "editor_to_ma3" or "ma3_to_editor"
        clear_target: Whether to clear target before sync (default True)
    """
    
    COMMAND_TYPE = "layer_sync.sync_layer"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        show_manager_block_id: str,
        editor_block_id: str,
        editor_layer_name: str,
        ma3_track_coord: str,
        direction: str = "editor_to_ma3",
        clear_target: bool = True
    ):
        super().__init__(facade, f"Sync Layer: {editor_layer_name} <-> {ma3_track_coord}")
        
        self._show_manager_block_id = show_manager_block_id
        self._editor_block_id = editor_block_id
        self._editor_layer_name = editor_layer_name
        self._ma3_track_coord = ma3_track_coord
        self._direction = direction
        self._clear_target = clear_target
        
        # State for undo
        self._original_editor_events: List[Dict[str, Any]] = []
        self._original_ma3_events: List[Dict[str, Any]] = []
        self._events_added_count: int = 0
    
    def redo(self):
        """Execute the sync operation."""
        if self._direction == "editor_to_ma3":
            self._sync_editor_to_ma3()
        elif self._direction == "ma3_to_editor":
            self._sync_ma3_to_editor()
        else:
            self._log_error(f"Invalid sync direction: {self._direction}")
    
    def _sync_editor_to_ma3(self):
        """
        Sync Editor layer events to MA3 track.
        
        1. Get EditorAPI for the block
        2. Read events from Editor layer using EditorAPI
        3. Convert to MA3 format
        4. Send OSC commands to MA3
        """
        from src.features.blocks.application.editor_api import EditorAPI, create_editor_api
        
        try:
            # Get or create EditorAPI
            editor_api = create_editor_api(self._facade, self._editor_block_id)
            
            # Get events from Editor layer
            events = editor_api.get_events_in_layer(self._editor_layer_name)
            
            if not events:
                Log.info(f"SyncLayerCommand: No events in Editor layer '{self._editor_layer_name}'")
                return
            
            Log.info(f"SyncLayerCommand: Syncing {len(events)} events from Editor to MA3")
            
            # Parse MA3 track coordinate
            timecode_no, track_group, track = self._parse_ma3_coord(self._ma3_track_coord)
            
            # Get MA3 communication service
            ma3_service = self._get_ma3_service()
            if not ma3_service:
                self._log_warning("MA3 communication service not available")
                return
            
            # Clear target track if requested
            if self._clear_target:
                self._send_clear_track_osc(ma3_service, timecode_no, track_group, track)
            
            # Send events to MA3
            for i, event in enumerate(events):
                self._send_add_event_osc(
                    ma3_service,
                    timecode_no,
                    track_group,
                    track,
                    event_index=i,
                    time=event.time,
                    event_type="cmd",
                    name=event.classification,
                    metadata=event.metadata or {}
                )
            
            self._events_added_count = len(events)
            Log.info(f"SyncLayerCommand: Sent {len(events)} events to MA3 track {self._ma3_track_coord}")
            
        except Exception as e:
            self._log_error(f"Failed to sync Editor to MA3: {e}")
            raise
    
    def _sync_ma3_to_editor(self):
        """
        Sync MA3 track events to Editor layer.
        
        1. Get MA3 events from ShowManager state
        2. Create/update Editor layer using EditorAPI
        3. Add events to layer using EditorAPI
        """
        from src.features.blocks.application.editor_api import EditorAPI, create_editor_api
        
        try:
            # Get or create EditorAPI
            editor_api = create_editor_api(self._facade, self._editor_block_id)
            
            # Store original events for undo
            original_events = editor_api.get_events_in_layer(self._editor_layer_name)
            self._original_editor_events = [e.to_dict() for e in original_events]
            
            # Get MA3 events from ShowManager state
            ma3_events = self._get_ma3_track_events()
            
            if not ma3_events:
                Log.info(f"SyncLayerCommand: No MA3 events for track '{self._ma3_track_coord}'")
                return
            
            Log.info(f"SyncLayerCommand: Syncing {len(ma3_events)} events from MA3 to Editor")
            
            # Ensure layer exists
            if not editor_api.layer_exists(self._editor_layer_name):
                editor_api.create_layer(
                    name=self._editor_layer_name,
                    is_synced=True,
                    show_manager_block_id=self._show_manager_block_id,
                    ma3_track_coord=self._ma3_track_coord
                )
            else:
                # Mark existing layer as synced
                editor_api.mark_layer_synced(
                    self._editor_layer_name,
                    self._show_manager_block_id,
                    self._ma3_track_coord
                )
            
            # Clear existing events if requested
            if self._clear_target:
                editor_api.clear_layer_events(self._editor_layer_name)
            
            # Convert MA3 events to Editor format and add
            events_to_add = []
            for ma3_event in ma3_events:
                events_to_add.append({
                    'time': ma3_event.get('time', 0.0),
                    'duration': 0.0,
                    'classification': self._editor_layer_name,
                    'metadata': {
                        'source': 'ma3',
                        'ma3_coord': self._ma3_track_coord,
                        'ma3_event_index': ma3_event.get('event_index'),
                        'ma3_name': ma3_event.get('name', ''),
                        'ma3_cmd': ma3_event.get('cmd', ''),
                    }
                })
            
            # Add events using EditorAPI (emits signals)
            added_count = editor_api.add_events(events_to_add, source="ma3")
            self._events_added_count = added_count
            
            Log.info(f"SyncLayerCommand: Added {added_count} events to Editor layer '{self._editor_layer_name}'")
            
        except Exception as e:
            self._log_error(f"Failed to sync MA3 to Editor: {e}")
            raise
    
    def _get_ma3_track_events(self) -> List[Dict[str, Any]]:
        """
        Get MA3 events for the specified track from ShowManager state.
        
        Returns:
            List of event dicts with time, name, cmd, etc.
        """
        # Try to get from ShowManager's MA3 event cache
        try:
            result = self._facade.describe_block(self._show_manager_block_id)
            if not result.success or not result.data:
                return []
            
            block = result.data
            ma3_events = block.metadata.get('ma3_events', [])
            
            # Filter events for this track coordinate
            track_events = []
            for event in ma3_events:
                event_coord = event.get('coord', '')
                if event_coord == self._ma3_track_coord:
                    track_events.append(event)
            
            return track_events
            
        except Exception as e:
            Log.warning(f"SyncLayerCommand: Failed to get MA3 events: {e}")
            return []
    
    def _parse_ma3_coord(self, coord: str) -> tuple:
        """
        Parse MA3 track coordinate.
        
        Args:
            coord: Coordinate string like "tc101_tg1_tr1"
            
        Returns:
            Tuple of (timecode_no, track_group, track)
        """
        try:
            parts = coord.split('_')
            timecode_no = int(parts[0].replace('tc', ''))
            track_group = int(parts[1].replace('tg', ''))
            track = int(parts[2].replace('tr', ''))
            return timecode_no, track_group, track
        except (IndexError, ValueError) as e:
            Log.warning(f"SyncLayerCommand: Failed to parse MA3 coord '{coord}': {e}")
            return 101, 1, 1  # Default values
    
    def _get_ma3_service(self):
        """Get MA3CommunicationService from ShowManager."""
        try:
            from src.features.ma3.application.ma3_communication_service import MA3CommunicationService
            
            # Try to get from facade or create new
            # In real usage, this would come from the ShowManager block
            result = self._facade.describe_block(self._show_manager_block_id)
            if result.success and result.data:
                block = result.data
                ma3_ip = block.metadata.get('ma3_ip', '127.0.0.1')
                ma3_port = block.metadata.get('ma3_port', 9001)
                
                service = MA3CommunicationService(
                    listen_port=9000,
                    send_address=ma3_ip,
                    send_port=ma3_port
                )
                return service
            
        except Exception as e:
            Log.warning(f"SyncLayerCommand: Failed to get MA3 service: {e}")
        
        return None
    
    def _send_add_event_osc(
        self,
        ma3_service,
        timecode_no: int,
        track_group: int,
        track: int,
        event_index: int,
        time: float,
        event_type: str = "cmd",
        name: str = "",
        metadata: dict = None
    ):
        """Send OSC command to add event in MA3."""
        import json
        
        message = (
            f"add_event|"
            f"tc={timecode_no}|"
            f"tg={track_group}|"
            f"tr={track}|"
            f"idx={event_index}|"
            f"time={time}|"
            f"type={event_type}|"
            f"name={name}"
        )
        
        if metadata:
            message += f"|meta={json.dumps(metadata)}"
        
        ma3_service.send_message(message)
    
    def _send_clear_track_osc(
        self,
        ma3_service,
        timecode_no: int,
        track_group: int,
        track: int
    ):
        """Send OSC command to clear a track in MA3."""
        message = (
            f"clear_track|"
            f"tc={timecode_no}|"
            f"tg={track_group}|"
            f"tr={track}"
        )
        ma3_service.send_message(message)
    
    def undo(self):
        """Reverse the sync operation."""
        if self._direction == "ma3_to_editor" and self._original_editor_events:
            # Restore original Editor events
            from src.features.blocks.application.editor_api import create_editor_api
            
            try:
                editor_api = create_editor_api(self._facade, self._editor_block_id)
                
                # Clear the synced events
                editor_api.clear_layer_events(self._editor_layer_name)
                
                # Restore original events
                if self._original_editor_events:
                    editor_api.add_events(self._original_editor_events, source="undo")
                
                Log.info(f"SyncLayerCommand: Restored {len(self._original_editor_events)} original events")
                
            except Exception as e:
                self._log_error(f"Failed to undo sync: {e}")
        
        # Note: Undo for editor_to_ma3 would require sending delete commands to MA3
        # This is more complex and may not always be possible
