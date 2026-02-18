"""
Editor block panel.

Provides a DAW-style audio editor with timeline visualization.
Features:
- Timeline widget for event visualization and editing
- Audio playback synchronized with timeline
- Event editing (move, resize, delete, create)
- Waveform display integration (future)
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QPushButton, QMessageBox, QFileDialog, QSplitter,
    QFrame, QCheckBox, QScrollArea
)
from PyQt6.QtCore import QTimer

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from ui.qt_gui.style_factory import StyleFactory
from ui.qt_gui.widgets.timeline.core import TimelineWidget
from ui.qt_gui.widgets.timeline.playback import SimpleAudioPlayer
from ui.qt_gui.widgets.timeline.types import (
    EventMoveResult, EventResizeResult, EventCreateResult, EventDeleteResult, EventSliceResult
)
from src.utils.message import Log

from src.features.blocks.domain import Block
from src.shared.domain.entities import EventDataItem
from src.shared.domain.entities import AudioDataItem
from src.shared.domain.entities.layer_order import LayerKey
from src.application.commands import (
    AddEventToDataItemCommand,
    UpdateEventInDataItemCommand,
    BatchUpdateDataItemEventsCommand,
)
from src.shared.application.services.event_filter_manager import EventFilterManager
from ui.qt_gui.dialogs.event_filter_dialog import EventFilterDialog

@register_block_panel("Editor")
class EditorPanel(BlockPanelBase):
    """
    Panel for Editor block - audio editor with timeline visualization.
    
    Provides:
    - Interactive timeline with events from connected blocks
    - Audio playback synchronized with playhead
    - Event editing capabilities
    - Audio file loading and playback
    """
    
    def __init__(self, block_id: str, facade, parent=None):
        """Initialize EditorPanel with dock state change protection."""
        super().__init__(block_id, facade, parent)
        
        
        # Connect to dock state changes to protect Qt Multimedia from crashes
        # during widget reparenting that happens when docking/undocking
        self.topLevelChanged.connect(self._on_dock_state_changing)

        # Layer order services
        self._layer_order_service = getattr(self.facade, "layer_order_service", None)
        self._layer_group_order_service = getattr(self.facade, "layer_group_order_service", None)
        
        # Guard flag to prevent recursive selection clearing loops
        self._clearing_selection = False
    
    def _subscribe_to_events(self):
        """Subscribe to block events including direct visual updates."""
        # Call parent subscription (BlockUpdated)
        super()._subscribe_to_events()
        
        # Subscribe to EventVisualUpdate for DIRECT MA3 sync updates
        # This bypasses the full reload cycle for fast visual updates
        self.facade.event_bus.subscribe("EventVisualUpdate", self._on_event_visual_update)
    
    def _on_event_visual_update(self, event):
        """
        Handle EventVisualUpdate - DIRECT visual update without full reload.
        
        This is the fast path for MA3 sync event moves. Bypasses the full
        BlockUpdated -> reload cycle by updating the TimelineWidget directly.
        
        If the event isn't found in the timeline (e.g., panel just reopened),
        falls back to reloading EventDataItems for the updated item.
        """
        from src.utils.message import Log
        
        try:
            data = event.data if hasattr(event, 'data') else {}
            block_id = data.get('block_id')
            event_id = data.get('event_id')
            new_time = data.get('new_time')
            
            # Only handle events for this Editor block
            if block_id != self.block_id:
                return
            
            if not event_id or new_time is None:
                Log.warning(f"EditorPanel: Invalid EventVisualUpdate data: {data}")
                return
            
            # DIRECT visual update - no full reload!
            if hasattr(self, 'timeline_widget') and self.timeline_widget:
                result = self.timeline_widget.update_event(event_id, start_time=float(new_time))
                if result:
                    Log.info(f"EditorPanel: DIRECT visual update SUCCESS - event {event_id} -> {new_time}s")
                else:
                    # Event not in timeline scene - do a targeted reload of EventDataItems
                    Log.debug(f"EditorPanel: Event {event_id} not in scene, reloading EventDataItems")
                    self._reload_event_data_items_in_place()
        except Exception as e:
            Log.warning(f"EditorPanel: EventVisualUpdate failed: {e}")
    
    def _reload_event_data_items_in_place(self):
        """
        Reload EventDataItems from repository and update timeline.
        
        This is a lighter-weight alternative to full _load_owned_data().
        Used when direct visual updates fail because the event isn't in the scene.
        """
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        
        try:
            if not hasattr(self, '_loaded_event_items'):
                self._loaded_event_items = []
            
            # Get fresh EventDataItems from repo
            items = self.facade.data_item_repo.list_by_block(self.block_id)
            event_items = [item for item in items if isinstance(item, EventDataItem)]
            
            if not event_items:
                return
            
            # Update cache (dict keyed by data_item_id to stay consistent)
            self._loaded_event_items = {item.id: item for item in event_items}
            
            # Update timeline
            if hasattr(self, 'timeline_widget') and self.timeline_widget:
                self.timeline_widget.set_events_from_data_items(event_items)
                Log.debug(f"EditorPanel: Reloaded {len(event_items)} EventDataItems in place")
        except Exception as e:
            Log.warning(f"EditorPanel: Failed to reload EventDataItems: {e}")
    
    # =========================================================================
    # Layer Classification Helpers
    # =========================================================================
    
    def _is_sync_layer_item(self, item: EventDataItem) -> bool:
        """
        Check if EventDataItem is a MA3 sync layer.
        
        Sync layers are identified by:
        - Metadata flag: _synced_from_ma3 = True
        - Name pattern: contains "ma3_sync" (case-insensitive)
        
        Args:
            item: EventDataItem to check
            
        Returns:
            True if item is a sync layer, False otherwise
        """
        if not isinstance(item, EventDataItem):
            return False
        
        # Check explicit metadata flags
        if item.metadata.get("_synced_from_ma3") is True:
            return True
        if item.metadata.get("_synced_to_ma3") is True:
            return True
        if item.metadata.get("_sync_type") in {"showmanager_layer", "ez_layer"}:
            return True
        if item.metadata.get("sync_type") in {"showmanager_layer", "ez_layer"}:
            return True

        # Check source-based metadata
        if item.metadata.get("source") in {"ma3", "ma3_sync"}:
            return True
        
        # Check name pattern (legacy fallback)
        if item.name and "ma3_sync" in item.name.lower():
            return True
        
        return False
    
    def _is_ez_layer_item(self, item: EventDataItem) -> bool:
        """
        Check if EventDataItem is an EchoZero layer (from upstream blocks).
        
        EZ layers are all EventDataItems that are NOT sync layers.
        
        Args:
            item: EventDataItem to check
            
        Returns:
            True if item is an EZ layer, False otherwise
        """
        return not self._is_sync_layer_item(item)
    
    def _get_preserved_sync_item_ids(self) -> List[str]:
        """
        Get list of sync layer EventDataItem IDs to preserve.
        
        Returns:
            List of EventDataItem IDs that are sync layers
        """
        if not hasattr(self.facade, 'data_item_repo') or not self.facade.data_item_repo:
            return []
        
        try:
            owned_items = self.facade.data_item_repo.list_by_block(self.block_id)
            sync_item_ids = [
                item.id for item in owned_items
                if isinstance(item, EventDataItem) 
                and item.metadata.get("output_port") == "events"
                and self._is_sync_layer_item(item)
            ]
            if sync_item_ids:
                Log.debug(f"EditorPanel: Found {len(sync_item_ids)} sync layer item(s) to preserve")
            return sync_item_ids
        except Exception as e:
            Log.warning(f"EditorPanel: Failed to get preserved sync item IDs: {e}")
            return []
    
    # =========================================================================
    # Layer-by-Layer Operations
    # =========================================================================
    
    def _load_layer_from_item(self, item, event_layer) -> str:
        """
        Load a single EventLayer from an EventDataItem into the timeline.
        
        This is the standard layer loading operation using the handler system.
        
        Args:
            item: EventDataItem containing the layer
            event_layer: EventLayer to load
            
        Returns:
            Layer ID of the created layer
        """
        from src.utils.message import Log
        from ui.qt_gui.widgets.timeline.types import TimelineLayer, TimelineEvent
        
        # Determine if this is a sync layer (MA3)
        is_synced = self._is_sync_layer_item(item)
        
        # Extract MA3 metadata if sync layer
        ma3_track_coord = None
        show_manager_block_id = None
        if is_synced:
            # MA3 sync items often store metadata about the track
            if item.metadata:
                ma3_track_coord = (
                    item.metadata.get('_ma3_track_coord')
                    or item.metadata.get('ma3_track_coord')
                    or item.metadata.get('ma3_coord')
                )
                show_manager_block_id = (
                    item.metadata.get('_show_manager_block_id')
                    or item.metadata.get('show_manager_block_id')
                )

            # Fallback: pull from event metadata if not present at item level
            if not ma3_track_coord or not show_manager_block_id:
                for evt in event_layer.get_events():
                    evt_meta = getattr(evt, 'metadata', {}) or {}
                    if not ma3_track_coord:
                        ma3_track_coord = (
                            evt_meta.get('_ma3_track_coord')
                            or evt_meta.get('ma3_track_coord')
                            or evt_meta.get('ma3_coord')
                        )
                    if not show_manager_block_id:
                        show_manager_block_id = (
                            evt_meta.get('_show_manager_block_id')
                            or evt_meta.get('show_manager_block_id')
                        )
                    if ma3_track_coord and show_manager_block_id:
                        break
        
        # Create layer config
        group_id, group_name = self._get_group_identity_for_item(item)
        layer_config = TimelineLayer(
            id=f"layer_{item.id}_{event_layer.id}",
            name=event_layer.name,
            index=-1,  # Append
            height=40.0,
            color=None,  # Auto-assign
            group_id=group_id,
            group_name=group_name,
            is_synced=is_synced,
            show_manager_block_id=show_manager_block_id,
            ma3_track_coord=ma3_track_coord,
        )
        
        # Convert EventLayer events to TimelineEvents
        # SINGLE SOURCE OF TRUTH: Use TimelineEvent.from_event() for conversion
        events = []
        for i, evt in enumerate(event_layer.get_events()):
            # SINGLE SOURCE OF TRUTH: Always use domain Event's UUID
            # Domain Events MUST have stable UUIDs - fail hard if missing
            event_id = getattr(evt, 'id', None)
            if not event_id:
                from src.utils.message import Log
                error_msg = (
                    f"EditorPanel._load_layer_from_item(): "
                    f"Event at index {i} in layer '{event_layer.name}' has no ID. "
                    f"All events must have stable UUIDs. Event: time={evt.time}, "
                    f"classification={evt.classification}, duration={evt.duration}"
                )
                Log.error(error_msg)
                raise ValueError(error_msg)
            
            # Ensure UUID is a string (should always be, but defensive)
            event_id = str(event_id)
            
            # Build user_data with source tracking
            user_data = {
                '_source_item_id': item.id,
                '_source_item_name': item.name,
                '_source_layer_id': event_layer.id,
                '_source_layer_name': event_layer.name,
            }
            
            # SINGLE SOURCE OF TRUTH: Use TimelineEvent.from_event() for conversion
            # This handles all normalization (duration, render_as_marker, metadata mapping)
            # TimelineEvent.from_event() already uses event.id as the TimelineEvent.id
            timeline_event = TimelineEvent.from_event(
                event=evt,
                layer_id=None,  # Will be set by handler
                user_data=user_data,
            )
            
            # Verify ID consistency (TimelineEvent.from_event() should use event.id)
            if timeline_event.id != event_id:
                from src.utils.message import Log
                Log.warning(
                    f"EditorPanel: TimelineEvent.id ({timeline_event.id}) != event.id ({event_id}). "
                    f"TimelineEvent.from_event() should preserve event.id. Overriding to match."
                )
                timeline_event.id = event_id
            
            events.append(timeline_event)
        
        # Load via TimelineWidget's standardized method
        layer_id = self.timeline_widget.load_layer(layer_config, events)
        
        Log.debug(
            f"EditorPanel._load_layer_from_item: Loaded layer '{event_layer.name}' "
            f"({len(events)} events, synced={is_synced})"
        )
        
        return layer_id
    
    def reload_sync_layers(self) -> int:
        """
        Reload only sync layers from their data sources.
        
        This is useful after MA3 changes - only reloads the affected
        sync layers without touching regular layers.
        
        Returns:
            Number of sync layers reloaded
        """
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        
        if not hasattr(self, 'timeline_widget') or not self.timeline_widget:
            return 0
        
        # Get sync layers
        sync_layers = self.timeline_widget.get_synced_layers()
        if not sync_layers:
            Log.debug("EditorPanel.reload_sync_layers: No sync layers to reload")
            return 0
        
        # Get fresh sync EventDataItems from repo
        items = self.facade.data_item_repo.list_by_block(self.block_id)
        sync_items = [
            item for item in items
            if isinstance(item, EventDataItem) and self._is_sync_layer_item(item)
        ]
        
        reloaded = 0
        for layer in sync_layers:
            try:
                # Clear and reload this layer
                self.timeline_widget.reload_layer(layer.id)
                
                # Find matching EventDataItem and reload events
                for item in sync_items:
                    for event_layer in getattr(item, '_layers', []):
                        if event_layer.name == layer.name:
                            # Re-load events for this layer
                            self._reload_events_for_layer(layer, item, event_layer)
                            break
                
                reloaded += 1
                Log.debug(f"EditorPanel.reload_sync_layers: Reloaded sync layer '{layer.name}'")
            except Exception as e:
                Log.warning(f"EditorPanel.reload_sync_layers: Failed to reload '{layer.name}': {e}")
        
        Log.info(f"EditorPanel.reload_sync_layers: Reloaded {reloaded}/{len(sync_layers)} sync layers")
        return reloaded
    
    def _reload_events_for_layer(self, layer, item, event_layer) -> int:
        """
        Reload events for a specific layer from an EventDataItem.
        
        Args:
            layer: TimelineLayer to reload
            item: EventDataItem containing the events
            event_layer: EventLayer with the events
            
        Returns:
            Number of events loaded
        """
        from src.utils.message import Log
        from ui.qt_gui.widgets.timeline.types import TimelineEvent
        from ui.qt_gui.widgets.timeline.handlers import get_handler_for_layer
        
        # Get handler for this layer
        handler = get_handler_for_layer(
            layer, 
            self.timeline_widget._scene, 
            self.timeline_widget._layer_manager
        )
        
        # Convert events
        events = []
        for i, evt in enumerate(event_layer.get_events()):
            # SINGLE SOURCE OF TRUTH: Always use domain Event's UUID
            # Domain Events MUST have stable UUIDs - fail hard if missing
            event_id = getattr(evt, 'id', None)
            if not event_id:
                from src.utils.message import Log
                error_msg = (
                    f"EditorPanel._reload_events_for_layer(): "
                    f"Event at index {i} in layer '{event_layer.name}' has no ID. "
                    f"All events must have stable UUIDs. Event: time={evt.time}, "
                    f"classification={evt.classification}, duration={evt.duration}"
                )
                Log.error(error_msg)
                raise ValueError(error_msg)
            
            # Ensure UUID is a string (should always be, but defensive)
            event_id = str(event_id)
            
            user_data = dict(evt.metadata) if evt.metadata else {}
            user_data['_source_item_id'] = item.id
            user_data['_source_item_name'] = item.name
            
            # SINGLE SOURCE OF TRUTH: Use TimelineEvent.from_event() for conversion
            # TimelineEvent.from_event() already uses event.id as the TimelineEvent.id
            timeline_event = TimelineEvent.from_event(
                event=evt,
                layer_id=layer.id,
                user_data=user_data,
            )
            
            # Verify ID consistency (TimelineEvent.from_event() should use event.id)
            if timeline_event.id != event_id:
                from src.utils.message import Log
                Log.warning(
                    f"EditorPanel: TimelineEvent.id ({timeline_event.id}) != event.id ({event_id}). "
                    f"TimelineEvent.from_event() should preserve event.id. Overriding to match."
                )
                timeline_event.id = event_id
            
            events.append(timeline_event)
        
        # Load events via handler
        loaded = handler.load_events(layer, events)
        Log.debug(f"EditorPanel._reload_events_for_layer: Loaded {loaded} events into '{layer.name}'")
        
        return loaded
    
    def reload_layer_by_name(self, layer_name: str) -> bool:
        """
        Reload a specific layer by name.
        
        Args:
            layer_name: Name of the layer to reload
            
        Returns:
            True if layer was found and reloaded
        """
        from src.utils.message import Log
        
        if not hasattr(self, 'timeline_widget') or not self.timeline_widget:
            return False
        
        # Find layer by name
        layer = self.timeline_widget._layer_manager.get_layer_by_name(layer_name)
        if not layer:
            Log.warning(f"EditorPanel.reload_layer_by_name: Layer '{layer_name}' not found")
            return False
        
        return self.timeline_widget.reload_layer(layer.id)
    
    def get_layer_validation_issues(self) -> dict:
        """
        Validate all layers and return any issues.
        
        Returns:
            Dict of layer_name -> list of issues
        """
        issues = {}
        
        if not hasattr(self, 'timeline_widget') or not self.timeline_widget:
            return issues
        
        for layer in self.timeline_widget.get_layers():
            layer_issues = self.timeline_widget.validate_layer(layer.id)
            if layer_issues:
                issues[layer.name] = layer_issues
        
        return issues
    
    def _on_dock_state_changing(self, floating: bool):
        """
        Handle dock state changes (floating/docked).
        
        CRITICAL: Stop audio playback before widget reparenting to prevent
        Qt Multimedia crashes. Qt Multimedia's QMediaPlayer can crash when
        the parent widget hierarchy changes while playing.
        """
        
        if hasattr(self, '_audio_player') and self._audio_player:
            try:
                # Stop playback to prevent crash during reparenting
                was_playing = self._audio_player.is_playing()
                if was_playing:
                    self._audio_player.pause()
                    Log.debug("EditorPanel: Paused audio during dock state change")
            except Exception as e:
                Log.warning(f"EditorPanel: Error pausing audio during dock change: {e}")
    
    def create_content_widget(self) -> QWidget:
        """Create Editor-specific UI with timeline and controls"""
        # Event persistence tracking (for looking up old values during edits)
        self._loaded_event_items: Dict[str, EventDataItem] = {}
        self._loaded_audio_items: Dict[str, AudioDataItem] = {}
        self._auto_loaded = False  # Track if we've auto-loaded events
        
        # Note: Batch handling is now done by MovementController in the timeline widget
        
        # Audio player
        self._audio_player: Optional[SimpleAudioPlayer] = None
        self._current_audio_path: Optional[str] = None
        
        # Stale data tracking
        self._has_stale_audio: bool = False
        self._stale_audio_errors: List[str] = []
        
        # Event filter manager (part of "processes" improvement area)
        self._event_filter_manager = EventFilterManager()
        # Initialize with block getter function
        def get_block(block_id: str) -> Optional[Block]:
            result = self.facade.describe_block(block_id)
            return result.data if result.success else None
        self._event_filter_manager.set_block_getter(get_block)
        
        from PyQt6.QtWidgets import QSizePolicy
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(widget)
        layout.setSpacing(0)  # No spacing between elements
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top controls area
        controls_widget = self._create_controls()
        layout.addWidget(controls_widget)
        
        # Warning banner for stale data (hidden by default)
        self._warning_banner = self._create_warning_banner()
        layout.addWidget(self._warning_banner)
        self._warning_banner.setVisible(False)
        
        # Main timeline widget (with preferences repo for settings persistence)
        preferences_repo = getattr(self.facade, 'preferences_repo', None)
        self.timeline_widget = TimelineWidget(preferences_repo=preferences_repo)
        # Store Editor block_id and facade in TimelineWidget for layer deletion
        self.timeline_widget._editor_block_id = self.block_id
        self.timeline_widget._facade = self.facade
        # Set command_bus on scene for undoable operations
        if hasattr(self.timeline_widget, '_scene') and self.facade.command_bus:
            self.timeline_widget._scene.set_command_bus(self.facade.command_bus)
        self.timeline_widget.setMinimumHeight(350)
        
        # Clear any stale selection when timeline widget is first created
        # This prevents stale selections from previous sessions
        if hasattr(self.timeline_widget, '_scene'):
            if hasattr(self.timeline_widget._scene, 'deselect_all'):
                self.timeline_widget._scene.deselect_all()
            elif hasattr(self.timeline_widget._scene, 'clearSelection'):
                self.timeline_widget._scene.clearSelection()
        if hasattr(self.timeline_widget, '_event_inspector'):
            inspector = self.timeline_widget._event_inspector
            if hasattr(inspector, 'update_selection'):
                inspector.update_selection([])  # Clear selection
        # Ensure timeline widget expands to fill available space and its bottom stays aligned
        self.timeline_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set data_item_repo on timeline widget for direct audio item lookup by ID.
        # No callback. No fallback. Events reference Editor-owned audio items by ID,
        # and repo.get(audio_id) is the single lookup mechanism.
        if hasattr(self.facade, 'data_item_repo') and self.facade.data_item_repo:
            self.timeline_widget.set_data_item_repo(self.facade.data_item_repo)
        
        # Set event update callback for EventInspector (for updating render_as_marker)
        # Track if we're currently clearing selection to prevent recursive loops
        self._clearing_selection = False
        
        def event_update_callback(event_id: str, metadata: Dict[str, Any]) -> bool:
            """Update event metadata via EditorAPI."""
            try:
                # Guard: Prevent recursive calls during selection clearing
                if getattr(self, '_clearing_selection', False):
                    return False
                
                
                # Validate event exists in scene before attempting update
                if hasattr(self, 'timeline_widget') and self.timeline_widget:
                    scene = getattr(self.timeline_widget, '_scene', None)
                    if scene and hasattr(scene, '_event_items'):
                        if event_id not in scene._event_items:
                            # Don't clear selection here - just skip the update.
                            # Selection validation happens in _on_selection_changed() which will filter out invalid events.
                            # Clearing selection here interferes with normal event selection.
                            Log.debug(f"EditorPanel: Event {event_id} not found in scene - skipping update (event may have been removed)")
                            return False
                
                # Parse event_id to get data_item_id and event_index
                parsed = self._parse_event_id(event_id)
                if not parsed:
                    Log.warning(f"EditorPanel: Could not parse event_id {event_id}")
                    return False
                
                data_item_id, event_index = parsed
                
                # Get EditorAPI
                editor_api = self._get_editor_api_for_sync()
                if not editor_api:
                    Log.warning("EditorPanel: No EditorAPI available for event update")
                    return False
                
                # Get full current metadata from database to merge with incoming changes
                # This ensures we don't lose other metadata fields when updating
                current_event = editor_api.get_event(event_id)
                if current_event and current_event.metadata:
                    # Merge incoming metadata into full current metadata
                    merged_metadata = dict(current_event.metadata)
                    merged_metadata.update(metadata)
                    metadata = merged_metadata
                
                if not self.facade.command_bus:
                    Log.warning("EditorPanel: Command bus unavailable for event metadata update")
                    return False

                # Route metadata updates through command bus so undo/redo works.
                self.facade.command_bus.execute(
                    UpdateEventInDataItemCommand(
                        self.facade,
                        data_item_id,
                        event_index,
                        new_metadata=metadata
                    )
                )

                # Update timeline visualization immediately; data layer update is undoable.
                if hasattr(self, 'timeline_widget') and self.timeline_widget:
                    self.timeline_widget.update_event(event_id, user_data=metadata)

                return True
            except Exception as e:
                Log.error(f"EditorPanel: Error updating event {event_id}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        self.timeline_widget.set_event_update_callback(event_update_callback)
        
        # Connect signals for event editing (new typed API)
        self.timeline_widget.selection_changed.connect(self._on_selection_changed)
        self.timeline_widget.events_moved.connect(self._on_events_moved)
        self.timeline_widget.events_resized.connect(self._on_events_resized)
        # Only connect batch deletion handler - single events are wrapped in batch
        self.timeline_widget.events_deleted.connect(self._on_events_deleted_batch)
        self.timeline_widget.event_created.connect(self._on_event_created)
        self.timeline_widget.event_sliced.connect(self._on_event_sliced)
        
        # Connect status messages from timeline scene
        # Access scene through the timeline widget's internal structure
        if hasattr(self.timeline_widget, 'scene') and self.timeline_widget.scene():
            scene = self.timeline_widget.scene()
            if hasattr(scene, 'status_message'):
                scene.status_message.connect(self._on_timeline_status_message)
        self.timeline_widget.position_changed.connect(self._on_position_changed)
        self.timeline_widget.playback_state_changed.connect(self._on_playback_state_changed)
        
        # Connect layer change signals to auto-save layer state
        if hasattr(self.timeline_widget, '_layer_manager'):
            self.timeline_widget._layer_manager.layer_updated.connect(self._on_layer_updated)
            # Also connect to layers_changed to update order when layers are reordered
            self.timeline_widget._layer_manager.layers_changed.connect(self._on_layers_changed)
        
        # Add timeline widget with stretch factor 1 so it expands to fill space
        # The stretch ensures it fills available space between top controls and status bar
        layout.addWidget(self.timeline_widget, 1)
        
        # Status bar at bottom - always stays at the bottom, preventing timeline from going behind
        status_frame = self._create_status_bar()
        # Ensure status bar doesn't shrink and stays visible
        status_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(status_frame, 0)  # Stretch factor 0 ensures it stays at bottom
        
        # Set minimum height on parent container to respect timeline widget's minimum
        # This prevents the container from shrinking below the sum of fixed component heights
        # When timeline reaches its min height, parent container should also respect that constraint
        controls_height = controls_widget.minimumHeight()
        timeline_min_height = self.timeline_widget.minimumHeight()
        status_height = status_frame.minimumHeight()
        widget.setMinimumHeight(controls_height + timeline_min_height + status_height)
        
        return widget
    
    def _apply_local_styles(self):
        """Re-apply variant/context-specific styles on theme change.
        
        The global QApplication stylesheet handles default widget styling.
        This method refreshes only the local overrides that use variant
        colors or computed tokens (toolbar, status bar, filter button,
        warning banner, event source elements).
        """
        # Control bar / toolbar
        if hasattr(self, '_controls_frame') and self._controls_frame:
            self._controls_frame.setStyleSheet(StyleFactory.toolbar())
        
        # Status bar
        if hasattr(self, '_status_frame') and self._status_frame:
            self._status_frame.setStyleSheet(StyleFactory.status_bar())
        
        # Status labels
        label_style = f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;"
        for lbl in [
            getattr(self, 'event_count_label', None),
            getattr(self, 'layer_count_label', None),
            getattr(self, 'duration_label', None),
        ]:
            if lbl:
                lbl.setStyleSheet(label_style)
        
        if hasattr(self, 'audio_label') and self.audio_label:
            self.audio_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        if hasattr(self, 'info_label') and self.info_label:
            self.info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-style: italic;")
    
    def _create_controls(self) -> QWidget:
        """Create top control bar"""
        controls = QFrame()
        self._controls_frame = controls  # Store for theme refresh
        controls.setFixedHeight(44)  # Fixed height for top controls
        controls.setStyleSheet(StyleFactory.toolbar())
        
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        layout.setSpacing(Spacing.MD)
        
        # Event filter button (part of "processes" improvement area)
        filter_button = QPushButton("Filter Events")
        filter_button.setToolTip("Configure event filters for visualization and processing")
        filter_button.clicked.connect(self._on_open_filter_dialog)
        filter_button.setStyleSheet(StyleFactory.button())
        layout.addWidget(filter_button)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {Colors.BORDER.name()};")
        layout.addWidget(sep)
        
        # Audio file label
        self.audio_label = QLabel("No audio loaded")
        self.audio_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(self.audio_label)
        
        layout.addStretch()
        
        # Info label
        self.info_label = QLabel("Connect audio and events inputs, then pull input data")
        self.info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-style: italic;")
        layout.addWidget(self.info_label)
        
        return controls

    
    def _create_status_bar(self) -> QWidget:
        """Create bottom status bar"""
        status = QFrame()
        self._status_frame = status  # Store for theme refresh
        status.setFixedHeight(28)
        status.setStyleSheet(StyleFactory.status_bar())
        
        layout = QHBoxLayout(status)
        layout.setContentsMargins(Spacing.MD, 2, Spacing.MD, 2)
        layout.setSpacing(Spacing.LG)
        
        # Event count
        self.event_count_label = QLabel("Events: 0")
        self.event_count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        layout.addWidget(self.event_count_label)
        
        # Layer count
        self.layer_count_label = QLabel("Layers: 0")
        self.layer_count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        layout.addWidget(self.layer_count_label)
        
        # Duration
        self.duration_label = QLabel("Duration: 0.0s")
        self.duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        layout.addWidget(self.duration_label)
        
        layout.addStretch()
        
        # Dirty indicator
        
        return status
    
    def _create_warning_banner(self) -> QWidget:
        """Create warning banner for stale data"""
        banner = QFrame()
        warn_bg = Colors.STATUS_WARNING.lighter(180).name()
        warn_border = Colors.STATUS_WARNING.name()
        warn_fg = Colors.STATUS_WARNING.darker(200).name()
        banner.setStyleSheet(f"""
            QFrame {{
                background-color: {warn_bg};
                border: 1px solid {warn_border};
                border-radius: {border_radius(4)};
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        layout.setSpacing(Spacing.MD)
        
        # Warning icon
        icon_label = QLabel("⚠")
        icon_label.setStyleSheet(f"color: {warn_fg}; font-size: 16px; font-weight: bold;")
        layout.addWidget(icon_label)
        
        # Warning message
        self._warning_message = QLabel()
        self._warning_message.setWordWrap(True)
        self._warning_message.setStyleSheet(f"color: {warn_fg}; font-size: 12px;")
        layout.addWidget(self._warning_message, 1)
        
        # Action buttons
        self._clear_stale_btn = QPushButton("Clear Stale Events")
        self._clear_stale_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {warn_border};
                color: {Colors.BG_DARK.name()};
                border: none;
                border-radius: {border_radius(3)};
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Colors.STATUS_WARNING.darker(110).name()};
            }}
        """)
        self._clear_stale_btn.clicked.connect(self._on_clear_stale_data)
        layout.addWidget(self._clear_stale_btn)
        
        # Dismiss button
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {warn_fg};
                border: 1px solid {warn_fg};
                border-radius: {border_radius(3)};
                padding: 6px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {Colors.STATUS_WARNING.lighter(300).name()};
            }}
        """)
        dismiss_btn.clicked.connect(lambda: self._warning_banner.setVisible(False))
        layout.addWidget(dismiss_btn)
        
        return banner
    
    def _update_status_labels(self):
        """Update status bar labels (event count, layer count, duration) from current state."""
        if not hasattr(self, 'timeline_widget') or not self.timeline_widget:
            return
        total_events = sum(
            len(item.get_events())
            for item in self._loaded_event_items.values()
        ) if self._loaded_event_items else 0
        layers = self.timeline_widget.get_layers()
        duration = self.timeline_widget.get_duration()
        self.event_count_label.setText(f"Events: {total_events}")
        self.layer_count_label.setText(f"Layers: {len(layers)}")
        self.duration_label.setText(f"Duration: {duration:.1f}s")
    
    def _on_timeline_status_message(self, message: str, is_error: bool):
        """Handle status messages from timeline scene"""
        self.set_status_message(message, error=is_error)
    
    def _update_events_layer_name(self, old_layer_name: str, new_layer_name: str):
        """
        Update all events in owned EventDataItems that reference old_layer_name.
        
        Called when a layer is renamed to keep event metadata in sync.
        
        Args:
            old_layer_name: The old layer name to find and replace
            new_layer_name: The new layer name to set
        """
        if not hasattr(self, 'facade') or not self.facade:
            return
        
        try:
            from src.shared.domain.entities import EventDataItem
            from src.application.commands import CommandBus, BatchUpdateDataItemEventsCommand
            
            # Get all owned EventDataItems for this block
            owned_items = self.facade.data_item_repo.list_by_block(self.block_id)
            event_items = [item for item in owned_items 
                          if isinstance(item, EventDataItem) and item.metadata.get("output_port") == "events"]
            
            if not event_items:
                return
            
            # Find all events that need updating
            updates_by_item: Dict[str, List[Dict[str, Any]]] = {}
            
            for item in event_items:
                events = item.get_events()
                item_updates = []
                
                for idx, event in enumerate(events):
                    if event.metadata and event.metadata.get('_visual_layer_name') == old_layer_name:
                        item_updates.append({
                            'event_index': idx,
                            'metadata': {'_visual_layer_name': new_layer_name}
                        })
                
                if item_updates:
                    updates_by_item[item.id] = item_updates
            
            # Apply updates via commands
            if updates_by_item:
                total_updates = sum(len(updates) for updates in updates_by_item.values())
                use_macro = len(updates_by_item) > 1
                
                if use_macro:
                    self.facade.command_bus.begin_macro(f"Update {total_updates} Event{'s' if total_updates > 1 else ''} Layer Name")
                
                for data_item_id, updates in updates_by_item.items():
                    self.facade.command_bus.execute(BatchUpdateDataItemEventsCommand(
                        self.facade,
                        data_item_id,
                        updates,
                        description=f"Update Layer Name: {old_layer_name} -> {new_layer_name}"
                    ))
                
                if use_macro:
                    self.facade.command_bus.end_macro()
                
                # Touch local state to reflect changes
                self._touch_local_state_db(reason="layer_renamed")
                
                Log.info(f"EditorPanel: Updated {total_updates} events from layer '{old_layer_name}' to '{new_layer_name}'")
        
        except Exception as e:
            Log.warning(f"EditorPanel: Failed to update events on layer rename: {e}")
    
    def _on_layer_updated(self, layer_id: str):
        """Handle layer property changes - auto-save layer state."""
        # Debounce saves - use a timer to avoid saving on every tiny change
        if not hasattr(self, '_layer_save_timer'):
            from PyQt6.QtCore import QTimer
            self._layer_save_timer = QTimer(self)
            self._layer_save_timer.setSingleShot(True)
            self._layer_save_timer.setInterval(500)  # 500ms debounce
            self._layer_save_timer.timeout.connect(self._save_layer_state)

        self._layer_save_timer.start()
    
    def _on_layers_changed(self):
        """Handle layer structure changes (add/delete/reorder) - update stored order."""
        # Persist layer order when layers are changed (reordered, added, deleted)
        layer_order_service = getattr(self, "_layer_order_service", None)
        if getattr(self, "_pull_data_active", False):
            return
        if not self.block or not self.facade:
            return
        result = self.facade.describe_block(self.block.id)
        if not result.success or not result.data:
            return  # Block no longer exists (e.g. deleted), skip save to avoid FK error
        if hasattr(self, 'timeline_widget'):
            layer_keys = self._get_current_layer_order_keys()
            if layer_keys and all(key.group_name for key in layer_keys):
                if layer_order_service:
                    layer_order_service.save_order(self.block.id, layer_keys)
            layer_group_service = getattr(self, "_layer_group_order_service", None)
            if layer_group_service:
                layer_manager = getattr(self.timeline_widget, "_layer_manager", None)
                if layer_manager:
                    group_order, layers_by_group = layer_group_service.build_from_layers(
                        layer_manager.get_all_layers()
                    )
                    layer_group_service.save_order(self.block.id, group_order, layers_by_group)
        self._save_layer_state()
    
    def refresh(self):
        """Update UI with current block data.
        
        SIMPLE MODEL:
        - Load existing owned DataItems (what the Editor has already created/edited)
        - Load audio reference from local state
        - Update labels
        
        Recreation of owned items ONLY happens via Pull Data (_on_block_updated).
        This prevents infinite duplication on panel reopen.
        """
        if not self.block:
            return

        # Load existing owned data if not already loaded
        if not self._loaded_event_items:
            if hasattr(self.facade, 'data_item_repo') and self.facade.data_item_repo:
                owned_outputs = self.facade.data_item_repo.list_by_block(self.block_id)
                
                if owned_outputs:
                    # Use the same layer-build path as Pull Data (local-only mode)
                    self._load_owned_data(skip_ui_state_restore=True)

        # Load audio from local state (audio is a reference, not owned)
        if not self._loaded_audio_items:
            try:
                self._load_audio_from_local_state()
            except Exception as e:
                Log.debug(f"EditorPanel: Failed to load audio from local state: {e}")
        
        # Update info label based on loaded data
        if self._loaded_event_items:
            total_events = sum(len(item.get_events()) for item in self._loaded_event_items.values())
            self.info_label.setText(f"Loaded: {total_events} events")
        else:
            self.info_label.setText("No data - pull input data to initialize Editor state")
        
        self.set_status_message("Ready")

    def _load_events_from_local_state(self):
        """
        Load referenced events into the Editor UI from persisted local state.
        
        This is the standard Pull Data behavior - load events from references stored in local state.
        Events are displayed as-is, layer by layer, from the EventDataItems referenced in local state.
        
        IMPORTANT: Also includes sync layer items (MA3 synced) that are owned by this Editor block.
        Sync layers are preserved during Pull Data - they're not in local state references but are
        owned by the Editor block and should always be included.
        
        No owned copies are created during Pull Data (those are only created during execution).
        """
        if not self.block:
            return
        if not hasattr(self.facade, "block_local_state_repo") or not self.facade.block_local_state_repo:
            return
        if not hasattr(self.facade, "data_item_repo") or not self.facade.data_item_repo:
            return

        local_inputs = self.facade.block_local_state_repo.get_inputs(self.block_id) or {}
        events_ref = local_inputs.get("events")
        
        # DEBUG: Log what we got from local state
        Log.info(
            f"EditorPanel: _load_events_from_local_state - local_inputs keys: {list(local_inputs.keys())}, "
            f"events_ref type: {type(events_ref).__name__}, "
            f"events_ref value: {events_ref if isinstance(events_ref, str) else (len(events_ref) if isinstance(events_ref, list) else 'unknown')}"
        )
        
        # Get EventDataItems from local state references (EZ layers from upstream)
        # These references were just set by _on_pull_data_clicked() which always pulls fresh data
        event_items: list[EventDataItem] = []
        if events_ref:
            event_ids = events_ref if isinstance(events_ref, list) else [events_ref]
            Log.info(f"EditorPanel: _load_events_from_local_state - processing {len(event_ids)} event reference(s)")
            
            for eid in event_ids:
                item = self.facade.data_item_repo.get(eid)
                if isinstance(item, EventDataItem):
                    # Check if this is a sync layer item (should be excluded from EZ layers)
                    if self._is_sync_layer_item(item):
                        Log.warning(
                            f"EditorPanel: WARNING - Sync layer item '{item.name}' (ID: {eid[:8]}...) "
                            f"found in local state events reference! This should not happen - sync layers "
                            f"should not be in local state references."
                        )
                    event_items.append(item)
                    Log.debug(f"EditorPanel: Loaded event item '{item.name}' (block_id={item.block_id}) from reference {eid[:8]}...")
                elif item is None:
                    Log.warning(f"EditorPanel: Reference {eid[:8]}... points to deleted/missing item (None)")
                else:
                    Log.warning(f"EditorPanel: Reference {eid[:8]}... is not an EventDataItem (type: {type(item).__name__})")
        
        # CRITICAL: Also include sync layer items (owned by this Editor block)
        # Sync layers are not in local state references - they're owned by Editor and should be preserved
        sync_items = self.facade.data_item_repo.list_by_block(self.block_id)
        for item in sync_items:
            if isinstance(item, EventDataItem) and self._is_sync_layer_item(item):
                # Only add if not already in event_items (avoid duplicates)
                if item.id not in [e.id for e in event_items]:
                    event_items.append(item)
                    Log.debug(f"EditorPanel: Including sync layer item '{item.name}' (owned by Editor)")

        if not event_items:
            # No events at all - clear timeline
            self.timeline_widget.clear()
            self._loaded_event_items.clear()
            self._update_status_labels()
            return

        

        # Apply event filters before loading into timeline
        filtered_event_items = self._apply_event_filters(event_items)

        self._apply_event_items_to_timeline(
            event_items,
            filtered_event_items,
            skip_ui_state_restore=False,
            clear_existing=True
        )

        # Update UI labels
        total_events = sum(len(item.get_events()) for item in filtered_event_items)
        layers = self.timeline_widget.get_layers()
        duration = self.timeline_widget.get_duration()
        self.event_count_label.setText(f"Events: {total_events}")
        self.layer_count_label.setText(f"Layers: {len(layers)}")
        self.duration_label.setText(f"Duration: {duration:.1f}s")
        
        ez_count = len([item for item in event_items if self._is_ez_layer_item(item)])
        sync_count = len([item for item in event_items if self._is_sync_layer_item(item)])
        Log.info(
            f"EditorPanel: Loaded {total_events} events across {len(event_items)} sources "
            f"({ez_count} EZ layer item(s), {sync_count} sync layer item(s)) from local state references"
        )

    def _apply_event_items_to_timeline(
        self,
        event_items: list["EventDataItem"],
        filtered_event_items: list["EventDataItem"],
        *,
        skip_ui_state_restore: bool,
        clear_existing: bool
    ) -> list:
        """
        Apply EventDataItems to the timeline using a unified layer-build path.

        Args:
            event_items: Raw EventDataItems (unfiltered).
            filtered_event_items: EventDataItems after filters are applied.
            skip_ui_state_restore: If True, build layers from EventLayers directly.
            clear_existing: If True, clear timeline and related caches before loading.
        Returns:
            The reconciled layer order applied (if any).
        """
        if not hasattr(self, "timeline_widget") or not self.timeline_widget:
            return []

        if clear_existing:
            self.timeline_widget.clear()

        # Reset caches for a consistent rebuild
        self._loaded_event_items.clear()

        for item in event_items:
            self._loaded_event_items[item.id] = item

        if event_items:
            if not skip_ui_state_restore:
                self._restore_layer_state()
            else:
                Log.info("EditorPanel: Skipping UI state restore - layers will be created from EventDataItems")

        # set_events_from_data_items creates layers from EventLayers if they don't exist
        try:
            self.timeline_widget.set_events_from_data_items(filtered_event_items)
        except Exception as e:
            raise

        layer_order = self._reconcile_layer_order()
        if layer_order:
            self._apply_layer_order(layer_order)

        self._update_status_labels()
        return layer_order

    def _get_group_identity_for_item(self, item: EventDataItem) -> tuple[str, str]:
        """
        Resolve the group_id and group_name for an EventDataItem.

        EventDataItem is the single source of truth. We only honor metadata
        group_id when it is an explicit MA3 sync group (tc_ prefix).
        """
        group_id = item.id
        group_name = item.name
        if item.metadata:
            meta_group_id = item.metadata.get("group_id")
            meta_group_name = item.metadata.get("group_name")
            if isinstance(meta_group_id, str) and meta_group_id.startswith("tc_"):
                group_id = meta_group_id
                group_name = meta_group_name or group_name
        return group_id, group_name
    
    def _load_audio_from_local_state(self):
        """
        Load referenced audio into the Editor UI from persisted local state.

        Unlike events (which become Editor-owned editable outputs), audio remains a reference.
        
        Validates audio file existence and shows warnings for stale data.
        """
        if not self.block:
            return
        if not hasattr(self.facade, "block_local_state_repo") or not self.facade.block_local_state_repo:
            return
        if not hasattr(self.facade, "data_item_repo") or not self.facade.data_item_repo:
            return

        local_inputs = self.facade.block_local_state_repo.get_inputs(self.block_id) or {}
        audio_ref = local_inputs.get("audio")
        if not audio_ref:
            return

        from src.shared.domain.entities import AudioDataItem
        from src.utils.tools import validate_audio_items

        audio_ids = audio_ref if isinstance(audio_ref, list) else [audio_ref]
        audio_items: list[AudioDataItem] = []
        for aid in audio_ids:
            item = self.facade.data_item_repo.get(aid)
            if isinstance(item, AudioDataItem):
                audio_items.append(item)

        if not audio_items:
            return

        # Validate audio items for file existence
        validation_result = validate_audio_items(audio_items)
        
        # Store validation results
        self._has_stale_audio = not validation_result['all_valid']
        self._stale_audio_errors = []
        
        if not validation_result['all_valid']:
            # Build error messages
            for item, error in validation_result['invalid']:
                self._stale_audio_errors.append(f"{item.name}: {error}")
                Log.warning(f"EditorPanel: Stale audio data - {item.name}: {error}")
            
            # Show warning banner
            self._show_stale_data_warning()

        # Load all audio items (even invalid ones, for metadata)
        for item in audio_items:
            self._loaded_audio_items[item.id] = item

        # Update playback + label (only with valid audio)
        valid_audio = validation_result['valid']
        if valid_audio:
            self._setup_audio_playback(valid_audio)
            self.audio_label.setText(f"Audio: {valid_audio[0].name}" if len(valid_audio) == 1 else f"Audio: {len(valid_audio)} items")
        else:
            self.audio_label.setText("⚠ Audio files missing")
            self.audio_label.setStyleSheet(f"color: {Colors.STATUS_WARNING.name()}; font-weight: bold;")

    def _touch_local_state_db(self, reason: str):
        """
        Ensure local-state DB is updated in real time after UI edits.

        Model: local state is the single source of truth for port routing; edits should update
        the local-state row (at least its updated_at) so downstream/UI refresh logic can observe changes.
        
        CRITICAL: During Pull Data, we must NOT overwrite local state with owned items,
        because Pull Data is setting local state to point to EXTERNAL upstream items.
        """
        if not hasattr(self.facade, "block_local_state_repo") or not self.facade.block_local_state_repo:
            return
        
        
        try:

            # Option A: prefer facade helper (centralizes DB orchestration).
            if hasattr(self.facade, "touch_block_local_state"):
                res = self.facade.touch_block_local_state(self.block_id, reason=reason)
                return

            # Rebuild from actual current Editor state:
            # - events: point to Editor-owned editable EventDataItems (what user is editing)
            # - audio: preserve existing audio ref if present; otherwise infer from incoming audio connection
            current = self.facade.block_local_state_repo.get_inputs(self.block_id) or {}

            # Events: always route from Editor-owned items
            # BUT: Only if we're not in Pull Data mode (Pull Data sets events to external references)
            event_ids = list(self._loaded_event_items.keys()) if hasattr(self, "_loaded_event_items") else []
            if event_ids:
                current["events"] = event_ids[0] if len(event_ids) == 1 else event_ids

            # Audio: first check Editor-owned outputs (like events), then fall back to source block
            if "audio" not in current or not current.get("audio"):
                # First, check if we have Editor-owned audio outputs (from execution)
                audio_ids = list(self._loaded_audio_items.keys()) if hasattr(self, "_loaded_audio_items") else []
                if audio_ids:
                    # Use Editor's own audio outputs (similar to how events work)
                    current["audio"] = audio_ids[0] if len(audio_ids) == 1 else audio_ids
                else:
                    # Fall back to source block (for initial connection setup)
                    try:
                        if hasattr(self.facade, "connection_service") and self.facade.connection_service:
                            conns = self.facade.connection_service.list_connections_by_block(self.block_id)
                            incoming_audio = [
                                c for c in conns
                                if c.target_block_id == self.block_id and c.target_input_name == "audio"
                            ]
                            if incoming_audio:
                                c = incoming_audio[0]
                                # Prefer upstream local state for the source port
                                src_local = self.facade.block_local_state_repo.get_inputs(c.source_block_id) or {}
                                ref = src_local.get(c.source_output_name)
                                if not ref and hasattr(self.facade, "data_item_repo") and self.facade.data_item_repo:
                                    # Bridge: fall back to persisted outputs on source block
                                    source_items = self.facade.data_item_repo.list_by_block(c.source_block_id)
                                    matching = [
                                        item for item in source_items
                                        if item.metadata.get("output_port") == c.source_output_name
                                    ]
                                    if len(matching) == 1:
                                        ref = matching[0].id
                                    elif len(matching) > 1:
                                        ref = [i.id for i in matching]
                                if ref:
                                    current["audio"] = ref
                    except Exception:
                        pass

            self.facade.block_local_state_repo.set_inputs(self.block_id, current)
        except Exception as e:
            Log.debug(f"EditorPanel: Failed to touch local state: {e}")

    def _initialize_owned_outputs_from_local_state(
        self,
        force: bool = False,
        preserve_sync_layers: bool = True,
        skip_ui_state_restore: bool = False
    ):
        """
        Initialize Editor-owned EventDataItems from persisted local inputs (references only).

        This enables the workflow:
        Pull Data (Overwrite) -> Editor creates owned editable outputs -> edits persist -> downstream pulls Editor outputs
        
        CRITICAL: Only recreates if local_state["events"] points to EXTERNAL items (not already owned by this block).
        This prevents cascading copies (Editor1_Editor1_..._edited_edited_edited).
        
        Args:
            force: Force recreation even if items are already owned
            preserve_sync_layers: If True, preserve sync layer EventDataItems during refresh
            skip_ui_state_restore: If True, skip UI state restore when loading owned data
        """
        if not self.block or not hasattr(self.facade, "block_local_state_repo") or not self.facade.block_local_state_repo:
            return
        if not hasattr(self.facade, "data_item_repo") or not self.facade.data_item_repo:
            return

        local_inputs = self.facade.block_local_state_repo.get_inputs(self.block_id) or {}
        if not local_inputs:
            return

        # Ensure audio is present in Editor local state if there's an incoming audio connection.
        if "audio" not in local_inputs:
            try:
                conns = self.facade.connection_service.list_connections_by_block(self.block_id)
                incoming_audio = [
                    c for c in conns
                    if c.target_block_id == self.block_id and c.target_input_name == "audio"
                ]
                if incoming_audio:
                    c = incoming_audio[0]
                    src_local = self.facade.block_local_state_repo.get_inputs(c.source_block_id) or {}
                    ref = src_local.get(c.source_output_name)
                    if not ref:
                        source_items = self.facade.data_item_repo.list_by_block(c.source_block_id)
                        matching = [
                            item for item in source_items
                            if item.metadata.get("output_port") == c.source_output_name
                        ]
                        if len(matching) == 1:
                            ref = matching[0].id
                        elif len(matching) > 1:
                            ref = [i.id for i in matching]
                    if ref:
                        local_inputs["audio"] = ref
                        self.facade.block_local_state_repo.set_inputs(self.block_id, local_inputs)
            except Exception:
                pass

        events_ref = local_inputs.get("events")
        if not events_ref:
            return

        # Normalize to list of ids
        source_ids = events_ref if isinstance(events_ref, list) else [events_ref]
        
        Log.debug(f"EditorPanel: Checking {len(source_ids)} source reference(s) for recreation (force={force})")

        from src.shared.domain.entities import EventDataItem

        # CRITICAL CHECK: Are the referenced items already owned by this block?
        # 
        # For Pull Data (force=True): Always recreate from upstream to restore deleted events.
        # For normal initialization (force=False): Skip recreation if items are already owned
        #   to avoid unnecessary work, but this should be rare since local state should
        #   point to external blocks after pull_block_inputs_overwrite().
        all_owned = True
        external_block_ids = []
        owned_source_ids = 0
        external_source_ids = 0
        missing_source_ids = 0
        owned_with_source_meta = 0
        owned_missing_source_meta = 0
        owned_original_found = 0
        owned_original_missing = 0
        source_item_sample = []
        for sid in source_ids:
            item = self.facade.data_item_repo.get(sid)
            if item:
                if item.block_id == self.block_id:
                    Log.debug(f"EditorPanel: Source ID {sid[:8]}... is owned by this block (name: {item.name})")
                    owned_source_ids += 1
                    source_item_sample.append({
                        "id": item.id,
                        "name": item.name,
                        "block_id": item.block_id,
                        "source_item_id": item.metadata.get("_source_item_id") if item.metadata else None,
                        "source_item_name": item.metadata.get("_source_item_name") if item.metadata else None
                    })
                    if item.metadata and item.metadata.get("_source_item_id") and item.metadata.get("_source_block_id"):
                        owned_with_source_meta += 1
                    else:
                        owned_missing_source_meta += 1
                else:
                    Log.debug(f"EditorPanel: Source ID {sid[:8]}... is from external block {item.block_id} (name: {item.name})")
                    all_owned = False
                    external_block_ids.append(item.block_id)
                    external_source_ids += 1
                    source_item_sample.append({
                        "id": item.id,
                        "name": item.name,
                        "block_id": item.block_id,
                        "source_item_id": item.metadata.get("_source_item_id") if item.metadata else None,
                        "source_item_name": item.metadata.get("_source_item_name") if item.metadata else None
                    })
            else:
                Log.warning(f"EditorPanel: Source ID {sid[:8]}... not found in repository")
                all_owned = False
                external_block_ids.append("MISSING")
                missing_source_ids += 1

        if all_owned and not force:
            # Items are already our owned outputs - just reload them, don't recreate
            # This is an optimization for normal initialization, but should be rare.
            # Pull Data should always use force=True to ensure recreation from upstream.
            Log.debug("EditorPanel: All items already owned, skipping recreation (force=False)")
            self._load_owned_data()
            return
        
        if all_owned and force:
            # This shouldn't normally happen after pull_block_inputs_overwrite(), but if it does,
            # we'll recreate anyway to ensure deleted events are restored.
            Log.warning(
                "EditorPanel: Items appear owned but force=True - this suggests pull_block_inputs_overwrite() "
                "may not have updated references correctly. Recreating anyway to restore deleted events."
            )

        # Get source items (either from external blocks, or from our own if force=True)
        # CRITICAL: Exclude sync layer items from source_items - they should be preserved, not recreated
        source_items: list[EventDataItem] = []
        sync_item_ids_set = set(self._get_preserved_sync_item_ids()) if preserve_sync_layers else set()
        
        for sid in source_ids:
            # Skip sync layer items - they should never be recreated, only preserved
            if sid in sync_item_ids_set:
                Log.debug(f"EditorPanel: Skipping sync layer item {sid[:8]}... from source items (will preserve)")
                continue
            
            item = self.facade.data_item_repo.get(sid)
            if isinstance(item, EventDataItem):
                # Double-check it's not a sync layer (in case it wasn't in preserved list yet)
                if preserve_sync_layers and self._is_sync_layer_item(item):
                    Log.warning(f"EditorPanel: Source item {sid[:8]}... is a sync layer - skipping recreation")
                    continue
                
                # If item is owned by this block but force=True, try to get original source
                if item.block_id == self.block_id and force:
                    # Check if item has source metadata - if so, try to get original source
                    source_block_id = item.metadata.get("_source_block_id")
                    source_item_id = item.metadata.get("_source_item_id")
                    if source_block_id and source_block_id != self.block_id and source_item_id:
                        # Try to get the original source item from upstream block
                        original_source = self.facade.data_item_repo.get(source_item_id)
                        if original_source and original_source.block_id == source_block_id:
                            owned_original_found += 1
                            Log.info(
                                f"EditorPanel: Using original source item '{original_source.name}' "
                                f"from block {source_block_id} instead of owned copy"
                            )
                            source_items.append(original_source)
                            continue
                        owned_original_missing += 1
                    else:
                        owned_original_missing += 1
                
                source_items.append(item)
        
        if not source_items:
            Log.warning("EditorPanel: No source items found to recreate from")
            return
        
        Log.info(
            f"EditorPanel: Recreating {len(source_items)} EventDataItem(s) from "
            f"{'owned' if all_owned else 'external'} sources (force={force})"
        )
        
        # Get sync layer item IDs to preserve (if requested)
        sync_item_ids = set()
        if preserve_sync_layers:
            sync_item_ids = set(self._get_preserved_sync_item_ids())
            if sync_item_ids:
                Log.info(f"EditorPanel: Preserving {len(sync_item_ids)} sync layer item(s) during refresh")
        
        # Delete existing owned items EXCEPT sync layers
        owned_items = self.facade.data_item_repo.list_by_block(self.block_id)
        deleted_event_count = 0
        deleted_audio_count = 0
        for item in owned_items:
            if isinstance(item, EventDataItem) and item.metadata.get("output_port") == "events":
                # Skip sync layer items
                if item.id in sync_item_ids:
                    Log.debug(f"EditorPanel: Preserving sync layer item '{item.name}' (ID: {item.id})")
                    continue
                try:
                    self.facade.data_item_repo.delete(item.id)
                    deleted_event_count += 1
                except Exception:
                    pass
            elif isinstance(item, AudioDataItem) and item.metadata.get("output_port") == "audio":
                # Delete Editor-owned audio items (will be recreated from upstream)
                try:
                    self.facade.data_item_repo.delete(item.id)
                    deleted_audio_count += 1
                except Exception:
                    pass
        if deleted_event_count > 0:
            Log.info(f"EditorPanel: Deleted {deleted_event_count} EZ layer EventDataItem(s) before refresh")
        if deleted_audio_count > 0:
            Log.info(f"EditorPanel: Deleted {deleted_audio_count} Editor-owned AudioDataItem(s) before refresh")

        created_ids: list[str] = []
        # Create owned editable copies
        # PRESERVE EventLayers from source items (same approach as EditorBlockProcessor)
        from src.shared.domain.entities import EventLayer
        
        for src in source_items:
            # Check if source has native layers (single source of truth)
            has_internal_layers = (
                hasattr(src, '_layers') and 
                src._layers and 
                len(src._layers) > 0 and
                (not hasattr(src, '_events') or not src._events or len(src._events) == 0)
            )
            
            if has_internal_layers:
                # PRESERVE EventLayers - layers are the single source of truth
                input_layers = src._layers
                
                # Create new EventDataItem with preserved layers
                owned_layers = []
                for layer in input_layers:
                    # Create new EventLayer with same name and events
                    owned_layer = EventLayer(
                        name=layer.name,
                        events=layer.events.copy(),  # Copy events but keep same objects (preserve metadata)
                        metadata=layer.metadata.copy() if layer.metadata else {}
                    )
                    owned_layers.append(owned_layer)
                
                owned = EventDataItem(
                    id="",
                    block_id=self.block_id,
                    name=f"{self.block.name}_{src.name}_edited",
                    type="Event",
                    metadata={
                        "output_port": "events",
                        "_source_item_id": src.id,
                        "_source_item_name": src.name,
                        "_source_block_id": src.block_id,
                    },
                    layers=owned_layers  # PRESERVE LAYERS
                )
                
                total_events = sum(len(l.events) for l in owned_layers)
                Log.info(
                    f"EditorPanel: Created owned EventDataItem '{owned.name}' "
                    f"with {len(owned_layers)} layer(s) and {total_events} event(s) from '{src.name}'"
                )
            else:
                # LEGACY: Source has no layers, create flat structure (backward compatibility)
                owned = EventDataItem(
                    id="",
                    block_id=self.block_id,
                    name=f"{self.block.name}_{src.name}_edited",
                    type="Event",
                    metadata={
                        "output_port": "events",
                        "_source_item_id": src.id,
                        "_source_item_name": src.name,
                        "_source_block_id": src.block_id,
                    },
                )
                for ev in src.get_events():
                    # PRESERVE _visual_layer_name - layer names are shared/global, not block-specific
                    event_metadata = {}
                    if ev.metadata:
                        # Copy all metadata including _visual_layer_name
                        event_metadata = dict(ev.metadata)
                    
                    # For legacy flat events, use classification as layer name
                    layer_name = ev.classification or "default"
                    owned.add_event(
                        time=ev.time,
                        classification=ev.classification,
                        duration=ev.duration,
                        metadata=event_metadata,
                        layer_name=layer_name,
                    )
                
                Log.debug(
                    f"EditorPanel: Created owned EventDataItem '{owned.name}' "
                    f"in legacy flat format (no layers) from '{src.name}'"
                )

            self.facade.data_item_repo.create(owned)
            created_ids.append(owned.id)

        # =====================================================================
        # Phase 1: Create Editor-owned copies of upstream AudioDataItems
        # =====================================================================
        # The Editor should own everything it needs to serve downstream.
        # Audio items from upstream (e.g., Separator) can go stale if re-executed.
        # By creating owned copies, the Editor has stable audio references that
        # survive upstream re-execution. A Pull Data refreshes them.
        audio_id_mapping: Dict[str, str] = {}  # upstream_audio_id -> editor_owned_audio_id
        created_audio_ids: list[str] = []

        audio_ref = local_inputs.get("audio")
        if audio_ref:
            upstream_audio_ids = audio_ref if isinstance(audio_ref, list) else [audio_ref]
            for upstream_id in upstream_audio_ids:
                upstream_item = self.facade.data_item_repo.get(upstream_id)
                if isinstance(upstream_item, AudioDataItem):
                    # Create Editor-owned copy (lightweight - same file_path, copied metadata)
                    owned_audio = AudioDataItem(
                        id="",  # New UUID will be auto-generated
                        block_id=self.block_id,
                        name=upstream_item.name,
                        type="Audio",
                        file_path=upstream_item.file_path,
                        sample_rate=upstream_item.sample_rate,
                        length_ms=upstream_item.length_ms,
                        channels=upstream_item.channels,
                        metadata={
                            **dict(upstream_item.metadata),
                            "output_port": "audio",
                            "_source_item_id": upstream_item.id,
                            "_source_item_name": upstream_item.name,
                            "_source_block_id": upstream_item.block_id,
                        },
                    )
                    self.facade.data_item_repo.create(owned_audio)
                    audio_id_mapping[upstream_item.id] = owned_audio.id
                    created_audio_ids.append(owned_audio.id)
                    Log.info(
                        f"EditorPanel: Created owned AudioDataItem '{owned_audio.name}' "
                        f"(ID: {owned_audio.id[:8]}...) from upstream '{upstream_item.name}' "
                        f"(ID: {upstream_item.id[:8]}...)"
                    )
                else:
                    Log.warning(f"EditorPanel: Upstream audio ID {upstream_id[:8]}... not found or not AudioDataItem")

        # Update audio_id references in owned events to point to Editor-owned audio items
        if audio_id_mapping:
            remapped_count = 0
            for event_id in created_ids:
                owned_event = self.facade.data_item_repo.get(event_id)
                if isinstance(owned_event, EventDataItem):
                    changed = False
                    for layer in owned_event.get_layers():
                        for ev in layer.events:
                            old_audio_id = ev.metadata.get("audio_id")
                            if old_audio_id and old_audio_id in audio_id_mapping:
                                ev.metadata["audio_id"] = audio_id_mapping[old_audio_id]
                                changed = True
                                remapped_count += 1
                    if changed:
                        # Persist the updated event metadata
                        self.facade.data_item_repo.update(owned_event)
            if remapped_count > 0:
                Log.info(
                    f"EditorPanel: Remapped {remapped_count} audio_id reference(s) in events "
                    f"to Editor-owned audio items"
                )

        # Update local state to point to owned items (so downstream can pull them)
        try:
            local_inputs["events"] = created_ids if len(created_ids) > 1 else created_ids[0]
            if created_audio_ids:
                local_inputs["audio"] = created_audio_ids if len(created_audio_ids) > 1 else created_audio_ids[0]
            self.facade.block_local_state_repo.set_inputs(self.block_id, local_inputs)
        except Exception:
            pass

        # Reload owned data into timeline for editing
        self._load_owned_data(skip_ui_state_restore=skip_ui_state_restore)
    
    def _pull_upstream_data(self, skip_confirmation: bool = False) -> bool:
        """
        Pull upstream data from connections using standard facade method.
        
        This is the standard Pull Data pattern used by all blocks.
        It pulls fresh data from upstream connections and stores references in local state.
        
        Args:
            skip_confirmation: If True, skip the confirmation dialog (used by Execute)
            
        Returns:
            True if pull was successful, False otherwise
        """
        if not self.block_id or not self.facade:
            return False
        
        if not skip_confirmation:
            reply = QMessageBox.question(
                self,
                "Pull Data (Overwrite)",
                "Overwrite this block's local inputs and re-pull from its connections?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
        
        result = self.facade.pull_block_inputs_overwrite(self.block_id)
        
        if hasattr(result, "success") and result.success:
            self.set_status_message("Pulled local inputs", error=False)
            return True
        
        # Handle missing upstream data
        missing_txt = ""
        try:
            if hasattr(result, "errors") and result.errors:
                # We encode missing details into errors as: "missing=[...]"
                for e in result.errors:
                    if isinstance(e, str) and e.startswith("missing="):
                        missing_txt = e[len("missing="):]
                        break
        except Exception:
            missing_txt = ""
        
        if missing_txt:
            msg_lines = ["Upstream has no data for one or more connections:", ""]
            msg_lines.append(missing_txt)
            msg_lines.append("")
            msg_lines.append("Execute upstream blocks first, then pull again.")
            if not skip_confirmation:
                QMessageBox.information(self, "Pull Data", "\n".join(msg_lines))
            self.set_status_message("Pull failed: upstream missing data", error=True)
        else:
            if not skip_confirmation:
                QMessageBox.warning(self, "Pull Data", getattr(result, "message", "Pull failed"))
            self.set_status_message("Pull failed", error=True)
        
        return False
    
    def _continue_pull_data_after_block_updated(self):
        """
        Continue Pull Data operation after BlockUpdated event has been processed.
        
        This is called via QTimer to ensure BlockUpdated event (if published by pull_block_inputs_overwrite)
        has been handled before we try to load from local state.
        """
        from src.utils.message import Log
        
        # Step 3: Rebuild owned outputs from local state (preserve sync layers)
        # Use the same unified layer-build path as load and Pull Data.
        self._initialize_owned_outputs_from_local_state(
            force=True,
            preserve_sync_layers=True,
            skip_ui_state_restore=True
        )

        # Notify sync system so actively synced layers push fresh events to MA3
        try:
            self._notify_sync_of_pulled_data()
        except Exception as e:
            Log.debug(f"EditorPanel: Pull Data sync notification failed: {e}")

        # Step 4: Load and display audio from the fresh references
        self._load_audio_from_local_state()
        
        self.set_status_message("Pulled fresh data from upstream", error=False)
        Log.info(f"EditorPanel: Pull Data complete - cleaned and recreated EZ layers from upstream data")
        
        # Clear flag after Pull Data completes
        self._pull_data_active = False
    
    def _delete_ez_layers(self) -> int:
        """
        Delete all EZ (EchoZero) layers from both repository and timeline widget.
        
        This deletes:
        - EventDataItems from repository (EZ layers only)
        - Timeline widget layers (EZ layers only)
        
        SYNC layers are preserved and not deleted.
        
        Returns:
            Number of EZ layers deleted
        """
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        
        if not hasattr(self.facade, 'data_item_repo') or not self.facade.data_item_repo:
            Log.warning("EditorPanel: Data item repository not available for EZ layer deletion")
            return 0
        
        # Get all owned EventDataItems
        owned_items = self.facade.data_item_repo.list_by_block(self.block_id)
        
        # Identify EZ layer items and their corresponding timeline layer IDs
        ez_items_to_delete = []
        layer_ids_to_delete = []
        
        for item in owned_items:
            if isinstance(item, EventDataItem) and item.metadata.get("output_port") == "events":
                # Skip sync layer items
                if self._is_sync_layer_item(item):
                    Log.debug(f"EditorPanel: Preserving sync layer item '{item.name}' (ID: {item.id})")
                    continue
                
                # This is an EZ layer item - mark for deletion
                ez_items_to_delete.append(item)
                
                # Find corresponding timeline widget layers
                # Layers are created with IDs like "layer_{item_id}_{layer_id}"
                if hasattr(self, 'timeline_widget') and self.timeline_widget:
                    layers = self.timeline_widget.get_layers()
                    for layer in layers:
                        # Match by group_id (which is set to item.id) or by layer name from EventLayers
                        if layer.group_id == item.id:
                            layer_ids_to_delete.append(layer.id)
                        # Also check if layer name matches any EventLayer name in the item
                        elif hasattr(item, '_layers') and item._layers:
                            for event_layer in item._layers:
                                if layer.name == event_layer.name and layer.group_id == item.id:
                                    if layer.id not in layer_ids_to_delete:
                                        layer_ids_to_delete.append(layer.id)
        
        deleted_repo_count = 0
        deleted_ui_count = 0
        
        # Delete EventDataItems from repository
        for item in ez_items_to_delete:
            try:
                self.facade.data_item_repo.delete(item.id)
                deleted_repo_count += 1
                Log.debug(f"EditorPanel: Deleted EZ layer EventDataItem '{item.name}' (ID: {item.id})")
            except Exception as e:
                Log.warning(f"EditorPanel: Failed to delete EventDataItem {item.id}: {e}")
        
        # Delete timeline widget layers
        if hasattr(self, 'timeline_widget') and self.timeline_widget:
            for layer_id in layer_ids_to_delete:
                try:
                    if self.timeline_widget.remove_layer(layer_id):
                        deleted_ui_count += 1
                        Log.debug(f"EditorPanel: Deleted timeline widget layer '{layer_id}'")
                except Exception as e:
                    Log.warning(f"EditorPanel: Failed to delete timeline layer {layer_id}: {e}")
        
        total_deleted = deleted_repo_count
        if total_deleted > 0:
            Log.info(
                f"EditorPanel: Deleted {deleted_repo_count} EZ layer EventDataItem(s) "
                f"and {deleted_ui_count} timeline widget layer(s)"
            )
        
        return total_deleted
    
    def _recreate_layers_from_pulled_data(self):
        """
        Recreate EZ layers from pulled upstream data.
        
        This method ensures layers are ready for display after Pull Data.
        The actual layer creation happens in _load_events_from_local_state() via
        set_events_from_data_items(), which creates layers from EventLayers in EventDataItems.
        
        This method is a placeholder for any pre-loading preparation needed.
        SYNC layers are preserved and not recreated.
        """
        from src.utils.message import Log
        
        # Layers will be created automatically by _load_events_from_local_state()
        # when it calls set_events_from_data_items() with the pulled EventDataItems.
        # This method exists for consistency with the plan, but the actual recreation
        # happens during the load step.
        Log.debug("EditorPanel: Layers will be recreated during _load_events_from_local_state()")
    
    def _on_pull_data_clicked(self):
        """
        Pull data from upstream and rebuild owned outputs (execute-like).

        Flow:
        1. Pull upstream inputs into local state (overwrite)
        2. Recreate owned EventDataItems from upstream (preserve sync layers)
        3. Refresh UI from owned outputs
        """
        try:
            Log.info(f"EditorPanel: Pull Data - rebuilding owned outputs for block '{self.block_id}'")
            self.set_status_message("Pulling data...", error=False)

            self._pull_data_active = True

            if not self._pull_upstream_data(skip_confirmation=False):
                return
            
            # Recreate owned outputs to match execution behavior
            self._initialize_owned_outputs_from_local_state(
                force=True,
                preserve_sync_layers=True,
                skip_ui_state_restore=True
            )

            # Notify sync system so actively synced layers push fresh events to MA3
            try:
                self._notify_sync_of_pulled_data()
            except Exception as e:
                Log.debug(f"EditorPanel: Pull Data sync notification failed: {e}")

            try:
                self._load_audio_from_local_state()
            except Exception as e:
                Log.debug(f"EditorPanel: Pull Data load audio failed: {e}")

            self.set_status_message("Pulled data and rebuilt outputs", error=False)
        except Exception as e:
            import traceback
            Log.error(f"EditorPanel: Pull Data failed: {e}")
            Log.debug(f"EditorPanel: Pull Data traceback: {traceback.format_exc()}")
            QMessageBox.warning(self, "Pull Data", f"Failed to pull data: {e}")
            self.set_status_message("Pull Data failed", error=True)
        finally:
            self._pull_data_active = False
    
    def _notify_sync_of_pulled_data(self):
        """
        Publish BlockUpdated after Pull Data so synced layers push fresh events to MA3.
        
        When an editor layer is actively synced to an MA3 track and the user pulls
        fresh data from upstream, the new events must be sent to MA3. This method
        collects all layer names from the freshly loaded EventDataItems and publishes
        a BlockUpdated event that the SyncSystemManager will pick up to schedule
        Editor->MA3 pushes for any matching synced layers.
        
        Uses source="pull_data" to:
        - Avoid being skipped (source="ma3_sync" is skipped)
        - Avoid being treated as a user edit (source="editor")
        - Allow the EditorPanel handler to ignore it (pull_data_active is True)
        """
        from src.application.events.events import BlockUpdated
        
        if not hasattr(self, 'facade') or not self.facade:
            return
        if not hasattr(self.facade, 'event_bus') or not self.facade.event_bus:
            return
        
        # Collect all layer names from freshly loaded EventDataItems
        layer_names = []
        for item in self._loaded_event_items.values():
            for layer in item.get_layers():
                if layer.name and layer.name not in layer_names:
                    layer_names.append(layer.name)
        
        if not layer_names:
            Log.debug("EditorPanel: No layer names to publish after Pull Data")
            return
        
        Log.info(f"EditorPanel: Publishing BlockUpdated after Pull Data for layers: {layer_names}")
        self.facade.event_bus.publish(BlockUpdated(
            data={
                "id": self.block_id,
                "events_updated": True,
                "source": "pull_data",
                "layer_names": layer_names,
            }
        ))
    
    def _on_execute_clicked(self):
        """
        Execute the block synchronously on the main thread (for testing; UI blocks until done).

        Prefer MainWindow execution when available, otherwise run facade.execute_block directly.
        """
        try:
            Log.info(f"EditorPanel: Execute - starting execution for block '{self.block_id}'")
            self.set_status_message("Executing block...", error=False)

            main_window = self.parent()
            while main_window and not hasattr(main_window, '_on_execute_single_block'):
                main_window = main_window.parent()

            if main_window and hasattr(main_window, '_on_execute_single_block'):
                main_window._on_execute_single_block(self.block_id)
            else:
                result = self.facade.execute_block(self.block_id)
                if result.success:
                    self.set_status_message(
                        "Execution completed",
                        error=False,
                    )
                else:
                    self.set_status_message(
                        f"Execution failed: {result.message or 'Unknown error'}",
                        error=True,
                    )

        except Exception as e:
            import traceback
            Log.error(f"EditorPanel: Execute failed: {e}")
            Log.debug(f"EditorPanel: Execute traceback: {traceback.format_exc()}")
            QMessageBox.warning(self, "Execute", f"Failed to execute block: {e}")
            self.set_status_message("Execute failed", error=True)
    
    def _on_block_updated_base(self, event):
        """Keep specialized EditorPanel BlockUpdated behavior active."""
        self._on_block_updated(event)

    def _on_block_updated(self, event):
        """
        Handle BlockUpdated event - refresh editor panel with latest data.
        
        OPTIMIZED: Only reload on full block executions, not on event additions.
        Event additions (via EditorAddEventsCommand) trigger BlockUpdated but don't
        require a full reload - we can update in place.
        
        EXECUTION: When execution_triggered=True, force complete layer recreation
        to ensure deleted layers are restored from source data.
        """
        try:
            updated_block_id = event.data.get("id") if hasattr(event, "data") else None
        except Exception:
            updated_block_id = None

        if updated_block_id != self.block_id:
            return

        # Check if this is just an event update (not a full execution)
        # EditorAddEventsCommand sets "events_updated": True
        is_event_only_update = event.data.get("events_updated", False) if hasattr(event, "data") else False
        
        # Check if this is an execution-triggered update (requires full layer recreation)
        is_execution_triggered = event.data.get("execution_triggered", False) if hasattr(event, "data") else False

        if getattr(self, "_pull_data_active", False) and not is_execution_triggered:
            return
        
        if is_event_only_update and not is_execution_triggered:
            # Just events were updated - check if this is a metadata-only update from editor
            # If source is "editor", we already updated the timeline widget directly, so skip reload
            # to avoid clearing and reloading all events (which can lose metadata for other events)
            update_source = event.data.get("source", "") if hasattr(event, "data") else ""
            if update_source == "editor":
                # Metadata updates from editor are already applied to timeline widget in event_update_callback
                # Skip reload to prevent clearing events and losing metadata for other events
                Log.debug(f"EditorPanel: BlockUpdated - editor metadata update, skipping reload (already applied)")
                return
            
            # Just events were added - reload EventDataItems from repo and update timeline
            # This prevents expensive full reload loops when ShowManager adds events
            Log.debug(f"EditorPanel: BlockUpdated - events only update, reloading EventDataItems in place")
            try:
                # Reload EventDataItems from repo (they now have new events)
                owned_items = self.facade.data_item_repo.list_by_block(self.block_id)
                event_items = [item for item in owned_items if isinstance(item, EventDataItem)]
                # Update our loaded items cache
                for item in event_items:
                    self._loaded_event_items[item.id] = item
                
                # Update timeline with refreshed data (layers already exist, just update events)
                if event_items:
                    filtered_event_items = self._apply_event_filters(event_items)
                    # set_events_from_data_items will use existing layers, only add new events
                    self.timeline_widget.set_events_from_data_items(filtered_event_items)
            except Exception as e:
                Log.debug(f"EditorPanel: Event-only update failed: {e}")
            
            return  # Skip full reload

        # Full block execution - reload everything
        if is_execution_triggered:
            Log.info(f"EditorPanel: BlockUpdated - execution triggered, forcing full layer recreation from EventDataItems")
        
        # Reload block model
        result = self.facade.describe_block(self.block_id)
        if result.success:
            self.block = result.data
            self._update_header()

        
        # For execution-triggered updates, skip UI state restoration
        # Layers will be created directly from EventDataItem's EventLayers
        # This ensures deleted layers are properly recreated from source data
        try:
            # Try loading owned data (from execution)
            # When execution_triggered=True, skip UI state restore to avoid stale data
            self._load_owned_data(skip_ui_state_restore=is_execution_triggered)
            
            # Check if we actually loaded anything
            if not self._loaded_event_items and not self._loaded_audio_items:
                # No owned data - initialize from local state (standard behavior)
                Log.debug(f"EditorPanel: BlockUpdated - no owned data, initializing from local state")
                self._initialize_owned_outputs_from_local_state(force=True)
        except Exception as e:
            Log.debug(f"EditorPanel: BlockUpdated load failed: {e}")
            # Fallback: try to initialize from local state
            try:
                self._initialize_owned_outputs_from_local_state(force=True)
            except Exception as e2:
                Log.debug(f"EditorPanel: BlockUpdated fallback init also failed: {e2}")

        try:
            self._load_audio_from_local_state()
        except Exception as e:
            Log.debug(f"EditorPanel: BlockUpdated load audio failed: {e}")

        # Ensure UI refreshes - use QTimer to ensure it happens after event processing
        # This is especially important when events come from background threads
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self.refresh)
        Log.debug(f"EditorPanel: BlockUpdated - scheduled UI refresh for block {self.block_id}")
    
    def _load_owned_data(self, skip_ui_state_restore: bool = False):
        """
        Load Editor block's execution outputs (EventDataItems and AudioDataItems).
        
        Editor UI loads from the block's own execution outputs (block_id = editor_block_id),
        not from input connections. This ensures UI edits and processor outputs are aligned.
        
        Args:
            skip_ui_state_restore: If True, skip restoring layers from UI state and let
                EventDataItems define the layers directly. Use this when execution has
                just recreated the data and UI state might be stale.
        """
        if not self.block:
            return
        
        try:
            if not hasattr(self.facade, 'data_item_repo') or not self.facade.data_item_repo:
                Log.warning("EditorPanel: data_item_repo not available")
                return
            
            # Preserve current layer order before clearing on reload.
            # This prevents stale saved order from reshuffling layers after pull/add/delete.
            if not getattr(self, "_layer_order_service", None):
                self._layer_order_service = getattr(self.facade, "layer_order_service", None)
            if not getattr(self, "_layer_group_order_service", None):
                self._layer_group_order_service = getattr(self.facade, "layer_group_order_service", None)
            layer_order_service = getattr(self, "_layer_order_service", None)
            layer_group_service = getattr(self, "_layer_group_order_service", None)
            # Preserve scroll position BEFORE clearing (prevents view reset on updates)
            preserved_scroll = None
            if hasattr(self, "timeline_widget") and self.timeline_widget:
                view = getattr(self.timeline_widget, "_view", None)
                if view:
                    v_bar = view.verticalScrollBar()
                    h_bar = view.horizontalScrollBar()
                    preserved_scroll = {
                        'vertical': v_bar.value() if v_bar else 0,
                        'horizontal': h_bar.value() if h_bar else 0,
                    }
            
            if hasattr(self, "timeline_widget"):
                # Verify block still exists before saving layer order (avoids FK constraint after block deletion)
                block_exists = False
                if self.facade:
                    result = self.facade.describe_block(self.block.id)
                    block_exists = result.success and result.data is not None
                if block_exists:
                    existing_layer_keys = self._get_current_layer_order_keys()
                    if existing_layer_keys and layer_order_service:
                        layer_order_service.save_order(self.block.id, existing_layer_keys)
                    group_order = []
                    layer_manager = None
                    if layer_group_service:
                        layer_manager = getattr(self.timeline_widget, "_layer_manager", None)
                        if layer_manager:
                            group_order, layers_by_group = layer_group_service.build_from_layers(
                                layer_manager.get_all_layers()
                            )
                            layer_group_service.save_order(self.block.id, group_order, layers_by_group)
            # Clear existing data
            self.timeline_widget.clear()
            self._loaded_event_items.clear()
            self._loaded_audio_items.clear()
            
            # Load owned EventDataItems and AudioDataItems from repository
            owned_items = self.facade.data_item_repo.list_by_block(self.block_id)
            event_items = [item for item in owned_items if isinstance(item, EventDataItem)]
            audio_items = [item for item in owned_items if isinstance(item, AudioDataItem)]

            # Verify we have both EZ and sync layer items
            ez_items = [item for item in event_items if self._is_ez_layer_item(item)]
            sync_items = [item for item in event_items if self._is_sync_layer_item(item)]
            if ez_items and sync_items:
                Log.debug(
                    f"EditorPanel: Loading {len(ez_items)} EZ layer item(s) and "
                    f"{len(sync_items)} sync layer item(s)"
                )
            
            # Store loaded items (both EZ and sync)
            for item in event_items:
                self._loaded_event_items[item.id] = item
            
            for item in audio_items:
                self._loaded_audio_items[item.id] = item
            
            filtered_event_items = []

            # Only restore layer state if we have events - layers should not exist without events
            # This ensures that when filter is empty and events are cleared, layers are also cleared
            if event_items:
                # Restore layer state FIRST (creates layers from UI state)
                # UNLESS skip_ui_state_restore is True (execution-triggered, let EventDataItems define layers)
                if not skip_ui_state_restore:
                    # This ensures layers exist before events are loaded
                    self._restore_layer_state()
                else:
                    Log.info(f"EditorPanel: Skipping UI state restore - layers will be created from EventDataItems")
                
                # Apply event filters before loading into timeline (part of "processes" improvement area)
                filtered_event_items = self._apply_event_filters(event_items)
            try:
                total_raw = sum(len(item.get_events()) for item in event_items)
                total_filtered = sum(len(item.get_events()) for item in filtered_event_items)
                sync_items = [item for item in event_items if self._is_sync_layer_item(item)]
                sync_filtered_items = [item for item in filtered_event_items if self._is_sync_layer_item(item)]
                sync_raw_count = sum(len(item.get_events()) for item in sync_items)
                sync_filtered_count = sum(len(item.get_events()) for item in sync_filtered_items)
                Log.info(
                    "EditorPanel: Filter summary "
                    f"(raw_events={total_raw}, filtered_events={total_filtered}, "
                    f"sync_raw={sync_raw_count}, sync_filtered={sync_filtered_count})"
                )
            except Exception as e:
                Log.debug(f"EditorPanel: Failed to log filter summary: {e}")
            layer_order = self._apply_event_items_to_timeline(
                event_items,
                filtered_event_items,
                skip_ui_state_restore=skip_ui_state_restore,
                clear_existing=False
            )

            # After loading, sync layer state back to UI state to persist any new layers
            if skip_ui_state_restore:
                self._save_layer_state()
            else:
                if not event_items:
                    # No events - ensure timeline is completely clear (clear() already called, but double-check)
                    # This handles the case where filter is empty and we want to clear everything
                    if hasattr(self.timeline_widget, '_layer_manager'):
                        # Ensure layers are cleared if somehow they weren't
                        current_layers = self.timeline_widget.get_layers()
                        if current_layers:
                            Log.debug(
                                f"EditorPanel: Clearing {len(current_layers)} layer(s) because no events loaded"
                            )
                            self.timeline_widget.clear()
            
            self._update_status_labels()
            
            if event_items:
                total_events = sum(len(item.get_events()) for item in event_items)
                ez_count = len([item for item in event_items if self._is_ez_layer_item(item)])
                sync_count = len([item for item in event_items if self._is_sync_layer_item(item)])
                Log.info(
                    f"EditorPanel: Loaded {total_events} events across {len(event_items)} sources "
                    f"({ez_count} EZ layer item(s), {sync_count} sync layer item(s))"
                )
                self.set_status_message(f"Loaded {total_events} events")
                
                # Update UI labels
                layers = self.timeline_widget.get_layers()
                duration = self.timeline_widget.get_duration()
                self.event_count_label.setText(f"Events: {total_events}")
                self.layer_count_label.setText(f"Layers: {len(layers)}")
                self.duration_label.setText(f"Duration: {duration:.1f}s")
            else:
                self.event_count_label.setText("Events: 0")
                self.layer_count_label.setText("Layers: 0")
                self.duration_label.setText("Duration: 0.0s")
                Log.info("EditorPanel: No events loaded")
            
            # Setup audio playback
            if audio_items:
                self._setup_audio_playback(audio_items)
            else:
                self.audio_label.setText("No audio loaded")
                Log.info("EditorPanel: No audio loaded")
            
            # Update info label
            if event_items or audio_items:
                audio_info = f"{len(audio_items)} audio" if audio_items else "no audio"
                event_info = f"{len(event_items)} event sources" if event_items else "no events"
                self.info_label.setText(f"Loaded: {audio_info}, {event_info}")
            else:
                self.info_label.setText("No data - execute block to generate outputs")
                # Don't show message box - just update label (less intrusive)
            
            # NOTE: Do not touch local state here.
            # Local state should be updated only on explicit user edits or post-execution writes.
            
            # Restore scroll position after loading (prevents view reset on updates)
            if preserved_scroll and hasattr(self, "timeline_widget") and self.timeline_widget:
                view = getattr(self.timeline_widget, "_view", None)
                if view:
                    from PyQt6.QtCore import QTimer
                    def restore_scroll():
                        v_bar = view.verticalScrollBar()
                        h_bar = view.horizontalScrollBar()
                        if h_bar:
                            h_bar.setValue(preserved_scroll['horizontal'])
                        if v_bar:
                            max_v = v_bar.maximum()
                            v_bar.setValue(min(preserved_scroll['vertical'], max_v) if max_v > 0 else 0)
                    # Use QTimer to ensure scroll restoration happens after Qt layout updates
                    QTimer.singleShot(0, restore_scroll)

        except Exception as e:
            Log.error(f"EditorPanel: Load error: {e}")
            import traceback
            traceback.print_exc()
            self.set_status_message("Load error", error=True)
    
    def _get_event_data_from_connections(self) -> List[EventDataItem]:
        """Get EventDataItem objects from connected source blocks"""
        event_items = []
        
        try:
            if not hasattr(self.facade, 'data_item_repo') or not self.facade.data_item_repo:
                Log.warning("EditorPanel: data_item_repo not available")
                return event_items
            
            # Get connections to this block
            connections_result = self.facade.list_connections()
            if not connections_result.success:
                return event_items
            
            # Find connections to this block's "events" input
            for conn in connections_result.data:
                if conn.target_block_id == self.block_id and conn.target_input_name == "events":
                    # Load data items from source block
                    source_data_items = self.facade.data_item_repo.list_by_block(conn.source_block_id)
                    
                    # Filter for EventDataItem outputs matching the source output port
                    for item in source_data_items:
                        if (item.metadata.get('output_port') == conn.source_output_name and
                            isinstance(item, EventDataItem)):
                            events = item.get_events()
                            if events:
                                event_items.append(item)
                            elif item.file_path and item.event_count > 0:
                                # Try to load from file
                                self._load_events_from_file(item)
                                if item.get_events():
                                    event_items.append(item)
        
        except Exception as e:
            Log.error(f"EditorPanel: Error getting event data: {e}")
            import traceback
            traceback.print_exc()
        
        return event_items
    
    def _get_audio_data_from_connections(self) -> List[AudioDataItem]:
        """Get AudioDataItem objects from connected source blocks
        
        Validates audio file existence and sets stale data flags.
        """
        audio_items = []
        
        try:
            if not hasattr(self.facade, 'data_item_repo') or not self.facade.data_item_repo:
                return audio_items
            
            # Get connections to this block
            connections_result = self.facade.list_connections()
            if not connections_result.success:
                return audio_items
            
            # Find connections to this block's "audio" input
            for conn in connections_result.data:
                if conn.target_block_id == self.block_id and conn.target_input_name == "audio":
                    # Load data items from source block
                    source_data_items = self.facade.data_item_repo.list_by_block(conn.source_block_id)
                    
                    # Filter for AudioDataItem outputs
                    for item in source_data_items:
                        if (item.metadata.get('output_port') == conn.source_output_name and
                            isinstance(item, AudioDataItem)):
                            audio_items.append(item)
            
            # Validate audio items if any were found
            if audio_items:
                from src.utils.tools import validate_audio_items
                validation_result = validate_audio_items(audio_items)
                
                # Track stale data
                if not validation_result['all_valid']:
                    self._has_stale_audio = True
                    self._stale_audio_errors = []
                    for item, error in validation_result['invalid']:
                        self._stale_audio_errors.append(f"{item.name}: {error}")
                        Log.warning(f"EditorPanel: Stale audio in connection - {item.name}: {error}")
        
        except Exception as e:
            Log.error(f"EditorPanel: Error getting audio data: {e}")
        
        return audio_items
    
    def _load_events_from_file(self, item: EventDataItem):
        """Load events from file into EventDataItem"""
        try:
            import json
            
            file_path = Path(item.file_path)
            if file_path.exists() and file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    events_data = data.get("events", [])
                elif isinstance(data, list):
                    events_data = data
                else:
                    events_data = []
                
                from src.domain.entities.event_data_item import Event
                for event_data in events_data:
                    event = Event.from_dict(event_data)
                    # Use layer_name from data if present, otherwise use classification
                    layer_name = event_data.get("layer_name") or event.classification or "default"
                    item.add_event(event.time, event.classification, 
                                   event.duration, event.metadata, layer_name=layer_name)
        except Exception as e:
            Log.debug(f"EditorPanel: Could not load events from file: {e}")
    
    def _setup_audio_playback(self, audio_items: List[AudioDataItem]):
        """Setup audio playback from loaded audio items
        
        Note: This method should only be called with validated audio items.
        The caller should use validate_audio_items() first to filter out stale data.
        """
        if not audio_items:
            return
        
        # Use first audio item with valid file path (should already be validated)
        audio_item = None
        for item in audio_items:
            if item.file_path and Path(item.file_path).exists():
                audio_item = item
                break
        
        if not audio_item:
            # Check metadata for file_path
            for item in audio_items:
                file_path = item.metadata.get('file_path')
                if file_path and Path(file_path).exists():
                    audio_item = item
                    self._current_audio_path = file_path
                    break
        else:
            self._current_audio_path = audio_item.file_path
        
        if not self._current_audio_path:
            self.audio_label.setText("⚠ Audio file not accessible")
            self.audio_label.setStyleSheet(f"color: {Colors.STATUS_WARNING.name()}; font-weight: bold;")
            return
        
        try:
            # Create or reuse audio player
            if self._audio_player is None:
                self._audio_player = SimpleAudioPlayer()
            
            # Load audio file
            if self._audio_player.load(self._current_audio_path):
                # Get current playhead position before connecting audio
                playhead_pos = self.timeline_widget.current_time if hasattr(self.timeline_widget, 'current_time') else 0.0
                
                # Connect to timeline
                self.timeline_widget.set_playback_controller(self._audio_player)
                
                # Sync playhead position after connecting (preserve user's playhead position)
                if playhead_pos > 0:
                    self.timeline_widget.seek(playhead_pos)
                
                # Update duration from audio
                duration = self._audio_player.get_duration()
                if duration > 0:
                    self.timeline_widget.set_duration(duration + 1)  # Add 1 second padding
                    self.duration_label.setText(f"Duration: {duration:.1f}s")
                
                # Update audio label
                filename = Path(self._current_audio_path).name
                self.audio_label.setText(f"Audio: {filename}")
                
                Log.info(f"EditorPanel: Audio playback ready: {filename}")
            else:
                self.audio_label.setText("Failed to load audio")
                Log.warning(f"EditorPanel: Failed to load audio: {self._current_audio_path}")
        
        except Exception as e:
            Log.error(f"EditorPanel: Audio setup error: {e}")
            self.audio_label.setText("Audio error")
    
    # ==================== Event Handlers ====================
    
    def _on_selection_changed(self, selected_ids: List[str]):
        """Handle selection change (new API)"""
        if selected_ids:
            Log.debug(f"EditorPanel: Selected {len(selected_ids)} event(s)")
            if len(selected_ids) == 1:
                self.set_status_message(f"Selected: {selected_ids[0]}")
            else:
                self.set_status_message(f"Selected {len(selected_ids)} events")
    
    def _on_position_changed(self, seconds: float):
        """Handle playhead position change"""
        pass  # Could update position display if needed
    
    def _on_playback_state_changed(self, is_playing: bool):
        """Handle playback state change"""
        pass  # Could update UI state if needed
    
    # ==================== Event Persistence ====================
    
    def _parse_event_id(self, event_id: str) -> Optional[tuple]:
        """
        Resolve event_id into (data_item_id, event_index).
        
        Expected: event_id is a stable UUID from the domain Event.
        """
        # Reject composite IDs (no fallback)
        if event_id and '_' in event_id:
            raise RuntimeError(f"Composite event_id not allowed: {event_id}")

        if not event_id:
            raise RuntimeError("Missing event_id (expected UUID)")

        # Resolve UUID via source item
        resolved = None
        try:
            import uuid as _uuid
            _uuid.UUID(event_id)
        except Exception:
            raise RuntimeError(f"Invalid event_id (expected UUID): {event_id}")

        
        item = None
        user_data = None
        if hasattr(self, 'timeline_widget') and self.timeline_widget:
            scene = getattr(self.timeline_widget, "_scene", None)
            
            
            if scene and hasattr(scene, "get_event_item"):
                item = scene.get_event_item(event_id)
                
                
                if item:
                    user_data = dict(item.user_data) if item.user_data else {}
                    
        
        source_item_id = user_data.get("_source_item_id") if user_data else None
        
        # If we didn't get source_item_id from scene, search through EventDataItems
        if not source_item_id and self.facade:
            
            from src.shared.domain.entities import EventDataItem
            event_data_items = self.facade.data_item_repo.list_by_block(self.block_id)
            
            for data_item in event_data_items:
                if not isinstance(data_item, EventDataItem):
                    continue
                events = data_item.get_events()
                
                for idx, ev in enumerate(events):
                    ev_id = getattr(ev, "id", None)
                    if ev_id == event_id:
                        source_item_id = data_item.id
                        resolved = (source_item_id, idx)
                        
                        break
                if resolved:
                    break
        
        # If we have source_item_id (from scene or repository), resolve event index
        if source_item_id and self.facade and not resolved:
            data_item = self.facade.data_item_repo.get(source_item_id)
            if data_item:
                events = data_item.get_events()
                for idx, ev in enumerate(events):
                    if getattr(ev, "id", None) == event_id:
                        resolved = (source_item_id, idx)
                        break

        if not resolved:
            raise RuntimeError(f"Event ID not found: {event_id} (searched scene and repository)")
        return resolved
        

    def _get_editor_api_for_sync(self):
        """Get or create shared EditorAPI for sync signals."""
        if not self.facade:
            return None
        registry = getattr(self.facade, "editor_api_registry", None)
        if isinstance(registry, dict) and self.block_id in registry:
            return registry.get(self.block_id)
        from src.features.blocks.application.editor_api import create_editor_api
        api = create_editor_api(self.facade, self.block_id)
        if isinstance(registry, dict) and api:
            registry[self.block_id] = api
        return api

    def _emit_editor_sync_signal(self, change_type: str, counts_by_layer: Dict[str, int]) -> None:
        """Emit EditorAPI signals for sync engine."""
        for layer_name, count in counts_by_layer.items():
            if not layer_name or count <= 0:
                continue
            sync_port = getattr(self.facade, "sync_port", None) if self.facade else None
            if sync_port:
                sync_port.apply_from_editor(
                    block_id=self.block_id,
                    layer_name=layer_name,
                    change_type=change_type,
                    count=count,
                    events=[]
                )
            else:
                editor_api = self._get_editor_api_for_sync()
                if not editor_api:
                    return
                if change_type == "added":
                    editor_api.events_added.emit(layer_name, count)
                elif change_type == "updated":
                    editor_api.events_updated.emit(layer_name, count)
                elif change_type == "deleted":
                    editor_api.events_deleted.emit(layer_name, count)
    
    def _on_events_moved(self, move_results: List[EventMoveResult]):
        """
        Handle events moved in timeline (new typed API).
        
        Receives already-batched results from MovementController.
        Creates a single batch command for all moves.
        """
        if not move_results or not self.facade:
            return
        
        # Group moves by data_item_id
        moves_by_item: Dict[str, List[Dict[str, Any]]] = {}
        counts_by_layer: Dict[str, int] = {}
        
        for result in move_results:
            resolved_layer_name = None
            if hasattr(self, 'timeline_widget') and self.timeline_widget:
                layer_manager = self.timeline_widget._layer_manager
                new_layer = layer_manager.get_layer(result.new_layer_id)
                if new_layer and new_layer.name:
                    resolved_layer_name = new_layer.name
            if resolved_layer_name:
                counts_by_layer[resolved_layer_name] = counts_by_layer.get(resolved_layer_name, 0) + 1
            parsed = self._parse_event_id(result.event_id)
            if not parsed:
                continue
            
            data_item_id, event_index = parsed
            
            if data_item_id not in moves_by_item:
                moves_by_item[data_item_id] = []
            
            # Persist time and layer changes (layer NAME stored in metadata for persistence)
            update = {
                'event_index': event_index,
                'time': result.new_time,
            }
            
            # Always track the current visual layer in metadata for DB persistence
            if hasattr(self, 'timeline_widget') and self.timeline_widget:
                layer_manager = self.timeline_widget._layer_manager
                new_layer = layer_manager.get_layer(result.new_layer_id)
                old_layer = layer_manager.get_layer(result.old_layer_id)
                
                if new_layer:
                    # Always store the current layer name in metadata (for persistence)
                    update['metadata'] = {'_visual_layer_name': new_layer.name}
                    
                    if result.layer_changed:
                        # Store old layer name separately for undo restoration
                        old_layer_name = old_layer.name if old_layer else None
                        update['_old_layer_name'] = old_layer_name
                        Log.info(f"EditorPanel: Event {result.event_id} layer changed: '{old_layer_name}' -> '{new_layer.name}' (saving to DB)")
                    else:
                        Log.debug(f"EditorPanel: Event {result.event_id} moved in layer '{new_layer.name}' (updating DB)")
            
            moves_by_item[data_item_id].append(update)
            
            # Update in-memory copy (always sync layer name)
            if data_item_id in self._loaded_event_items:
                item = self._loaded_event_items[data_item_id]
                events = item.get_events()
                if 0 <= event_index < len(events):
                    events[event_index].time = result.new_time
                    # Always update the visual layer name in memory
                    if hasattr(self, 'timeline_widget') and self.timeline_widget:
                        layer_manager = self.timeline_widget._layer_manager
                        new_layer = layer_manager.get_layer(result.new_layer_id)
                        if new_layer:
                            events[event_index].metadata['_visual_layer_name'] = new_layer.name
                    # Single source of truth: timeline layer name from new_layer_id
                    resolved_layer_name = None
                    if hasattr(self, 'timeline_widget') and self.timeline_widget:
                        layer_manager = self.timeline_widget._layer_manager
                        new_layer = layer_manager.get_layer(result.new_layer_id)
                        if new_layer and new_layer.name:
                            resolved_layer_name = new_layer.name
                    if resolved_layer_name:
                        counts_by_layer[resolved_layer_name] = counts_by_layer.get(resolved_layer_name, 0) + 1
        
        # Sync event moves to MA3 if layer is synced
        self._sync_event_moves_to_ma3(move_results)
        
        # Create batch commands
        total_events = sum(len(updates) for updates in moves_by_item.values())
        total_items = len(moves_by_item)
        
        use_macro = total_items > 1
        
        if use_macro:
            description = f"Move {total_events} Event{'s' if total_events > 1 else ''}"
            self.facade.command_bus.begin_macro(description)
        
        for data_item_id, updates in moves_by_item.items():
            count = len(updates)
            item_description = "Move Events" if use_macro else f"Move {count} Event{'s' if count > 1 else ''}"
            
            self.facade.command_bus.execute(BatchUpdateDataItemEventsCommand(
                self.facade,
                data_item_id,
                updates,
                description=item_description
            ))
        
        if use_macro:
            self.facade.command_bus.end_macro()
        
        # Touch local state so DB reflects edit immediately (no execute required)
        self._touch_local_state_db(reason="events_moved")
        self._emit_editor_sync_signal("updated", counts_by_layer)
        
        # Emit BlockUpdated so SyncSystemManager can push to MA3
        # source="editor" indicates this is a user edit (not from MA3 sync)
        # Include layer_names so only the changed layers are synced
        from src.application.events.events import BlockUpdated
        if hasattr(self, 'facade') and self.facade and hasattr(self.facade, 'event_bus') and self.facade.event_bus:
            changed_layer_names = list(counts_by_layer.keys())  # All layers that had events moved
            self.facade.event_bus.publish(BlockUpdated(
                data={
                    "id": self.block_id,
                    "events_updated": True,
                    "source": "editor",
                    "layer_names": changed_layer_names  # List of layer names that changed
                }
            ))
            Log.debug(f"EditorPanel: Published BlockUpdated for events_moved in layers: {changed_layer_names}")
    
    def _sync_event_moves_to_ma3(self, move_results: List[EventMoveResult]):
        """Sync event moves to MA3 through ShowManager if events are on synced layers."""
        from src.utils.message import Log
        
        
        if not hasattr(self, 'timeline_widget') or not self.timeline_widget:
            return
        
        layer_manager = self.timeline_widget._layer_manager
        
        # Group moves by synced layer
        moves_by_synced_layer: Dict[str, List[EventMoveResult]] = {}
        
        for result in move_results:
            # Check if the new layer is synced
            new_layer = layer_manager.get_layer(result.new_layer_id)
            if new_layer and new_layer.is_synced and new_layer.show_manager_block_id and new_layer.ma3_track_coord:
                layer_key = f"{new_layer.show_manager_block_id}:{new_layer.ma3_track_coord}"
                if layer_key not in moves_by_synced_layer:
                    moves_by_synced_layer[layer_key] = []
                moves_by_synced_layer[layer_key].append(result)
        
        # Sync each group to its ShowManager
        for layer_key, moves in moves_by_synced_layer.items():
            show_manager_block_id, ma3_track_coord = layer_key.split(':', 1)
            # Parse MA3 coord format "tc.tg.track" or "tc.tg.tr"
            tc = tg = tr = None
            try:
                parts = ma3_track_coord.split('.')
                if len(parts) >= 3:
                    tc = int(parts[0])
                    tg = int(parts[1])
                    tr = int(parts[2])
            except Exception:
                tc = tg = tr = None
            
            try:
                # Get event metadata to find MA3 event index
                for move_result in moves:
                    # Parse event ID to get data_item_id and event_index
                    event_id_parts = move_result.event_id.split('_', 2)
                    if len(event_id_parts) < 3:
                        continue
                    
                    data_item_id = f"{event_id_parts[0]}_{event_id_parts[1]}"
                    try:
                        event_index = int(event_id_parts[2])
                    except ValueError:
                        continue
                    
                    # Get event from EventDataItem
                    if data_item_id in self._loaded_event_items:
                        item = self._loaded_event_items[data_item_id]
                        events = item.get_events()
                        if 0 <= event_index < len(events):
                            event = events[event_index]
                            
                            # Check if event has MA3 metadata
                            ma3_coord = event.metadata.get('ma3_coord')
                            if ma3_coord == ma3_track_coord:
                                # Push move directly to MA3 using stored ma3_idx
                                ma3_idx = event.metadata.get('ma3_idx') if hasattr(event, 'metadata') else None
                                if ma3_idx and tc is not None and tg is not None and tr is not None:
                                    try:
                                        ma3_comm = getattr(self.facade, "ma3_communication_service", None) if self.facade else None
                                        if ma3_comm:
                                            ma3_comm.update_event(
                                                tc, tg, tr,
                                                int(ma3_idx),
                                                time=float(move_result.new_time),
                                                cmd=None
                                            )
                                    except Exception:
                                        Log.warning("EditorPanel: Failed to send MA3 UpdateEvent")
                                else:
                                    # This event is from MA3 - sync the move back
                                    # TODO: Send OSC command to MA3 to move the event
                                    # For now, just log
                                    Log.debug(f"EditorPanel: Event move on synced layer - would sync to MA3 track {ma3_track_coord} at time {move_result.new_time}")
                                    
                                    # Find ShowManager block and send move command
                                    # This would use the MA3 event commands to update the event time
                                    # For now, this is a placeholder for the sync logic
            except Exception as e:
                Log.warning(f"EditorPanel: Failed to sync event moves to MA3: {e}")
    
    def _on_events_resized(self, resize_results: List[EventResizeResult]):
        """
        Handle events resized in timeline (new typed API).
        
        Receives already-batched results from MovementController.
        """
        if not resize_results or not self.facade:
            return
        
        counts_by_layer: Dict[str, int] = {}
        for result in resize_results:
            parsed = self._parse_event_id(result.event_id)
            if not parsed:
                continue
            
            data_item_id, event_index = parsed
            
            self.facade.command_bus.execute(UpdateEventInDataItemCommand(
                self.facade,
                data_item_id,
                event_index,
                new_time=result.new_time,
                new_duration=result.new_duration
            ))
            
            # Update in-memory copy
            if data_item_id in self._loaded_event_items:
                item = self._loaded_event_items[data_item_id]
                events = item.get_events()
                if 0 <= event_index < len(events):
                    events[event_index].time = result.new_time
                    events[event_index].duration = result.new_duration
                    layer_name = events[event_index].classification
                    counts_by_layer[layer_name] = counts_by_layer.get(layer_name, 0) + 1

        # Touch local state so DB reflects edit immediately (no execute required)
        if resize_results:
            self._touch_local_state_db(reason="events_resized")
            self._emit_editor_sync_signal("updated", counts_by_layer)
    
    def _on_events_deleted_batch(self, delete_results: list):
        """Handle batch event deletion using event matching.
        
        Uses event metadata (_source_item_id) to correctly identify which
        EventDataItem each event belongs to, then finds and deletes the event
        by matching time/classification (robust to layer structure changes).
        
        This is the unified pathway for all event deletions.
        """
        
        if not delete_results or not self.facade:
            Log.warning(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: No results or no facade. Results: {delete_results}")
            return
        
        count = len(delete_results)
        event_ids = [r.event_id if hasattr(r, 'event_id') else str(r) for r in delete_results]
        Log.debug(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: Received {count} deletion results: {event_ids}")
        
        # Group deletions by data_item_id using event_data.user_data._source_item_id
        # This is reliable because events store their source item ID when loaded
        deletions_by_item: Dict[str, List[Dict[str, Any]]] = {}
        counts_by_layer: Dict[str, int] = {}
        parse_failures = []
        
        for result in delete_results:
            event_id = result.event_id if hasattr(result, 'event_id') else str(result)
            event_data = result.event_data if hasattr(result, 'event_data') else None
            
            if event_data and hasattr(event_data, 'user_data') and event_data.user_data:
                # Extract source info from event metadata (single source of truth)
                source_item_id = event_data.user_data.get('_source_item_id')
                if source_item_id:
                    if source_item_id not in deletions_by_item:
                        deletions_by_item[source_item_id] = []
                    deletions_by_item[source_item_id].append({
                        'time': event_data.time,
                        'classification': event_data.classification,
                        'duration': getattr(event_data, 'duration', 0.0),
                        'event_id': event_id
                    })
                    # Get layer name from layer_id (not classification - classification is "onset", layer name is "Separator1_bass-Onsets")
                    layer_name = None
                    if hasattr(event_data, 'layer_id') and event_data.layer_id:
                        if hasattr(self, 'timeline_widget') and self.timeline_widget:
                            layer_manager = self.timeline_widget._layer_manager
                            layer = layer_manager.get_layer(event_data.layer_id)
                            if layer and layer.name:
                                layer_name = layer.name
                    # Fallback to classification if layer_id not available (legacy events)
                    if not layer_name:
                        layer_name = event_data.classification
                    counts_by_layer[layer_name] = counts_by_layer.get(layer_name, 0) + 1
                    Log.debug(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: {event_id} -> item {source_item_id} (time={event_data.time}, layer={layer_name})")
                    continue
            
            # Fallback to parsing event ID if no user_data (legacy events)
            parsed = self._parse_event_id(event_id)
            if parsed:
                data_item_id, event_index = parsed
                # For legacy events, we still need time/classification - try to get from loaded items
                if data_item_id in self._loaded_event_items:
                    item = self._loaded_event_items[data_item_id]
                    events = item.get_events()
                    if 0 <= event_index < len(events):
                        ev = events[event_index]
                        if data_item_id not in deletions_by_item:
                            deletions_by_item[data_item_id] = []
                        deletions_by_item[data_item_id].append({
                            'time': ev.time,
                            'classification': ev.classification,
                            'duration': ev.duration,
                            'event_id': event_id
                        })
                        # Try to get layer name from event metadata or use classification as fallback
                        layer_name = None
                        if hasattr(ev, 'metadata') and ev.metadata:
                            # Check for _visual_layer_name in metadata (set when events are moved)
                            layer_name = ev.metadata.get('_visual_layer_name')
                        # If still no layer name, use classification (legacy)
                        if not layer_name:
                            layer_name = ev.classification
                        counts_by_layer[layer_name] = counts_by_layer.get(layer_name, 0) + 1
                        Log.debug(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: {event_id} -> item {data_item_id} via fallback (layer={layer_name})")
                        continue
            
            parse_failures.append(event_id)
            Log.warning(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: Could not resolve event: {event_id}")
        
        if parse_failures:
            Log.warning(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: {len(parse_failures)} events failed to resolve: {parse_failures}")
        
        if not deletions_by_item:
            Log.warning("[DELETE DEBUG] EditorPanel._on_events_deleted_batch: No valid deletions after parsing")
            return
        
        # Build (data_item_id, event_index) tuples for BatchDeleteEventsFromDataItemCommand
        # Event index is position in data_item.get_events() flat list
        deletions: List[Tuple[str, int]] = []
        seen: set = set()
        for data_item_id, events_to_delete in deletions_by_item.items():
            data_item = self.facade.data_item_repo.get(data_item_id)
            if not data_item:
                Log.warning(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: Data item {data_item_id} not found")
                continue
            
            from src.shared.domain.entities import EventDataItem
            if not isinstance(data_item, EventDataItem):
                Log.warning(f"[DELETE DEBUG] EditorPanel._on_events_deleted_batch: {data_item_id} is not EventDataItem")
                continue
            
            all_events = data_item.get_events()
            for ev_info in events_to_delete:
                target_time = ev_info['time']
                target_class = ev_info['classification']
                target_id = ev_info.get('event_id')
                
                for idx, event in enumerate(all_events):
                    if (data_item_id, idx) in seen:
                        continue
                    if abs(event.time - target_time) < 0.001 and event.classification == target_class:
                        if target_id and hasattr(event, 'id') and event.id != target_id:
                            continue
                        deletions.append((data_item_id, idx))
                        seen.add((data_item_id, idx))
                        break
        
        if not deletions:
            Log.warning("[DELETE DEBUG] EditorPanel._on_events_deleted_batch: No events resolved for deletion")
            return
        
        # Execute through command bus for undo support
        from src.application.commands import BatchDeleteEventsFromDataItemCommand
        use_macro = len(deletions) > 1
        if use_macro:
            self.facade.command_bus.begin_macro(f"Delete {len(deletions)} Event{'s' if len(deletions) != 1 else ''}")
        self.facade.command_bus.execute(BatchDeleteEventsFromDataItemCommand(self.facade, deletions))
        if use_macro:
            self.facade.command_bus.end_macro()
        
        deleted_count = len(deletions)
        Log.info(f"EditorPanel: Batch deleted {deleted_count} events (undoable)")
        self.set_status_message(f"Deleted {deleted_count} events")

        # Touch local state so DB reflects edit immediately (no execute required)
        self._touch_local_state_db(reason="events_deleted")
        self._emit_editor_sync_signal("deleted", counts_by_layer)
        
        # Emit BlockUpdated so SyncSystemManager can push to MA3
        # source="editor" indicates this is a user edit (not from MA3 sync)
        # Include layer_names so only the changed layers are synced
        from src.application.events.events import BlockUpdated
        if hasattr(self, 'facade') and self.facade and hasattr(self.facade, 'event_bus') and self.facade.event_bus:
            changed_layer_names = list(counts_by_layer.keys())  # All layers that had events deleted
            self.facade.event_bus.publish(BlockUpdated(
                data={
                    "id": self.block_id,
                    "events_updated": True,
                    "source": "editor",
                    "layer_names": changed_layer_names  # List of layer names that changed
                }
            ))
            Log.debug(f"EditorPanel: Published BlockUpdated for events_deleted_batch in layers: {changed_layer_names}")
        
        # Update count
        total = sum(len(item.get_events()) for item in self._loaded_event_items.values())
        self.event_count_label.setText(f"Events: {total}")
    
    def _on_event_created(self, create_result: EventCreateResult):
        """Handle new event created in timeline (new typed API)."""
        
        if not self._loaded_event_items:
            self.set_status_message("Load data first", error=True)
            return
        
        # Resolve target item by layer group_id (source EventDataItem)
        target_item = None
        layer = None
        if create_result.layer_id and hasattr(self.timeline_widget, '_layer_manager'):
            layer = self.timeline_widget._layer_manager.get_layer(create_result.layer_id)
            if layer and layer.group_id and layer.group_id in self._loaded_event_items:
                target_item = self._loaded_event_items[layer.group_id]
        
        if not target_item:
            target_item = list(self._loaded_event_items.values())[0] if self._loaded_event_items else None
        
        if not target_item:
            self.set_status_message("No data item available", error=True)
            return
        
        classification = layer.name if layer else create_result.classification
        if self.facade:
            cmd = AddEventToDataItemCommand(
                self.facade,
                data_item_id=target_item.id,
                time=create_result.time,
                duration=create_result.duration,
                classification=classification
            )
            self.facade.command_bus.execute(cmd)
            
            Log.info(f"EditorPanel: Created event at t={create_result.time:.3f}")
            self.set_status_message("Event created")
            target_item.add_event(create_result.time, classification, create_result.duration, layer_name=classification)
            
            # Reload to get correct ID
            if hasattr(self, "_reload_event_data_items_in_place"):
                self._reload_event_data_items_in_place()
            elif hasattr(self, "refresh"):
                self.refresh()
            self._emit_editor_sync_signal("added", {classification: 1})
            
            # Emit BlockUpdated so SyncSystemManager can push to MA3
            # source="editor" indicates this is a user edit (not from MA3 sync)
            # Include layer_names so only the changed layers are synced
            from src.application.events.events import BlockUpdated
            if hasattr(self.facade, 'event_bus') and self.facade.event_bus:
                changed_layer_names = [classification] if classification else []  # Single layer that changed
                self.facade.event_bus.publish(BlockUpdated(
                    data={
                        "id": self.block_id,
                        "events_updated": True,
                        "source": "editor",
                        "layer_names": changed_layer_names  # List of layer names that changed
                    }
                ))
                Log.debug(f"EditorPanel: Published BlockUpdated for event_created in layer: {classification}")
    
    def _on_event_sliced(self, slice_result: EventSliceResult):
        """Handle event sliced in timeline (split into two events)."""
        
        if not self.facade or not self._loaded_event_items:
            self.set_status_message("Load data first", error=True)
            return
        
        # Parse original event ID to get (data_item_id, event_index)
        parsed = self._parse_event_id(slice_result.original_event_id)
        if not parsed:
            Log.warning(f"EditorPanel: Failed to parse event ID: {slice_result.original_event_id}")
            self.set_status_message("Invalid event ID", error=True)
            return
        
        data_item_id, event_index = parsed
        
        # Verify data item exists
        if data_item_id not in self._loaded_event_items:
            Log.warning(f"EditorPanel: Data item not found: {data_item_id}")
            self.set_status_message("Data item not found", error=True)
            return
        
        # Use macro to combine delete + two adds into single undo operation
        self.facade.command_bus.begin_macro("Slice Event")
        
        try:
            # Delete the original event
            from src.application.commands import BatchDeleteEventsFromDataItemCommand
            delete_cmd = BatchDeleteEventsFromDataItemCommand(self.facade, [(data_item_id, event_index)])
            self.facade.command_bus.execute(delete_cmd)
            
            # Add first event (from start to slice)
            first_cmd = AddEventToDataItemCommand(
                self.facade,
                data_item_id=data_item_id,
                time=slice_result.first_event_data.time,
                duration=slice_result.first_event_data.duration,
                classification=slice_result.first_event_data.classification,
                metadata=slice_result.first_event_data.user_data
            )
            self.facade.command_bus.execute(first_cmd)
            
            # Add second event (from slice to end)
            second_cmd = AddEventToDataItemCommand(
                self.facade,
                data_item_id=data_item_id,
                time=slice_result.second_event_data.time,
                duration=slice_result.second_event_data.duration,
                classification=slice_result.second_event_data.classification,
                metadata=slice_result.second_event_data.user_data
            )
            self.facade.command_bus.execute(second_cmd)
            
            Log.info(f"EditorPanel: Sliced event {slice_result.original_event_id} at {slice_result.slice_time:.3f}s")
            self.set_status_message("Event sliced")
            
        finally:
            self.facade.command_bus.end_macro()
        
        # Touch local state so DB reflects edit immediately
        self._touch_local_state_db(reason="event_sliced")
        deleted_layer = getattr(slice_result.original_event_data, "classification", None) if hasattr(slice_result, "original_event_data") else None
        if not deleted_layer:
            deleted_layer = slice_result.first_event_data.classification
        self._emit_editor_sync_signal("deleted", {deleted_layer: 1})
        self._emit_editor_sync_signal(
            "added",
            {
                slice_result.first_event_data.classification: 1,
                slice_result.second_event_data.classification: 1
            }
        )
        
        # Reload timeline to get new event IDs
        self._on_reload_clicked()
    
    def _get_update_description(self, updates: list) -> str:
        """Get a descriptive name for the update operation."""
        count = len(updates)
        
        # Check what type of changes were made
        has_duration = any('duration' in u for u in updates)
        has_time = any('time' in u for u in updates)
        has_classification = any('classification' in u for u in updates)
        
        if has_duration and not has_time:
            action = "Resize"
        elif has_classification:
            action = "Move"  # Layer change
        else:
            action = "Move"
        
        if count == 1:
            return f"{action} Event"
        else:
            return f"{action} {count} Events"
    
    # ==================== Undo/Redo UI Refresh ====================
    
    def refresh_for_undo(self):
        """
        Refresh timeline after undo/redo operation using in-place updates.
        
        Called by MainWindow when QUndoStack.indexChanged fires.
        Uses smart refresh: updates events in place when possible,
        only does full reload when structure changes.
        """
        
        if not self._loaded_event_items:
            return
        
        Log.debug(f"EditorPanel: Refreshing timeline for undo/redo (smart refresh)")
        
        # DEBUG: Log current state before refresh
        if hasattr(self, 'timeline_widget') and self.timeline_widget:
            scene_event_count = len(self.timeline_widget._scene.get_all_event_items())
            Log.debug(f"[DELETE DEBUG] EditorPanel refresh: Scene has {scene_event_count} events before refresh")
        
        # Phase 3: Get current event data from owned data (not from connections)
        if not hasattr(self.facade, 'data_item_repo') or not self.facade.data_item_repo:
            return
        
        owned_items = self.facade.data_item_repo.list_by_block(self.block_id)
        new_event_items = [item for item in owned_items if isinstance(item, EventDataItem)]
        new_event_items_dict = {item.id: item for item in new_event_items}
        
        # DEBUG: Log data layer state
        total_data_events = sum(len(item.get_events()) for item in new_event_items_dict.values())
        Log.debug(f"[DELETE DEBUG] EditorPanel refresh: Data layer has {total_data_events} events across {len(new_event_items_dict)} items")
        for item_id, item in new_event_items_dict.items():
            Log.debug(f"[DELETE DEBUG] EditorPanel refresh: Data item {item_id} has {len(item.get_events())} events")
        
        # Check if structure changed (sources added/removed)
        current_source_ids = set(self._loaded_event_items.keys())
        new_source_ids = set(new_event_items_dict.keys())
        
        structure_changed = (current_source_ids != new_source_ids)
        
        if structure_changed:
            # Structure changed - need full reload
            Log.debug(f"EditorPanel refresh: Structure changed, reloading owned data")
            self._load_owned_data()  # Phase 3: Load owned data instead of from connections
            return
        
        # Structure unchanged - check if event counts match
        # If counts differ, we need full reload (events added/removed)
        event_count_changed = False
        for item_id in current_source_ids:
            if item_id in new_event_items_dict:
                old_count = len(self._loaded_event_items[item_id].get_events())
                new_count = len(new_event_items_dict[item_id].get_events())
                if old_count != new_count:
                    Log.debug(f"[DELETE DEBUG] EditorPanel refresh: Event count changed for {item_id}: {old_count} -> {new_count}")
                    event_count_changed = True
                    break
        
        if event_count_changed:
            # Event count changed - need full reload
            Log.debug("EditorPanel: Event count changed, doing full reload")
            self._load_owned_data()  # Phase 3: Load owned data
            return
        
        # Structure and count unchanged - use in-place updates
        # Disable viewport updates for batch operation (Qt best practice)
        view = None
        if hasattr(self, 'timeline_widget') and self.timeline_widget:
            view = self.timeline_widget._view
            if view:
                view.setUpdatesEnabled(False)
        
        try:
            # Update only events that actually changed (optimization)
            updated_count = 0
            checked_count = 0
            skipped_not_found = 0
            for item_id, new_item in new_event_items_dict.items():
                if item_id in self._loaded_event_items:
                    new_events = new_item.get_events()
                    
                    # DEBUG: Log event counts
                    Log.debug(f"[DELETE DEBUG] EditorPanel refresh: Data item {item_id} has {len(new_events)} events")
                    
                    # Update timeline events in place
                    for i, new_event in enumerate(new_events):
                        event_id = str(getattr(new_event, "id", "") or "")
                        if not event_id:
                            skipped_not_found += 1
                            if skipped_not_found <= 5:
                                Log.debug(
                                    f"[DELETE DEBUG] EditorPanel refresh: Event at index {i} in {item_id} has no id - skipping"
                                )
                            continue
                        checked_count += 1
                        
                        # Check if event exists in timeline
                        if hasattr(self, 'timeline_widget') and self.timeline_widget:
                            # Get current visual state from timeline
                            current_event = self.timeline_widget.get_event(event_id)
                            if not current_event:
                                skipped_not_found += 1
                                if skipped_not_found <= 5:  # Log first 5 to avoid spam
                                    Log.debug(f"[DELETE DEBUG] EditorPanel refresh: Event {event_id} not found in timeline (was deleted or never existed)")
                                continue
                            
                            # Resolve target layer by NAME (stable across sessions)
                            visual_layer_id = None
                            visual_layer_name = new_event.metadata.get('_visual_layer_name')
                            
                            if visual_layer_name:
                                layer = self.timeline_widget._layer_manager.get_layer_by_name(visual_layer_name)
                                if layer:
                                    visual_layer_id = layer.id
                            
                            # Compare with current state - only update if changed
                            time_changed = abs(current_event.time - new_event.time) > 0.0001
                            duration_changed = abs(current_event.duration - new_event.duration) > 0.0001
                            layer_changed = visual_layer_id is not None and current_event.layer_id != visual_layer_id
                            
                            if not (time_changed or duration_changed or layer_changed):
                                continue  # Skip unchanged events
                            
                            # Use public API for update
                            # Note: update_event accepts user_data, not metadata
                            # Metadata should be passed as user_data if needed
                            if self.timeline_widget.update_event(
                                event_id=event_id,
                                start_time=new_event.time,
                                duration=new_event.duration,
                                layer_id=visual_layer_id
                            ):
                                updated_count += 1
            
            # Update loaded items dict (replace with new items)
            self._loaded_event_items = new_event_items_dict
            
            # Update UI labels (events already updated in place, no need to reload timeline)
            if self._loaded_event_items:
                total_events = sum(len(item.get_events()) for item in self._loaded_event_items.values())
                layers = len(self.timeline_widget.get_layers())
                duration = self.timeline_widget.get_duration()
                
                self.event_count_label.setText(f"Events: {total_events}")
                self.layer_count_label.setText(f"Layers: {layers}")
                self.duration_label.setText(f"Duration: {duration:.1f}s")
            else:
                self.event_count_label.setText("Events: 0")
                self.layer_count_label.setText("Layers: 0")
                self.duration_label.setText("Duration: 0.0s")
            
            if skipped_not_found > 0:
                Log.debug(f"[DELETE DEBUG] EditorPanel refresh: Skipped {skipped_not_found} events not found in timeline")
            Log.debug(f"EditorPanel: Checked {checked_count} events, updated {updated_count} (skipped {checked_count - updated_count} unchanged)")
            
        finally:
            # Re-enable viewport updates (Qt batches all changes)
            if view:
                view.setUpdatesEnabled(True)
    
    # ==================== Cleanup ====================
    
    def cleanup(self):
        """Clean up resources"""
        # Unsubscribe from EventVisualUpdate
        try:
            self.facade.event_bus.unsubscribe("EventVisualUpdate", self._on_event_visual_update)
        except:
            pass
        
        # Cleanup audio player
        if self._audio_player:
            try:
                self._audio_player.cleanup()
            except Exception as e:
                Log.warning(f"EditorPanel: Error cleaning up audio player: {e}")
            finally:
                self._audio_player = None
        
        # Cleanup timeline
        if hasattr(self, 'timeline_widget'):
            self.timeline_widget.cleanup()
        
        super().cleanup() if hasattr(super(), 'cleanup') else None
    
    def closeEvent(self, event):
        """Handle panel close"""
        # Clear selection before closing to prevent stale selections
        if hasattr(self, 'timeline_widget') and self.timeline_widget:
            if hasattr(self.timeline_widget, '_scene'):
                scene = self.timeline_widget._scene
                if hasattr(scene, 'deselect_all'):
                    scene.deselect_all()
                elif hasattr(scene, 'clearSelection'):
                    scene.clearSelection()
            if hasattr(self.timeline_widget, '_event_inspector'):
                inspector = self.timeline_widget._event_inspector
                if hasattr(inspector, 'update_selection'):
                    inspector.update_selection([])  # Clear selection
        
        # Save layer configuration before closing
        self._save_layer_state()
        self.cleanup()
        super().closeEvent(event)
    
    def showEvent(self, event):
        """Handle panel show - no selection clearing here to avoid interfering with user selections"""
        super().showEvent(event)
        # Note: Selection clearing removed - it was interfering with normal event selection.
        # Selection is only cleared on initial widget creation (create_content_widget) and panel close (closeEvent).

    def _get_current_layer_order_keys(self) -> List[LayerKey]:
        """Return current layer order as LayerKey list."""
        if not hasattr(self, 'timeline_widget') or not self.timeline_widget:
            return []
        layer_manager = getattr(self.timeline_widget, "_layer_manager", None)
        if not layer_manager:
            return []
        keys = []
        group_ids = []
        for layer in layer_manager.get_all_layers():
            if layer.name:
                group_key = layer.group_id or layer.group_name
                keys.append(LayerKey(group_name=group_key, name=layer.name))
                group_ids.append({
                    "group_id": getattr(layer, "group_id", None),
                    "group_name": getattr(layer, "group_name", None),
                    "name": getattr(layer, "name", None),
                })
        return keys

    def _reconcile_layer_order(self) -> List[LayerKey]:
        """Reconcile saved order with current layers and persist if needed."""
        layer_group_service = getattr(self, "_layer_group_order_service", None)
        layer_order_service = getattr(self, "_layer_order_service", None)
        if not hasattr(self, 'timeline_widget'):
            return []
        layer_manager = getattr(self.timeline_widget, "_layer_manager", None)
        layers = layer_manager.get_all_layers() if layer_manager else self.timeline_widget.get_layers()
        if layer_group_service:
            return layer_group_service.reconcile_and_save(self.block.id, layers) if self.block else []
        if not layer_order_service:
            return []
        return layer_order_service.reconcile_and_save(self.block.id, layers)

    def _apply_layer_order(self, order: List[LayerKey]) -> None:
        """Apply a saved layer order to the layer manager."""
        if not order or not hasattr(self, 'timeline_widget'):
            return

        layer_manager = getattr(self.timeline_widget, "_layer_manager", None)
        if not layer_manager:
            return

        layer_by_key = {}
        for layer in layer_manager.get_all_layers():
            group_key = layer.group_id or layer.group_name
            layer_by_key[(group_key, layer.name)] = layer

        expected = []
        ordered_layers = []
        for key in order:
            expected.append({"group_name": key.group_name, "name": key.name})
            layer = layer_by_key.get((key.group_name, key.name))
            if layer:
                ordered_layers.append(layer)

        # Append any remaining layers not in order
        remaining_layers = [l for l in layer_manager.get_all_layers() if l not in ordered_layers]
        ordered_layers.extend(remaining_layers)

        for index, layer in enumerate(ordered_layers):
            layer_manager.reorder_layer(layer.id, index)

        actual = [
            {"group_name": layer.group_name, "name": layer.name}
            for layer in layer_manager.get_all_layers()
        ]
        Log.debug(f"EditorPanel: Applied layer order (expected={expected}, actual={actual})")
    
    def _save_layer_state(self):
        """
        Save layer configuration (presentation overrides) to UI state.

        EventDataItem is the single source of truth for group/layer existence.
        UI state stores only overrides keyed by (group_id, layer_name).
        Any stale overrides are treated as errors (fail loud).
        """
        if not self.block or not hasattr(self, 'timeline_widget'):
            return
        
        try:
            layers = self.timeline_widget.get_layers()
            if not getattr(self, "_loaded_event_items", None):
                return

            allowed_keys = set()
            for item in self._loaded_event_items.values():
                if not isinstance(item, EventDataItem):
                    continue
                group_id, _ = self._get_group_identity_for_item(item)
                for event_layer in item.get_layers():
                    if event_layer.name:
                        allowed_keys.add((group_id, event_layer.name))

            existing_data = {}
            stale_existing_keys = []
            result = self.facade.get_ui_state(
                state_type='editor_layers',
                entity_id=self.block.id
            )
            if result.success and result.data:
                for saved in result.data.get('layers', []):
                    if not isinstance(saved, dict):
                        continue
                    name = saved.get('name')
                    group_id = saved.get('group_id')
                    if not name or not group_id:
                        raise ValueError(f"Invalid editor_layers override entry: {saved}")
                    layer_key = (group_id, name)
                    if allowed_keys and layer_key not in allowed_keys:
                        stale_existing_keys.append(layer_key)
                        continue
                    existing_data[layer_key] = saved
            # Note: stale_existing_keys handling was removed during debug cleanup

            for layer in layers:
                if not layer.name:
                    raise ValueError("Timeline layer missing name")
                if not layer.group_id:
                    raise ValueError(f"Timeline layer missing group_id: {layer.name}")
                layer_key = (layer.group_id, layer.name)
                if allowed_keys and layer_key not in allowed_keys:
                    raise ValueError(f"Timeline layer not in EventDataItem set: {layer_key}")
                layer_state = {
                    'name': layer.name,
                    'height': layer.height,
                    'color': layer.color,
                    'visible': layer.visible,
                    'locked': layer.locked,
                    'group_id': layer.group_id,
                    'group_name': layer.group_name,
                    'group_index': layer.group_index,
                    'is_synced': getattr(layer, 'is_synced', False),
                    'show_manager_block_id': getattr(layer, 'show_manager_block_id', None),
                    'ma3_track_coord': getattr(layer, 'ma3_track_coord', None),
                }
                if getattr(layer, 'derived_from_ma3', False):
                    layer_state['derived_from_ma3'] = True
                existing_data[layer_key] = layer_state

            layer_data = list(existing_data.values())

            self.facade.set_ui_state(
                state_type='editor_layers',
                entity_id=self.block.id,
                data={'layers': layer_data}
            )
            Log.debug(f"EditorPanel: Saved {len(layer_data)} layer override(s) for block {self.block.id}")
        except Exception as e:
            Log.warning(f"EditorPanel: Failed to save layer state: {e}")
    
    
    def _restore_layer_state(self):
        """Restore layer configuration from project UI state.
        
        Applies saved presentation overrides to existing layers only.
        EventDataItem defines group/layer existence; UI state never creates layers.
        """
        if not self.block or not hasattr(self, 'timeline_widget'):
            return
        
        try:
            result = self.facade.get_ui_state(
                state_type='editor_layers',
                entity_id=self.block.id
            )
            
            if not result.success or not result.data:
                return
            
            layer_data = result.data.get('layers', [])
            if not layer_data:
                return
            
            # Get current layers indexed by (group_id, name) for unique matching
            current_layers = self.timeline_widget.get_layers()
            layer_by_key = {}
            for l in current_layers:
                if not l.group_id:
                    raise ValueError(f"Timeline layer missing group_id: {l.name}")
                layer_key = (l.group_id, l.name)
                layer_by_key[layer_key] = l
            
            # Build allowed keys from loaded EventDataItems to avoid restoring stale layers
            from src.shared.domain.entities import EventDataItem
            allowed_keys = set()

            for item in self._loaded_event_items.values():
                if not isinstance(item, EventDataItem):
                    continue
                group_id, _ = self._get_group_identity_for_item(item)
                for event_layer in item.get_layers():
                    if event_layer.name:
                        allowed_keys.add((group_id, event_layer.name))
            
            restored_count = 0
            
            filtered_layer_data = []
            stale_override_keys = []
            for saved_layer in layer_data:
                name = saved_layer.get('name')
                if not name:
                    raise ValueError(f"Invalid editor_layers override entry: {saved_layer}")

                group_id = saved_layer.get('group_id')
                if not group_id:
                    raise ValueError(f"Invalid editor_layers override entry: {saved_layer}")
                layer_key = (group_id, name)

                if allowed_keys and layer_key not in allowed_keys:
                    stale_override_keys.append(layer_key)
                    continue
                if layer_key not in layer_by_key:
                    raise ValueError(f"Missing timeline layer for override: {layer_key}")

                filtered_layer_data.append(saved_layer)

            if stale_override_keys:
                try:
                    self.facade.set_ui_state(
                        state_type='editor_layers',
                        entity_id=self.block.id,
                        data={'layers': filtered_layer_data},
                    )
                except Exception:
                    pass

            for saved_layer in filtered_layer_data:
                name = saved_layer.get('name')
                group_id = saved_layer.get('group_id')
                layer_key = (group_id, name)

                layer = layer_by_key[layer_key]
                
                # Apply all saved properties including group properties
                self.timeline_widget._layer_manager.update_layer(
                    layer.id,
                    height=saved_layer.get('height'),
                    color=saved_layer.get('color'),
                    visible=saved_layer.get('visible', True),
                    locked=saved_layer.get('locked', False),
                    group_id=group_id,
                    group_name=saved_layer.get('group_name'),
                    group_index=saved_layer.get('group_index'),
                )
            
                # Restore synced layer properties if this is a synced layer
                if saved_layer.get('is_synced'):
                    # Directly update the layer object to set synced properties
                    layer.is_synced = True
                    layer.show_manager_block_id = saved_layer.get('show_manager_block_id')
                    layer.ma3_track_coord = saved_layer.get('ma3_track_coord')
                
                # Restore derived_from_ma3 flag
                if saved_layer.get('derived_from_ma3'):
                    layer.derived_from_ma3 = True
                restored_count += 1
            
            if restored_count > 0:
                Log.debug(f"EditorPanel: Restored {restored_count} layer override(s) for block {self.block.id}")

        except Exception as e:
            Log.warning(f"EditorPanel: Failed to restore layer state: {e}")

    
    # ==================== Stale Data Handling ====================
    
    def _show_stale_data_warning(self):
        """Show warning banner when stale audio data is detected"""
        if not self._has_stale_audio or not self._stale_audio_errors:
            return
        
        # Build warning message
        error_count = len(self._stale_audio_errors)
        if error_count == 1:
            message = f"<b>Audio file not found:</b> {self._stale_audio_errors[0]}<br>"
        else:
            message = f"<b>{error_count} audio files not found:</b><br>"
            for error in self._stale_audio_errors[:3]:  # Show first 3
                message += f"• {error}<br>"
            if error_count > 3:
                message += f"• ... and {error_count - 3} more<br>"
        
        message += "<br>Events are displayed but audio playback is unavailable. Clear stale events or reconnect audio sources."
        
        self._warning_message.setText(message)
        self._warning_banner.setVisible(True)
        
        Log.warning(f"EditorPanel: Stale data warning shown - {error_count} missing audio file(s)")
    
    def _on_clear_stale_data(self):
        """Handle 'Clear Stale Events' button click"""
        if not self._has_stale_audio:
            return
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Clear Stale Events",
            "This will remove all events whose source audio files are missing.\n\n"
            "This action cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            from src.utils.tools import validate_audio_data_item
            
            # Find events with missing audio
            events_to_remove = []
            
            for item_id, event_item in self._loaded_event_items.items():
                # Check if this event item references missing audio
                source_audio_id = event_item.metadata.get('_source_audio_id')
                if source_audio_id and source_audio_id in self._loaded_audio_items:
                    audio_item = self._loaded_audio_items[source_audio_id]
                    is_valid, _ = validate_audio_data_item(audio_item)
                    if not is_valid:
                        events_to_remove.append(item_id)
            
            if not events_to_remove:
                # No events directly linked to stale audio, but we have stale audio
                # This means events exist but their audio is missing
                # Clear all events as they're orphaned
                events_to_remove = list(self._loaded_event_items.keys())
            
            # Remove stale event items
            for item_id in events_to_remove:
                if item_id in self._loaded_event_items:
                    # Delete from database
                    self.facade.data_item_repo.delete(item_id)
                    # Remove from loaded items
                    del self._loaded_event_items[item_id]
                    Log.info(f"EditorPanel: Removed stale event item: {item_id}")
            
            # Clear timeline
            if hasattr(self, 'timeline_widget'):
                self.timeline_widget.clear_all_events()
            
            # Reset stale data flags
            self._has_stale_audio = False
            self._stale_audio_errors = []
            
            # Hide warning banner
            self._warning_banner.setVisible(False)
            
            # Update labels
            self.info_label.setText(f"Removed {len(events_to_remove)} stale event item(s)")
            self.event_count_label.setText("Events: 0")
            self.layer_count_label.setText("Layers: 0")
            
            # Touch local state
            self._touch_local_state_db(reason="cleared_stale_events")
            
            Log.info(f"EditorPanel: Cleared {len(events_to_remove)} stale event item(s)")
            
        except Exception as e:
            Log.error(f"EditorPanel: Failed to clear stale data: {e}")
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to clear stale data: {str(e)}"
            )
    
    # ==================== Event Filtering (Processes Improvement Area) ====================
    
    def _on_open_filter_dialog(self):
        """Open event filter dialog"""
        try:
            dialog = EventFilterDialog(self.block_id, self.facade, self)
            if dialog.exec():
                # Filter was applied - reload events to apply filter
                self._reload_events_with_filter()
        except Exception as e:
            Log.error(f"EditorPanel: Failed to open filter dialog: {e}")
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to open filter dialog: {str(e)}"
            )
    
    def _apply_event_filters(self, event_items: List[EventDataItem]) -> List[EventDataItem]:
        """
        Apply event filters to a list of EventDataItems.
        
        Part of the "processes" improvement area.
        Creates filtered copies of EventDataItems based on the block's filter configuration.
        
        Args:
            event_items: List of EventDataItems to filter
            
        Returns:
            List of filtered EventDataItems (new instances with filtered events)
        """
        # Get block to load filter
        block_result = self.facade.describe_block(self.block_id)
        if not block_result.success or not block_result.data:
            return event_items
        
        block = block_result.data
        filter = self._event_filter_manager.get_filter_for_block(block)
        
        if not filter or not filter.enabled:
            # No filter or filter disabled - return original items
            Log.info("EditorPanel: Event filter disabled or missing")
            return event_items
        
        # Apply filter to each event item
        filtered_items = []
        for item in event_items:
            filtered_item = self._event_filter_manager.filter_event_data_item(item, filter, block)
            filtered_items.append(filtered_item)
            try:
                raw_count = len(item.get_events())
                filtered_count = len(filtered_item.get_events())
                is_sync = self._is_sync_layer_item(item)
                Log.debug(
                    "EditorPanel: Event filter result "
                    f"(item='{item.name}', sync={is_sync}, raw={raw_count}, filtered={filtered_count})"
                )
            except Exception:
                pass
        
        return filtered_items
    
    def _reload_events_with_filter(self):
        """Reload events from local state with filters applied"""
        try:
            # Get current event items from loaded state
            event_items = list(self._loaded_event_items.values())
            
            if not event_items:
                return
            
            # Apply filters
            filtered_event_items = self._apply_event_filters(event_items)
            
            # Reload into timeline
            # Use existing layers only - never auto-create from classifications
            self.timeline_widget.set_events_from_data_items(filtered_event_items)
            
            # Update event count
            total_events = sum(len(item.get_events()) for item in filtered_event_items)
            self.event_count_label.setText(f"Events: {total_events}")
            
            # Update status
            self.set_status_message(f"Applied filter: {total_events} events visible")
            
            Log.info(f"EditorPanel: Reloaded {total_events} filtered events")
            
        except Exception as e:
            Log.error(f"EditorPanel: Failed to reload events with filter: {e}")
            self.set_status_message("Failed to apply filter", error=True)

