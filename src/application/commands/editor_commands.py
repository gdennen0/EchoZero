"""
Editor Block Commands

Standardized API commands for Editor block operations.
These commands handle all Editor complexity internally (layers, EventDataItems, events, UI state, UI refreshing).

External blocks (like ShowManager) send simple commands via command bus,
and Editor handles everything under the hood.
"""
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from .base_command import EchoZeroCommand


if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade

class EditorCreateLayerCommand(EchoZeroCommand):
    """
    Create a layer in an Editor block (undoable).
    
    Redo: Creates layer entry in UI state
    Undo: Removes the created layer
    
    Handles:
    - Creating layer entry in UI state (editor_layers)
    - Setting default properties (height, color, visibility)
    - Handling duplicate names (adds suffix if needed)
    - Triggering UI refresh
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        layer_name: Name for the layer
        properties: Optional layer properties (height, color, visible, locked)
    """
    
    COMMAND_TYPE = "editor.create_layer"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        layer_name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(facade, f"Create Layer: {layer_name}")
        
        self._block_id = block_id
        self._layer_name = layer_name
        self._properties = properties or {}
        
        # Track created layer for undo
        self._created_layer_name: Optional[str] = None
        self._was_duplicate: bool = False
        self._is_synced: bool = self._properties.get('is_synced', False)
        self._show_manager_block_id: Optional[str] = self._properties.get('show_manager_block_id')
        self._ma3_track_coord: Optional[str] = self._properties.get('ma3_track_coord')
    
    def redo(self):
        """Create the layer."""
        if not self._facade.ui_state_repo:
            self._log_error("UI state repository not available")
            return
        
        # Get current layers from UI state
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        
        existing_layers = []
        if result.success and result.data:
            existing_layers = result.data.get('layers', [])
        
        # Check if layer already exists
        existing_layer_names = {layer.get('name') for layer in existing_layers if layer.get('name')}
        
        # Handle duplicate names
        layer_name = self._layer_name
        if layer_name in existing_layer_names:
            # Add suffix to make unique
            counter = 1
            original_name = layer_name
            while layer_name in existing_layer_names:
                layer_name = f"{original_name}_{counter}"
                counter += 1
            self._was_duplicate = True
        else:
            self._was_duplicate = False
        
        self._created_layer_name = layer_name
        
        # Get default layer height from TimelineWidget settings if not specified
        default_height = self._properties.get('height')
        if default_height is None:
            # Get default from TimelineWidget settings via preferences repository
            try:
                if hasattr(self._facade, 'preferences_repo') and self._facade.preferences_repo:
                    from ui.qt_gui.widgets.timeline.settings.storage import TimelineSettingsManager
                    timeline_settings = TimelineSettingsManager(preferences_repo=self._facade.preferences_repo)
                    default_height = timeline_settings.default_layer_height
                else:
                    # Fallback to default if preferences repo not available
                    default_height = 40  # TimelineSettings default
            except Exception:
                # Fallback if settings manager fails
                default_height = 40
        
        # Create layer entry with default properties
        layer_data = {
            'name': layer_name,
            'height': default_height,
            'color': self._properties.get('color'),
            'visible': self._properties.get('visible', True),
            'locked': self._properties.get('locked', False),
            # Group properties for visual grouping (e.g., MA3 timecode pool)
            'group_id': self._properties.get('group_id'),
            'group_name': self._properties.get('group_name'),
            'group_index': self._properties.get('group_index'),
        }
        
        # Add synced properties if this is a synced layer
        if self._is_synced:
            layer_data['is_synced'] = True
            if self._show_manager_block_id:
                layer_data['show_manager_block_id'] = self._show_manager_block_id
            if self._ma3_track_coord:
                layer_data['ma3_track_coord'] = self._ma3_track_coord
        
        # Add derived_from_ma3 flag if set
        if self._properties.get('derived_from_ma3'):
            layer_data['derived_from_ma3'] = True
        
        from src.utils.message import Log
        import traceback
        stack = traceback.extract_stack()
        caller_info = f"{stack[-2].filename}:{stack[-2].lineno}" if len(stack) >= 2 else "unknown"
        Log.info(f"[LAYER_CREATE] EditorCreateLayerCommand creating layer '{layer_name}' from {caller_info}")
        Log.info(f"[LAYER_CREATE]   group_name='{layer_data.get('group_name')}', group_id='{layer_data.get('group_id')}'")
        
        # Add to layers list
        existing_layers.append(layer_data)
        
        # Save to UI state
        result = self._facade.set_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id,
            data={'layers': existing_layers}
        )
        
        if not result.success:
            self._log_error(f"Failed to create layer: {result.message}")
            return

        # Persist layer order in repo (append to group order)
        try:
            layer_order_service = getattr(self._facade, "layer_order_service", None)
            if layer_order_service:
                layer_order_service.add_layer(
                    self._block_id,
                    layer_data.get('group_name'),
                    layer_name,
                    is_synced=self._is_synced
                )
        except Exception as e:
            Log.warning(f"EditorCreateLayerCommand: Failed to persist layer order: {e}")
        
        # Ensure layer is immediately restored in TimelineWidget if panel is open
        # This is critical: events will be added immediately after layer creation,
        # so we need the TimelineWidget layer to exist NOW, not later via async event
        self._ensure_layer_restored_in_timeline()
        
        # Trigger UI refresh (async, for other UI updates)
        try:
            from src.application.events.event_bus import EventBus
            from src.application.events.block_events import BlockUpdated
            event_bus = EventBus()
            event_bus.publish(BlockUpdated(
                project_id=self._facade.current_project_id,
                data={
                    "id": self._block_id,
                    "layers_updated": True,
                    "layer_name": layer_name
                }
            ))
        except ImportError:
            pass
    
    def _ensure_layer_restored_in_timeline(self):
        """
        Ensure the layer is immediately restored in TimelineWidget if EditorPanel is open.
        
        This is critical because AddSyncedMA3TrackCommand will immediately try to add events
        after creating the layer. If the layer doesn't exist in TimelineWidget yet, events
        won't be added to the correct layer.
        """
        from src.utils.message import Log
        
        try:
            # Try to find and restore layer in open EditorPanel
            # We use Qt's QApplication to find open widgets
            from PyQt6.QtWidgets import QApplication
            from ui.qt_gui.block_panels.editor_panel import EditorPanel
            
            app = QApplication.instance()
            if not app:
                return  # No Qt app running, skip restoration (will happen on panel open)
            
            # Find EditorPanel for this block_id
            for widget in app.allWidgets():
                if isinstance(widget, EditorPanel) and widget.block_id == self._block_id:
                    # Panel is open - restore layer state immediately
                    if hasattr(widget, '_restore_layer_state'):
                        Log.debug(f"EditorCreateLayerCommand: Immediately restoring layer in open EditorPanel for block {self._block_id}")
                        widget._restore_layer_state()

                        # Reapply layer order after restore
                        if hasattr(widget, "_reconcile_layer_order") and hasattr(widget, "_apply_layer_order"):
                            order = widget._reconcile_layer_order()
                            if order:
                                widget._apply_layer_order(order)
                        
                        # If this is a synced layer, also mark it in TimelineWidget
                        if self._is_synced and hasattr(widget, 'timeline_widget'):
                            timeline_widget = widget.timeline_widget
                            if hasattr(timeline_widget, '_layer_manager'):
                                layer_manager = timeline_widget._layer_manager
                                # Find the layer we just created
                                for layer in layer_manager.get_all_layers():
                                    if layer.name == self._created_layer_name:
                                        # Update layer to mark as synced
                                        layer_manager.update_layer(
                                            layer.id,
                                            check_synced_restrictions=False,  # Skip restrictions during creation
                                            # Note: We can't directly set is_synced, need to update the layer object
                                        )
                                        # Directly set synced properties on the layer object
                                        layer.is_synced = True
                                        layer.show_manager_block_id = self._show_manager_block_id
                                        layer.ma3_track_coord = self._ma3_track_coord
                                        break
                    return
            
            Log.debug(f"EditorCreateLayerCommand: EditorPanel not open for block {self._block_id}, layer will be restored when panel opens")
        except Exception as e:
            # Don't fail if we can't restore - layer will be restored when panel opens or via BlockUpdated event
            Log.debug(f"EditorCreateLayerCommand: Could not immediately restore layer (non-critical): {e}")
    
    def undo(self):
        """Remove the created layer."""
        if not self._created_layer_name:
            return
        
        if not self._facade.ui_state_repo:
            return
        
        # Get current layers
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        
        if not result.success or not result.data:
            return
        
        layers = result.data.get('layers', [])
        
        # Remove the layer
        layers = [layer for layer in layers if layer.get('name') != self._created_layer_name]
        
        # Save back to UI state
        self._facade.set_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id,
            data={'layers': layers}
        )
        
        # Trigger UI refresh
        try:
            from src.application.events.event_bus import EventBus
            from src.application.events.block_events import BlockUpdated
            event_bus = EventBus()
            event_bus.publish(BlockUpdated(
                project_id=self._facade.current_project_id,
                data={
                    "id": self._block_id,
                    "layers_updated": True
                }
            ))
        except ImportError:
            pass
    
    @property
    def created_layer_name(self) -> Optional[str]:
        """Get the name of the created layer (after redo, may differ from input if duplicate)."""
        return self._created_layer_name

class EditorAddEventsCommand(EchoZeroCommand):
    """
    Add events to Editor block (undoable).
    
    Handles all Editor complexity internally:
    - Get or create EventDataItem (with source metadata)
    - Create/update layers if needed (based on event classifications)
    - Add events to EventDataItem using AddEventToDataItemCommand
    - Trigger UI refresh
    - All operations grouped in macro
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        events: List of event dicts (each with time, duration, classification, metadata)
        source: Optional source identifier (e.g., "ma3" for MA3 events)
    """
    
    COMMAND_TYPE = "editor.add_events"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        events: List[Dict[str, Any]],
        source: Optional[str] = None
    ):
        event_count = len(events)
        desc = f"Add {event_count} event{'s' if event_count != 1 else ''} to Editor"
        super().__init__(facade, desc)
        
        self._block_id = block_id
        self._events = events
        self._source = source or "external"
    
    def redo(self):
        """Add events to Editor block."""
        from src.shared.domain.entities import EventDataItem
        
        # Get Editor block
        result = self._facade.describe_block(self._block_id)
        if not result.success or not result.data:
            self._log_error(f"Editor block not found: {self._block_id}")
            return
        
        editor_block = result.data
        
        # Begin macro to group all operations
        self._facade.command_bus.begin_macro(self.text())
        
        try:
            # Get or create EventDataItem for this source
            if not self._facade.data_item_repo:
                self._log_error("Data item repository not available")
                return
            
            existing_items = self._facade.data_item_repo.list_by_block(self._block_id)
            
            ma3_event_item = None
            for item in existing_items:
                if isinstance(item, EventDataItem) and item.metadata.get("source") == self._source:
                    ma3_event_item = item
                    break
            
            data_item_id: Optional[str] = None
            
            # Determine if these events represent a synced MA3 layer
            is_sync_source = self._source in ("ma3", "ma3_sync")
            has_sync_flag = any(
                (evt.get("metadata") or {}).get("_synced_from_ma3") is True
                for evt in self._events
            )
            is_synced_layer = is_sync_source or has_sync_flag
            show_manager_block_id = None
            sync_group_name = None
            sync_group_id = None
            if is_synced_layer:
                for evt in self._events:
                    metadata = evt.get("metadata") or {}
                    show_manager_block_id = (
                        metadata.get("_show_manager_block_id")
                        or metadata.get("show_manager_block_id")
                    )
                    if not sync_group_name:
                        coord = metadata.get("_ma3_track_coord") or metadata.get("ma3_track_coord") or metadata.get("ma3_coord")
                        if isinstance(coord, str):
                            coord_lower = coord.lower()
                            if coord_lower.startswith("tc"):
                                num = ""
                                for ch in coord_lower[2:]:
                                    if ch.isdigit():
                                        num += ch
                                    else:
                                        break
                                if num:
                                    sync_group_name = f"TC {num}"
                                    sync_group_id = f"tc_{num}"
                            elif coord_lower and coord_lower[0].isdigit():
                                head = coord_lower.split(".", 1)[0]
                                if head.isdigit():
                                    sync_group_name = f"TC {head}"
                                    sync_group_id = f"tc_{head}"
                    if show_manager_block_id:
                        break

            if not ma3_event_item:
                # Create EventDataItem using command
                from src.application.commands.data_item_commands import CreateEventDataItemCommand
                create_cmd = CreateEventDataItemCommand(
                    facade=self._facade,
                    block_id=self._block_id,
                    name=f"{editor_block.name}_{self._source}_events",
                    metadata={
                        "source": self._source,
                        "output_port": "events",
                        "_synced_from_ma3": True if is_synced_layer else False,
                        "_show_manager_block_id": show_manager_block_id,
                        "show_manager_block_id": show_manager_block_id,
                        "group_name": sync_group_name,
                        "group_id": sync_group_id,
                    }
                )
                self._facade.command_bus.execute(create_cmd)
                data_item_id = create_cmd.created_data_item_id
                if not data_item_id:
                    self._log_error("Failed to create EventDataItem - no ID returned")
                    return
            else:
                data_item_id = ma3_event_item.id
                if is_synced_layer:
                    updated_meta = False
                    if not ma3_event_item.metadata.get("_synced_from_ma3"):
                        ma3_event_item.metadata["_synced_from_ma3"] = True
                        updated_meta = True
                    if show_manager_block_id and not ma3_event_item.metadata.get("show_manager_block_id"):
                        ma3_event_item.metadata["_show_manager_block_id"] = show_manager_block_id
                        ma3_event_item.metadata["show_manager_block_id"] = show_manager_block_id
                        updated_meta = True
                    if sync_group_name and not ma3_event_item.metadata.get("group_name"):
                        ma3_event_item.metadata["group_name"] = sync_group_name
                        updated_meta = True
                    if sync_group_id and not ma3_event_item.metadata.get("group_id"):
                        ma3_event_item.metadata["group_id"] = sync_group_id
                        updated_meta = True
                    if updated_meta:
                        self._facade.data_item_repo.update(ma3_event_item)
            
            # Get current layers - events can ONLY be added to existing layers
            layer_result = self._facade.get_ui_state(
                state_type='editor_layers',
                entity_id=self._block_id
            )
            
            existing_layer_names = set()
            if layer_result.success and layer_result.data:
                existing_layers = layer_result.data.get('layers', [])
                existing_layer_names = {layer.get('name') for layer in existing_layers if layer.get('name')}
            
            # Validate that all event classifications map to existing layers
            # Events can ONLY be added to existing layers - never auto-create
            missing_layers = set()
            for event_dict in self._events:
                classification = event_dict.get("classification")
                if classification and classification not in existing_layer_names:
                    missing_layers.add(classification)
            
            if missing_layers:
                # Log warning but continue - events will be added but won't appear on a layer
                from src.utils.message import Log
                Log.warning(
                    f"EditorAddEventsCommand: Events have classifications for non-existent layers: {missing_layers}. "
                    f"Events will be added but won't appear until layers are created. "
                    f"Existing layers: {existing_layer_names}"
                )
            
            # Add events using AddEventToDataItemCommand
            from src.application.commands.data_item_commands import AddEventToDataItemCommand
            for event_dict in self._events:
                # Get layer_name (required) - fall back to classification if not set
                layer_name = event_dict.get("layer_name") or event_dict.get("classification") or "event"
                add_cmd = AddEventToDataItemCommand(
                    facade=self._facade,
                    data_item_id=data_item_id,
                    time=event_dict.get("time", 0.0),
                    duration=event_dict.get("duration", 0.0),
                    classification=event_dict.get("classification", "event"),
                    metadata=event_dict.get("metadata", {}),
                    layer_name=layer_name
                )
                self._facade.command_bus.execute(add_cmd)
            
            # Trigger UI refresh (macro end will handle it)
        finally:
            # End macro
            self._facade.command_bus.end_macro()
            
            # Trigger UI refresh after macro completes
            try:
                from src.application.events.event_bus import EventBus
                from src.application.events.block_events import BlockUpdated
                event_bus = EventBus()
                event_bus.publish(BlockUpdated(
                    project_id=self._facade.current_project_id,
                    data={
                        "id": self._block_id,
                        "name": editor_block.name,
                        "events_updated": True,
                        "events_count": len(self._events)
                    }
                ))
            except ImportError:
                pass
    
    def undo(self):
        """Undo is handled by the macro - individual commands handle their own undo."""
        pass

class ApplyLayerSnapshotCommand(EchoZeroCommand):
    """
    Apply a full snapshot of events to a single Editor layer (undoable).
    
    Redo: Replaces all events in the layer with provided snapshot
    Undo: Restores the original layer events
    
    Args:
        facade: ApplicationFacade instance
        data_item_id: EventDataItem ID to update
        layer_name: Target layer name
        events: List of event dicts (time, duration, classification, metadata, id)
    """
    
    COMMAND_TYPE = "editor.apply_layer_snapshot"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        data_item_id: str,
        layer_name: str,
        events: List[Dict[str, Any]]
    ):
        desc = f"Apply Snapshot: {layer_name}"
        super().__init__(facade, desc)
        
        self._data_item_id = data_item_id
        self._layer_name = layer_name
        self._events = events or []
        
        self._original_events: Optional[List[Dict[str, Any]]] = None
        self._layer_existed: Optional[bool] = None
        self._original_layer_metadata: Optional[Dict[str, Any]] = None
        self._original_layer_id: Optional[str] = None
    
    def redo(self):
        """Replace layer events with snapshot."""
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        from src.shared.domain.entities.event_data_item import Event
        from src.shared.domain.entities.event_layer import EventLayer
        
        if not self._facade.data_item_repo:
            self._log_error("Data item repository not available")
            return
        
        data_item = self._facade.data_item_repo.get(self._data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            self._log_error(f"EventDataItem not found: {self._data_item_id}")
            return
        
        layer = data_item.get_layer_by_name(self._layer_name)
        
        if self._original_events is None:
            self._layer_existed = bool(layer)
            if layer:
                self._original_events = [event.to_dict() for event in layer.get_events()]
                self._original_layer_metadata = (layer.metadata or {}).copy()
                self._original_layer_id = layer.id
            else:
                self._original_events = []
        
        # SINGLE SOURCE OF TRUTH: Capture existing event metadata BEFORE clearing
        # This preserves user-set fields like render_as_marker during MA3 sync
        existing_events_by_ma3_idx = {}
        existing_events_by_id = {}
        if layer and hasattr(layer, 'events'):
            # Index existing events by _ma3_idx (for MA3 sync matching) and by ID
            for existing_event in layer.events:
                existing_metadata = getattr(existing_event, 'metadata', {}) or {}
                ma3_idx = existing_metadata.get('_ma3_idx')
                if ma3_idx is not None:
                    existing_events_by_ma3_idx[ma3_idx] = existing_event
                existing_events_by_id[existing_event.id] = existing_event
        
        if not layer:
            try:
                layer = EventLayer(name=self._layer_name)
                data_item.add_layer(layer)
            except Exception as e:
                Log.error(f"ApplyLayerSnapshotCommand: Failed to create layer '{self._layer_name}': {e}")
                return
        
        layer.clear_events()
        
        # Create events, preserving existing metadata (including render_as_marker)
        for event_dict in self._events:
            try:
                time_val = float(event_dict.get("time", 0.0))
            except (ValueError, TypeError):
                time_val = 0.0
            try:
                duration_val = float(event_dict.get("duration", 0.0) or 0.0)
            except (ValueError, TypeError):
                duration_val = 0.0
            
            # SINGLE SOURCE OF TRUTH: Preserve existing event metadata from database
            new_metadata = event_dict.get("metadata", {})
            event_id = event_dict.get("id")
            
            # Try to find existing event by _ma3_idx first (MA3 sync matching)
            ma3_idx = new_metadata.get('_ma3_idx')
            existing_event = None
            if ma3_idx is not None and ma3_idx in existing_events_by_ma3_idx:
                existing_event = existing_events_by_ma3_idx[ma3_idx]
            elif event_id and event_id in existing_events_by_id:
                existing_event = existing_events_by_id[event_id]
            
            
            # Merge existing metadata (preserves render_as_marker) with new metadata (sync fields)
            if existing_event:
                existing_metadata = getattr(existing_event, 'metadata', {}) or {}
                # Merge: existing metadata first (preserves render_as_marker), then new metadata (sync fields)
                merged_metadata = dict(existing_metadata)
                merged_metadata.update(new_metadata)
                new_metadata = merged_metadata
                # Preserve the existing event ID to maintain database consistency
                event_id = existing_event.id
                
            
            event_obj = Event(
                time=time_val,
                classification=event_dict.get("classification", self._layer_name),
                duration=duration_val,
                metadata=new_metadata,  # Merged metadata preserving render_as_marker
                id=event_id
            )
            layer.add_event(event_obj)
        
        data_item.event_count = sum(len(l.events) for l in data_item.get_layers())
        self._facade.data_item_repo.update(data_item)
    
    def undo(self):
        """Restore original layer events."""
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        from src.shared.domain.entities.event_data_item import Event
        from src.shared.domain.entities.event_layer import EventLayer
        
        if not self._facade.data_item_repo or self._original_events is None:
            return
        
        data_item = self._facade.data_item_repo.get(self._data_item_id)
        if not data_item or not isinstance(data_item, EventDataItem):
            Log.warning(f"ApplyLayerSnapshotCommand: EventDataItem not found for undo: {self._data_item_id}")
            return
        
        layer = data_item.get_layer_by_name(self._layer_name)
        
        if not self._layer_existed:
            if layer:
                data_item.remove_layer(self._layer_name)
        else:
            if not layer:
                try:
                    layer = EventLayer(name=self._layer_name, id=self._original_layer_id)
                    data_item.add_layer(layer)
                except Exception as e:
                    Log.warning(f"ApplyLayerSnapshotCommand: Failed to recreate layer '{self._layer_name}': {e}")
                    return
            layer.clear_events()
            if self._original_layer_metadata is not None:
                layer.metadata = self._original_layer_metadata.copy()
            for event_dict in self._original_events:
                try:
                    time_val = float(event_dict.get("time", 0.0))
                except (ValueError, TypeError):
                    time_val = 0.0
                try:
                    duration_val = float(event_dict.get("duration", 0.0) or 0.0)
                except (ValueError, TypeError):
                    duration_val = 0.0
                event_obj = Event(
                    time=time_val,
                    classification=event_dict.get("classification", self._layer_name),
                    duration=duration_val,
                    metadata=event_dict.get("metadata", {}),
                    id=event_dict.get("id")
                )
                layer.add_event(event_obj)
        
        data_item.event_count = sum(len(l.events) for l in data_item.get_layers())
        self._facade.data_item_repo.update(data_item)

class EditorUpdateLayerCommand(EchoZeroCommand):
    """
    Update layer properties in an Editor block (undoable).
    
    Redo: Updates layer properties in UI state
    Undo: Restores original layer properties
    
    Handles:
    - Updating layer in UI state
    - Triggering UI refresh
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        layer_name: Name of the layer to update
        properties: Dict with properties to update (name, color, height, visible, locked)
    """
    
    COMMAND_TYPE = "editor.update_layer"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        layer_name: str,
        properties: Dict[str, Any]
    ):
        super().__init__(facade, f"Update Layer: {layer_name}")
        
        self._block_id = block_id
        self._layer_name = layer_name
        self._properties = properties
        
        # Store original values for undo
        self._original_properties: Optional[Dict[str, Any]] = None
        self._created_override: bool = False
    
    def redo(self):
        """Update the layer properties."""
        if not self._facade.ui_state_repo:
            self._log_error("UI state repository not available")
            return
        
        # Get current layers
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        
        if not result.success or not result.data:
            self._log_error("Failed to get layers")
            return
        
        layers = result.data.get('layers', [])
        
        # Find the layer
        layer_found = False
        for layer in layers:
            if layer.get('name') == self._layer_name:
                layer_found = True
                # Capture original values (first time only)
                if self._original_properties is None:
                    self._original_properties = layer.copy()
                
                # Update properties
                layer.update(self._properties)
                break
        
        if not layer_found:
            group_id = self._properties.get("group_id")
            group_name = self._properties.get("group_name")
            if group_id and group_name:
                # Build a new override entry for synced layers that exist in data items.
                default_height = self._properties.get('height')
                if default_height is None:
                    try:
                        if hasattr(self._facade, 'preferences_repo') and self._facade.preferences_repo:
                            from ui.qt_gui.widgets.timeline.settings.storage import TimelineSettingsManager
                            timeline_settings = TimelineSettingsManager(preferences_repo=self._facade.preferences_repo)
                            default_height = timeline_settings.default_layer_height
                        else:
                            default_height = 40
                    except Exception:
                        default_height = 40
                new_layer_data = {
                    'name': self._layer_name,
                    'height': default_height,
                    'color': self._properties.get('color'),
                    'visible': self._properties.get('visible', True),
                    'locked': self._properties.get('locked', False),
                    'group_id': group_id,
                    'group_name': group_name,
                    'group_index': self._properties.get('group_index'),
                    'is_synced': self._properties.get('is_synced', False),
                    'show_manager_block_id': self._properties.get('show_manager_block_id'),
                    'ma3_track_coord': self._properties.get('ma3_track_coord'),
                }
                if self._properties.get('derived_from_ma3'):
                    new_layer_data['derived_from_ma3'] = True
                layers.append(new_layer_data)
                self._created_override = True
                layer_found = True
            else:
                self._log_error(f"Layer not found: {self._layer_name}")
                return
        
        # Save to UI state
        result = self._facade.set_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id,
            data={'layers': layers}
        )
        
        if not result.success:
            self._log_error(f"Failed to update layer: {result.message}")
            return
        
        # Trigger UI refresh
        try:
            from src.application.events.event_bus import EventBus
            from src.application.events.block_events import BlockUpdated
            event_bus = EventBus()
            event_bus.publish(BlockUpdated(
                project_id=self._facade.current_project_id,
                data={
                    "id": self._block_id,
                    "layers_updated": True,
                    "layer_name": self._layer_name
                }
            ))
        except ImportError:
            pass
    
    def undo(self):
        """Restore original layer properties."""
        if self._created_override:
            if not self._facade.ui_state_repo:
                return
            result = self._facade.get_ui_state(
                state_type='editor_layers',
                entity_id=self._block_id
            )
            if not result.success or not result.data:
                return
            layers = result.data.get('layers', [])
            layers = [layer for layer in layers if layer.get('name') != self._layer_name]
            self._facade.set_ui_state(
                state_type='editor_layers',
                entity_id=self._block_id,
                data={'layers': layers}
            )
            return
        if not self._original_properties:
            return
        
        if not self._facade.ui_state_repo:
            return
        
        # Get current layers
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        
        if not result.success or not result.data:
            return
        
        layers = result.data.get('layers', [])
        
        # Restore original properties
        for layer in layers:
            if layer.get('name') == self._layer_name:
                layer.clear()
                layer.update(self._original_properties)
                break
        
        # Save to UI state
        self._facade.set_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id,
            data={'layers': layers}
        )
        
        # Trigger UI refresh
        try:
            from src.application.events.event_bus import EventBus
            from src.application.events.block_events import BlockUpdated
            event_bus = EventBus()
            event_bus.publish(BlockUpdated(
                project_id=self._facade.current_project_id,
                data={
                    "id": self._block_id,
                    "layers_updated": True,
                    "layer_name": self._layer_name
                }
            ))
        except ImportError:
            pass

class EditorGetLayersCommand(EchoZeroCommand):
    """
    Get all layers from an Editor block (query command, not undoable).
    
    Returns layers with all properties including group info.
    This is a query command - it doesn't modify state, just retrieves data.
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
    
    Result stored in:
        self.layers: List of layer dicts with all properties
    """
    
    COMMAND_TYPE = "editor.get_layers"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str
    ):
        super().__init__(facade, "Get Editor Layers")
        
        self._block_id = block_id
        self.layers: List[Dict[str, Any]] = []
    
    def redo(self):
        """Query layers derived from EventDataItems, applying UI overrides only."""
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem

        self.layers = []

        if not hasattr(self._facade, "data_item_repo") or not self._facade.data_item_repo:
            message = "Data item repository not available"
            self._log_error(message)
            raise RuntimeError(message)

        items = self._facade.data_item_repo.list_by_block(self._block_id)
        event_items = [item for item in items if isinstance(item, EventDataItem)]

        item_ids = set()
        for item in event_items:
            if not item.id:
                message = f"EventDataItem missing id for block {self._block_id}"
                self._log_error(message)
                raise ValueError(message)
            if item.id in item_ids:
                message = f"Duplicate EventDataItem id detected: {item.id}"
                self._log_error(message)
                raise ValueError(message)
            item_ids.add(item.id)

        overrides_by_key: Dict[tuple, Dict[str, Any]] = {}
        if hasattr(self._facade, "ui_state_repo") and self._facade.ui_state_repo:
            result = self._facade.get_ui_state(
                state_type='editor_layers',
                entity_id=self._block_id
            )
            if result.success and result.data:
                valid_overrides: List[Dict[str, Any]] = []
                needs_override_cleanup = False
                for saved in result.data.get('layers', []):
                    if not isinstance(saved, dict):
                        needs_override_cleanup = True
                        continue
                    name = saved.get('name')
                    group_id = saved.get('group_id')
                    if not name or not group_id:
                        Log.warning(f"EditorGetLayersCommand: Dropping invalid editor_layers override: {saved}")
                        needs_override_cleanup = True
                        continue
                    key = (group_id, name)
                    if key in overrides_by_key:
                        Log.warning(f"EditorGetLayersCommand: Dropping duplicate editor_layers override entry for {key}")
                        needs_override_cleanup = True
                        continue
                    overrides_by_key[key] = saved
                    valid_overrides.append(saved)
                if needs_override_cleanup:
                    try:
                        self._facade.set_ui_state(
                            state_type='editor_layers',
                            entity_id=self._block_id,
                            data={'layers': valid_overrides}
                        )
                    except Exception as exc:
                        Log.warning(f"EditorGetLayersCommand: Failed to clean editor_layers overrides: {exc}")

        allowed_keys = set()
        for item in event_items:
            group_id = item.id
            group_name = item.name
            is_synced = False
            show_manager_block_id = None
            ma3_track_coord = None
            if item.metadata:
                show_manager_block_id = (
                    item.metadata.get("_show_manager_block_id")
                    or item.metadata.get("show_manager_block_id")
                )
                ma3_track_coord = (
                    item.metadata.get("_ma3_track_coord")
                    or item.metadata.get("ma3_track_coord")
                    or item.metadata.get("ma3_coord")
                )
                if (
                    item.metadata.get("_synced_from_ma3") is True
                    or item.metadata.get("_synced_to_ma3") is True
                    or item.metadata.get("_sync_type") in {"showmanager_layer", "ez_layer"}
                    or item.metadata.get("sync_type") in {"showmanager_layer", "ez_layer"}
                    or item.metadata.get("source") in {"ma3", "ma3_sync"}
                ):
                    is_synced = True
                meta_group_id = item.metadata.get("group_id")
                meta_group_name = item.metadata.get("group_name")
                if isinstance(meta_group_id, str) and meta_group_id.startswith("tc_"):
                    group_id = meta_group_id
                    group_name = meta_group_name or group_name
                    is_synced = True

            if not group_id or not group_name:
                message = f"Invalid EventDataItem group identity: {item.id}"
                self._log_error(message)
                raise ValueError(message)

            for event_layer in item.get_layers():
                if not event_layer.name:
                    message = f"EventLayer missing name for item {item.id}"
                    self._log_error(message)
                    raise ValueError(message)
                key = (group_id, event_layer.name)
                if key in allowed_keys:
                    message = f"Duplicate layer key detected: {key}"
                    self._log_error(message)
                    raise ValueError(message)
                allowed_keys.add(key)

                layer_dict = {
                    'name': event_layer.name,
                    'height': 40,
                    'color': None,
                    'visible': True,
                    'locked': False,
                    'group_id': group_id,
                    'group_name': group_name,
                    'group_index': None,
                    'is_synced': is_synced,
                    'show_manager_block_id': show_manager_block_id,
                    'ma3_track_coord': ma3_track_coord,
                    'derived_from_ma3': False,
                    'event_count': len(getattr(event_layer, "events", []) or []),
                }

                override = overrides_by_key.get(key)
                if override:
                    for field in (
                        'height',
                        'color',
                        'visible',
                        'locked',
                        'group_index',
                        'is_synced',
                        'show_manager_block_id',
                        'ma3_track_coord',
                        'derived_from_ma3',
                    ):
                        if field in override:
                            layer_dict[field] = override.get(field)

                self.layers.append(layer_dict)

        stale_override_keys = [key for key in overrides_by_key.keys() if key not in allowed_keys]
        if stale_override_keys:
            message = (
                f"Stale editor_layers overrides for block {self._block_id}: "
                f"{stale_override_keys[:10]}"
            )
            Log.error(message)
            pruned_overrides = [
                override for key, override in overrides_by_key.items()
                if key in allowed_keys
            ]
            try:
                self._facade.set_ui_state(
                    state_type='editor_layers',
                    entity_id=self._block_id,
                    data={'layers': pruned_overrides},
                )
            except Exception as e:
                Log.error(f"Failed to prune stale editor_layers overrides: {e}")
    
    def undo(self):
        """No-op for query commands."""
        pass

class EditorGetEventsCommand(EchoZeroCommand):
    """
    Get events from an Editor block's EventDataItems (query command, not undoable).
    
    Returns events grouped by source/EventDataItem with metadata.
    This is a query command - it doesn't modify state, just retrieves data.
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        layer_name: Optional filter by EventLayer name (optimized - gets events directly from layer)
        source: Optional filter by source (e.g., "ma3", "onset")
    
    Result stored in:
        self.events: List of event dicts
        self.event_data_items: Dict of EventDataItem info by ID
    """
    
    COMMAND_TYPE = "editor.get_events"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        layer_name: Optional[str] = None,
        source: Optional[str] = None
    ):
        super().__init__(facade, "Get Editor Events")
        
        self._block_id = block_id
        self._layer_name = layer_name
        self._source = source
        
        self.events: List[Dict[str, Any]] = []
        self.event_data_items: Dict[str, Dict[str, Any]] = {}
    
    def redo(self):
        """Query events from EventDataItems."""
        self.events = []
        self.event_data_items = {}
        
        if not self._facade.data_item_repo:
            self._log_error("Data item repository not available")
            return
        
        # Get all data items for this block
        from src.shared.domain.entities import EventDataItem
        items = self._facade.data_item_repo.list_by_block(self._block_id)
        
        for item in items:
            if not isinstance(item, EventDataItem):
                continue
            
            # Filter by source if specified
            item_source = item.metadata.get("source") if item.metadata else None
            if self._source and item_source != self._source:
                continue
            
            # Store EventDataItem info
            self.event_data_items[item.id] = {
                "id": item.id,
                "name": item.name,
                "source": item_source,
                "event_count": item.event_count,
                "metadata": item.metadata,
            }
            
            
            # OPTIMIZED: If layer_name specified, get events directly from that layer
            if self._layer_name:
                layer = item.get_layer_by_name(self._layer_name)
                if layer:
                    
                    # Get all events from this specific layer directly (no iteration needed)
                    for event in layer.events:
                        self.events.append({
                            "id": getattr(event, "id", None),
                            "time": event.time,
                            "duration": event.duration,
                            "classification": event.classification,
                            "layer_id": getattr(event, 'layer_id', None),
                            "layer_name": layer.name,  # CRITICAL: Include EventLayer name
                            "metadata": event.metadata,
                            "event_data_item_id": item.id,
                        })
            else:
                # Get events from all layers (backward compatibility)
                for layer in item.get_layers():
                    
                    # Get all events from this layer directly
                    for event in layer.events:
                        self.events.append({
                            "id": getattr(event, "id", None),
                            "time": event.time,
                            "duration": event.duration,
                            "classification": event.classification,
                            "layer_id": getattr(event, 'layer_id', None),
                            "layer_name": layer.name,  # CRITICAL: Include EventLayer name
                            "metadata": event.metadata,
                            "event_data_item_id": item.id,
                        })
    
    def undo(self):
        """No-op for query commands."""
        pass

class EditorDeleteLayerCommand(EchoZeroCommand):
    """
    Delete a layer from an Editor block (undoable).
    
    Completely removes the layer from:
    - TimelineWidget LayerManager (UI)
    - EventDataItem EventLayers (database)
    - All events in that layer (database)
    - Editor block UI state (metadata)
    - ShowManager synced layers (if applicable)
    
    Redo: Deletes layer completely
    Undo: Restores layer with all events
    
    Args:
        facade: ApplicationFacade instance
        block_id: ID of the Editor block
        layer_name: Name of the layer to delete
    """
    
    COMMAND_TYPE = "editor.delete_layer"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        block_id: str,
        layer_name: str
    ):
        super().__init__(facade, f"Delete Layer: {layer_name}")
        
        self._block_id = block_id
        self._layer_name = layer_name
        
        # Store deleted layer data for undo
        self._deleted_layer_data: Optional[Dict[str, Any]] = None
        self._deleted_event_data_items: Dict[str, Dict[str, Any]] = {}
    
    def redo(self):
        """Delete the layer completely."""
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        
        # Step 1: Get layer from UI state (for undo)
        if self._deleted_layer_data is None:
            result = self._facade.get_ui_state(
                state_type='editor_layers',
                entity_id=self._block_id
            )
            
            if result.success and result.data:
                layers = result.data.get('layers', [])
                for layer in layers:
                    if layer.get('name') == self._layer_name:
                        self._deleted_layer_data = layer.copy()
                        break
        
        # Step 2: Check if layer is synced and handle ShowManager deletion
        synced_info = self._check_and_handle_synced_layer_deletion()
        
        # Step 3: Delete from TimelineWidget LayerManager (if panel is open)
        self._delete_from_timeline_widget()
        
        # Step 4: Remove EventLayer from all EventDataItems owned by this block
        if not self._facade.data_item_repo:
            self._log_error("Data item repository not available")
            return
        
        owned_items = self._facade.data_item_repo.list_by_block(self._block_id)
        event_items = [item for item in owned_items if isinstance(item, EventDataItem)]
        
        deleted_event_count = 0
        for item in event_items:
            # Store layer data for undo (first time only)
            if item.id not in self._deleted_event_data_items:
                layer = item.get_layer_by_name(self._layer_name)
                if layer:
                    self._deleted_event_data_items[item.id] = {
                        'layer_data': layer.to_dict(),
                        'event_count': len(layer.events)
                    }
            
            # Remove the layer (this also removes all events in the layer)
            if item.remove_layer(self._layer_name):
                deleted_event_count += len(self._deleted_event_data_items.get(item.id, {}).get('layer_data', {}).get('events', []))
                # Save updated EventDataItem to database
                self._facade.data_item_repo.update(item)
        
        Log.info(f"EditorDeleteLayerCommand: Deleted layer '{self._layer_name}' with {deleted_event_count} events from {len(event_items)} EventDataItem(s)")
        
        # Step 5: Remove from Editor block UI state
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        
        if result.success and result.data:
            layers = result.data.get('layers', [])
            layers = [layer for layer in layers if layer.get('name') != self._layer_name]
            
            self._facade.set_ui_state(
                state_type='editor_layers',
                entity_id=self._block_id,
                data={'layers': layers}
            )

        # Step 6: Remove from layer order repository
        layer_order_service = getattr(self._facade, "layer_order_service", None)
        if layer_order_service and self._deleted_layer_data:
            layer_order_service.remove_layer(
                self._block_id,
                self._deleted_layer_data.get('group_name'),
                self._layer_name
            )
        
        # Step 7: Trigger UI refresh
        try:
            from src.application.events.event_bus import EventBus
            from src.application.events.events import BlockUpdated
            event_bus = EventBus()
            event_bus.publish(BlockUpdated(
                project_id=self._facade.current_project_id,
                data={
                    "id": self._block_id,
                    "layers_updated": True,
                    "layer_deleted": self._layer_name
                }
            ))
        except Exception:
            pass
    
    def _delete_from_timeline_widget(self):
        """Delete layer from TimelineWidget if EditorPanel is open."""
        from src.utils.message import Log
        
        try:
            from PyQt6.QtWidgets import QApplication
            from ui.qt_gui.block_panels.editor_panel import EditorPanel
            
            app = QApplication.instance()
            if not app:
                return
            
            # Find EditorPanel for this block_id
            for widget in app.allWidgets():
                if isinstance(widget, EditorPanel) and widget.block_id == self._block_id:
                    # Panel is open - delete layer from TimelineWidget
                    if hasattr(widget, 'timeline_widget') and hasattr(widget.timeline_widget, '_layer_manager'):
                        layer_manager = widget.timeline_widget._layer_manager
                        # Find layer by name
                        for layer in layer_manager.get_all_layers():
                            if layer.name == self._layer_name:
                                layer_manager.delete_layer(layer.id)
                                Log.debug(f"EditorDeleteLayerCommand: Deleted layer '{self._layer_name}' from TimelineWidget")
                                break
                    return
        except Exception as e:
            Log.debug(f"EditorDeleteLayerCommand: Could not delete from TimelineWidget (non-critical): {e}")
    
    def _check_and_handle_synced_layer_deletion(self) -> Optional[Dict[str, Any]]:
        """Check if layer is synced and mark the sync entity as disconnected.
        
        Instead of removing the sync entity from ShowManager, marks it as
        DISCONNECTED so the user can see it in the ShowManager Layer Sync
        tab and decide whether to reconnect to another Editor layer or
        remove the sync entry entirely.
        
        Returns:
            Dict with synced info (show_manager_block_id, ma3_track_coord) if synced, None otherwise
        """
        from src.utils.message import Log
        
        # Check if layer is synced by looking at UI state
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        
        synced_info = None
        if result.success and result.data:
            layers = result.data.get('layers', [])
            for layer in layers:
                if layer.get('name') == self._layer_name and layer.get('is_synced'):
                    synced_info = {
                        'show_manager_block_id': layer.get('show_manager_block_id'),
                        'ma3_track_coord': layer.get('ma3_track_coord')
                    }
                    break
        
        if not synced_info:
            return None
        
        # Layer is synced - mark the sync entity as DISCONNECTED
        try:
            show_manager_block_id = synced_info['show_manager_block_id']
            
            if show_manager_block_id:
                updated = self._mark_sync_entity_disconnected(show_manager_block_id)
                
                if updated:
                    Log.info(
                        "EditorDeleteLayerCommand: Marked sync entity as DISCONNECTED for "
                        f"'{self._layer_name}' in ShowManager[{show_manager_block_id}]"
                    )
                else:
                    Log.warning(
                        "EditorDeleteLayerCommand: Could not find sync entity to mark as disconnected "
                        f"for '{self._layer_name}' in ShowManager[{show_manager_block_id}]"
                    )
        except Exception as e:
            Log.warning(f"EditorDeleteLayerCommand: Could not handle synced layer deletion: {e}")
        
        return synced_info

    def _mark_sync_entity_disconnected(self, show_manager_block_id: str) -> bool:
        """Mark the sync entity as DISCONNECTED in the live SyncSystemManager or settings.
        
        Tries the live SyncSystemManager first (handles both in-memory and persistence).
        Falls back to direct settings manipulation if the manager is not available.
        
        Returns:
            True if the entity was found and updated.
        """
        from src.utils.message import Log
        
        # Try live SyncSystemManager first (ShowManager panel may be open)
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                from ui.qt_gui.block_panels.show_manager_panel import ShowManagerPanel
                for widget in app.allWidgets():
                    if (isinstance(widget, ShowManagerPanel)
                            and hasattr(widget, 'block_id')
                            and widget.block_id == show_manager_block_id
                            and hasattr(widget, '_sync_system_manager')
                            and widget._sync_system_manager):
                        result = widget._sync_system_manager.notify_editor_layer_deleted(self._layer_name)
                        if result:
                            return True
        except Exception as e:
            Log.debug(f"EditorDeleteLayerCommand: Could not use live SyncSystemManager: {e}")
        
        # Fallback: update persisted settings directly
        try:
            from src.application.settings.show_manager_settings import ShowManagerSettingsManager
            settings_manager = ShowManagerSettingsManager(self._facade, show_manager_block_id)
            synced = settings_manager.synced_layers
            updated = False
            
            for i, entity_dict in enumerate(synced):
                # Match by editor_layer_id (new format) or layer_id (legacy format)
                layer_id = entity_dict.get('editor_layer_id') or entity_dict.get('layer_id')
                if layer_id == self._layer_name:
                    synced[i] = {
                        **entity_dict,
                        'sync_status': 'disconnected',
                        'error_message': (
                            f"Editor layer '{self._layer_name}' was deleted. "
                            "Reconnect to another Editor layer or remove this sync entry."
                        ),
                        'editor_layer_id': None,
                        'editor_block_id': None,
                        'editor_data_item_id': None,
                        # Also clear legacy fields if present
                        'layer_id': None,
                    }
                    updated = True
                    break
            
            if updated:
                settings_manager.synced_layers = synced
                settings_manager.force_save()
                
                # Publish BlockUpdated event to refresh ShowManager UI
                try:
                    from src.application.events.event_bus import EventBus
                    from src.application.events.events import BlockUpdated
                    event_bus = EventBus()
                    block_result = self._facade.describe_block(show_manager_block_id)
                    block_name = block_result.data.name if (block_result.success and block_result.data) else "ShowManager"
                    event_bus.publish(BlockUpdated(
                        project_id=self._facade.current_project_id,
                        data={
                            "id": show_manager_block_id,
                            "name": block_name,
                            "settings_updated": True,
                            "synced_layers_changed": True
                        }
                    ))
                except Exception as e:
                    Log.warning(f"EditorDeleteLayerCommand: Failed to publish BlockUpdated: {e}")
                
                return True
        except Exception as e:
            Log.warning(f"EditorDeleteLayerCommand: Settings fallback failed: {e}")
        
        return False
    
    def undo(self):
        """Restore the deleted layer with all events."""
        from src.utils.message import Log
        from src.shared.domain.entities import EventDataItem
        from src.shared.domain.entities import EventLayer
        
        if not self._deleted_layer_data:
            self._log_warning("No layer data stored, cannot undo")
            return
        
        # Step 1: Restore layer in UI state
        result = self._facade.get_ui_state(
            state_type='editor_layers',
            entity_id=self._block_id
        )
        
        layers = []
        if result.success and result.data:
            layers = result.data.get('layers', [])
        
        # Check if layer already exists
        layer_exists = any(layer.get('name') == self._layer_name for layer in layers)
        if not layer_exists:
            layers.append(self._deleted_layer_data.copy())
            self._facade.set_ui_state(
                state_type='editor_layers',
                entity_id=self._block_id,
                data={'layers': layers}
            )
        
        # Step 2: Restore EventLayers in EventDataItems
        if not self._facade.data_item_repo:
            return
        
        owned_items = self._facade.data_item_repo.list_by_block(self._block_id)
        event_items = [item for item in owned_items if isinstance(item, EventDataItem)]
        
        restored_count = 0
        for item in event_items:
            if item.id in self._deleted_event_data_items:
                layer_data = self._deleted_event_data_items[item.id]['layer_data']
                # Recreate EventLayer from stored data
                events = []
                for event_dict in layer_data.get('events', []):
                    from src.domain.entities.event_data_item import Event
                    event = Event.from_dict(event_dict)
                    events.append(event)
                
                layer = EventLayer(
                    name=layer_data['name'],
                    id=layer_data.get('id'),
                    events=events,
                    metadata=layer_data.get('metadata', {})
                )
                
                item.add_layer(layer)
                self._facade.data_item_repo.update(item)
                restored_count += 1
        
        Log.info(f"EditorDeleteLayerCommand: Restored layer '{self._layer_name}' with events in {restored_count} EventDataItem(s)")
        
        # Step 3: Restore in TimelineWidget (if panel is open)
        try:
            from PyQt6.QtWidgets import QApplication
            from ui.qt_gui.block_panels.editor_panel import EditorPanel
            
            app = QApplication.instance()
            if app:
                for widget in app.allWidgets():
                    if isinstance(widget, EditorPanel) and widget.block_id == self._block_id:
                        if hasattr(widget, '_restore_layer_state'):
                            widget._restore_layer_state()
                        break
        except Exception:
            pass
        
        # Step 4: Trigger UI refresh
        try:
            from src.application.events.event_bus import EventBus
            from src.application.events.block_events import BlockUpdated
            event_bus = EventBus()
            event_bus.publish(BlockUpdated(
                project_id=self._facade.current_project_id,
                data={
                    "id": self._block_id,
                    "layers_updated": True,
                    "layer_restored": self._layer_name
                }
            ))
        except ImportError:
            pass
