"""
Remove Synced Entity Command

Removes an entity (Editor layer or MA3 track) from the synced layers list.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any

from src.application.commands.base_command import EchoZeroCommand
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class RemoveSyncedEntityCommand(EchoZeroCommand):
    """
    Remove an entity from the synced layers list (undoable).
    
    Redo:
        - If removing an MA3 track: Optionally deletes the corresponding Editor layer (if it exists)
        - Removes the MA3 entity from synced_layers
        - Removes the mapped Editor entity from synced_layers if present
    Undo: Adds entity back to synced_layers
        - Note: Editor layer deletion is handled separately by EditorDeleteLayerCommand's undo
    
    Args:
        facade: ApplicationFacade instance
        show_manager_block_id: ShowManager block ID
        entity_type: "editor" or "ma3"
        entity_id: layer_id for editor, coord for ma3
        delete_editor_layer: Whether to delete the mapped Editor layer for MA3 entities
    """
    
    COMMAND_TYPE = "layer_sync.remove_synced_entity"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        show_manager_block_id: str,
        entity_type: str,
        entity_id: str,
        delete_editor_layer: bool = True,
    ):
        entity_name = entity_id
        super().__init__(facade, f"Remove Synced {entity_type.title()} Entity: {entity_name}")
        
        self._show_manager_block_id = show_manager_block_id
        self._entity_type = entity_type
        self._entity_id = entity_id
        self._should_delete_editor_layer = delete_editor_layer
        
        # State for undo (store the entity dict to restore it)
        self._entity_dict: Optional[dict] = None
        self._entity_was_removed: bool = False
        self._deleted_editor_layer: bool = False
        self._deleted_editor_layer_name: Optional[str] = None
        self._deleted_editor_block_id: Optional[str] = None
    
    def redo(self):
        """Remove entity from synced layers."""
        from src.application.settings.show_manager_settings import ShowManagerSettingsManager
        from src.shared.domain.entities import EventDataItem
        
        try:
            settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
            
            # Store entity dict for undo (first time only)
            if self._entity_dict is None:
                self._entity_dict = settings_manager.get_synced_layer(self._entity_type, self._entity_id)
                if not self._entity_dict:
                    self._log_error(f"{self._entity_type.title()} entity '{self._entity_id}' not found in synced layers")
                    return
            
            # If removing an MA3 track, optionally delete the corresponding Editor layer
            if self._entity_type == "ma3":
                mapped_editor_layer_id = self._entity_dict.get('mapped_editor_layer_id')
                if mapped_editor_layer_id:
                    if self._should_delete_editor_layer:
                        self._delete_editor_layer(mapped_editor_layer_id)
                    else:
                        self._clear_editor_layer_sync_metadata(
                            editor_layer_name=mapped_editor_layer_id,
                            ma3_track_coord=self._entity_id,
                            settings_manager=settings_manager,
                            event_data_item_cls=EventDataItem,
                        )
                    # Always remove the mapped Editor entity from synced layers
                    settings_manager.remove_synced_layer("editor", mapped_editor_layer_id)
            
            # Remove from synced layers
            removed = settings_manager.remove_synced_layer(self._entity_type, self._entity_id)
            if removed:
                # Force immediate save (bypass debounce) so panel can reload immediately
                settings_manager.force_save()
                self._entity_was_removed = True
                entity_display = self._entity_dict.get('name') or self._entity_dict.get('display_name') or self._entity_id
                Log.info(f"Removed {self._entity_type.title()} entity '{entity_display}' from synced layers")
                Log.debug(f"ShowManager[{self._show_manager_block_id}]: Removed synced {self._entity_type} entity: {self._entity_id}")
            else:
                self._log_warning(f"{self._entity_type.title()} entity '{self._entity_id}' not found in synced layers")
        except Exception as e:
            self._log_error(f"Failed to remove {self._entity_type.title()} entity '{self._entity_id}' from synced layers: {e}")
            raise
    
    def _find_connected_editor(self) -> Optional[str]:
        """Find connected Editor block."""
        try:
            connections_result = self._facade.list_connections()
            if not connections_result.success or not connections_result.data:
                return None
            
            # Look for connections where ShowManager's manipulator port connects to Editor
            for conn in connections_result.data:
                if conn.source_block_id == self._show_manager_block_id:
                    target_result = self._facade.describe_block(conn.target_block_id)
                    if target_result.success and target_result.data:
                        target_block = target_result.data
                        if target_block.type == "Editor":
                            return target_block.id
        except Exception as e:
            self._log_error(f"Error finding connected Editor: {e}")
        
        return None
    
    def _delete_editor_layer(self, layer_name: str) -> None:
        """
        Delete the Editor layer that was created for this synced MA3 track.
        
        Args:
            layer_name: Name of the Editor layer to delete
        """
        try:
            # Find connected Editor block
            editor_block_id = self._find_connected_editor()
            if not editor_block_id:
                Log.warning(f"Cannot delete Editor layer '{layer_name}': No connected Editor block found")
                return
            
            # Delete the layer using EditorDeleteLayerCommand
            from ..editor_commands import EditorDeleteLayerCommand
            
            delete_cmd = EditorDeleteLayerCommand(
                facade=self._facade,
                block_id=editor_block_id,
                layer_name=layer_name
            )
            
            # Execute the delete command
            try:
                result = self._facade.command_bus.execute(delete_cmd)
            except Exception as exc:
                raise
            if result:
                self._deleted_editor_layer = True
                self._deleted_editor_layer_name = layer_name
                self._deleted_editor_block_id = editor_block_id
                Log.info(f"Deleted Editor layer '{layer_name}' from block {editor_block_id}")
            else:
                Log.warning(f"Failed to delete Editor layer '{layer_name}'")
        except Exception as e:
            Log.warning(f"Error deleting Editor layer '{layer_name}': {e}")
            # Don't raise - continue with removing from synced layers even if layer deletion fails

    def _clear_editor_layer_sync_metadata(
        self,
        editor_layer_name: str,
        ma3_track_coord: str,
        settings_manager,
        event_data_item_cls,
    ) -> None:
        """Clear sync metadata so the Editor layer is no longer treated as synced."""
        editor_block_id = self._find_connected_editor()
        if not editor_block_id:
            Log.warning("RemoveSyncedEntityCommand: Cannot clear sync metadata - no connected Editor block found")
            return

        updated_ui_state = False
        try:
            result = self._facade.get_ui_state(
                state_type="editor_layers",
                entity_id=editor_block_id,
            )
            if result.success and result.data:
                layers = result.data.get("layers", [])
                for layer in layers:
                    if layer.get("name") == editor_layer_name:
                        updated_ui_state = self._clear_sync_keys(layer) or updated_ui_state
                        break
                if updated_ui_state:
                    self._facade.set_ui_state(
                        state_type="editor_layers",
                        entity_id=editor_block_id,
                        data={"layers": layers},
                    )
        except Exception as exc:
            Log.warning(f"RemoveSyncedEntityCommand: Failed clearing editor_layers state: {exc}")


        if not hasattr(self._facade, "data_item_repo") or not self._facade.data_item_repo:
            return

        items = self._facade.data_item_repo.list_by_block(editor_block_id)
        matched_items = 0
        for item in items:
            if not isinstance(item, event_data_item_cls):
                continue

            item_meta = getattr(item, "metadata", {}) or {}
            coord = item_meta.get("_ma3_track_coord") or item_meta.get("ma3_track_coord") or item_meta.get("ma3_coord")
            show_id = item_meta.get("_show_manager_block_id") or item_meta.get("show_manager_block_id")
            if coord and ma3_track_coord and not self._coords_match(coord, ma3_track_coord):
                continue
            if show_id and show_id != self._show_manager_block_id:
                continue

            layer = item.get_layer_by_name(editor_layer_name)
            if not layer:
                continue

            matched_items += 1
            changed = self._clear_sync_keys(item_meta)
            layer_meta = getattr(layer, "metadata", None)
            if isinstance(layer_meta, dict):
                changed = self._clear_sync_keys(layer_meta) or changed
            for evt in getattr(layer, "events", []) or []:
                evt_meta = getattr(evt, "metadata", None)
                if isinstance(evt_meta, dict):
                    changed = self._clear_sync_keys(evt_meta) or changed

            if changed:
                item.metadata = item_meta
                try:
                    self._facade.data_item_repo.update(item)
                except Exception as exc:
                    Log.warning(f"RemoveSyncedEntityCommand: Failed updating EventDataItem {item.id}: {exc}")


        if updated_ui_state:
            try:
                from src.application.events.event_bus import EventBus
                from src.application.events.events import BlockUpdated
                EventBus().publish(BlockUpdated(
                    project_id=self._facade.current_project_id,
                    data={"id": editor_block_id, "layers_updated": True},
                ))
            except Exception as exc:
                Log.warning(f"RemoveSyncedEntityCommand: Failed to publish BlockUpdated: {exc}")

        self._refresh_editor_panel_layer(editor_block_id, editor_layer_name)

    @staticmethod
    def _clear_sync_keys(metadata: Dict[str, Any]) -> bool:
        """Remove sync-related keys from metadata dict."""
        if not isinstance(metadata, dict):
            return False
        keys = {
            "is_synced",
            "_synced_from_ma3",
            "_synced_to_ma3",
            "_show_manager_block_id",
            "show_manager_block_id",
            "_ma3_track_coord",
            "ma3_track_coord",
            "ma3_coord",
            "_sync_type",
            "sync_type",
            "source",
            "group_id",
            "group_name",
        }
        changed = False
        for key in list(metadata.keys()):
            if key in keys:
                metadata.pop(key, None)
                changed = True
        return changed

    def _refresh_editor_panel_layer(self, editor_block_id: str, layer_name: str) -> None:
        """Update live EditorPanel layer flags if panel is open."""
        try:
            from PyQt6.QtWidgets import QApplication
            from ui.qt_gui.block_panels.editor_panel import EditorPanel

            app = QApplication.instance()
            if not app:
                return

            for widget in app.allWidgets():
                if isinstance(widget, EditorPanel) and widget.block_id == editor_block_id:
                    timeline = getattr(widget, "timeline_widget", None)
                    layer_manager = getattr(timeline, "_layer_manager", None) if timeline else None
                    if not layer_manager:
                        return
                    for layer in layer_manager.get_all_layers():
                        if layer.name == layer_name:
                            layer.is_synced = False
                            layer.ma3_track_coord = None
                            layer.show_manager_block_id = None
                            if hasattr(timeline, "_sync_layer_labels_height"):
                                timeline._sync_layer_labels_height()
                            return
        except Exception as exc:
            Log.warning(f"RemoveSyncedEntityCommand: Failed to refresh EditorPanel layer: {exc}")

    @staticmethod
    def _coords_match(coord_a: str, coord_b: str) -> bool:
        """Return True if MA3 coords refer to same track."""
        def _parse(coord: str) -> Optional[tuple[int, int, int]]:
            if not isinstance(coord, str):
                return None
            text = coord.strip()
            if text.startswith("tc"):
                parts = text.split("_")
                if len(parts) != 3:
                    return None
                try:
                    tc = int(parts[0].replace("tc", ""))
                    tg = int(parts[1].replace("tg", ""))
                    track = int(parts[2].replace("tr", ""))
                    return tc, tg, track
                except ValueError:
                    return None
            parts = text.split(".")
            if len(parts) != 3:
                return None
            try:
                return int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                return None

        parsed_a = _parse(coord_a)
        parsed_b = _parse(coord_b)
        if parsed_a and parsed_b:
            return parsed_a == parsed_b
        return coord_a == coord_b
    
    def undo(self):
        """Restore entity to synced layers."""
        if not self._entity_was_removed or not self._entity_dict:
            return
        
        try:
            from src.application.settings.show_manager_settings import ShowManagerSettingsManager
            
            settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
            settings_manager.add_synced_layer(self._entity_dict)
            
            entity_display = self._entity_dict.get('name') or self._entity_dict.get('display_name') or self._entity_id
            Log.info(f"Restored {self._entity_type.title()} entity '{entity_display}' to synced layers")
            Log.debug(f"ShowManager[{self._show_manager_block_id}]: Restored synced {self._entity_type} entity: {self._entity_id}")
            
            # Note: We don't restore the Editor layer on undo because EditorDeleteLayerCommand
            # has its own undo mechanism. The user can undo the layer deletion separately if needed.
        except Exception as e:
            self._log_error(f"Failed to restore {self._entity_type.title()} entity '{self._entity_id}' during undo: {e}")
            raise
