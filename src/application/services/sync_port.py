"""
SyncPort

Standardized bidirectional sync entrypoint between Editor and ShowManager.
"""
from typing import Any, Dict, List, Optional

from src.utils.message import Log


class SyncPort:
    """
    Thin port that standardizes Editor <-> ShowManager communication.
    """

    def __init__(self, facade: Any) -> None:
        self._facade = facade
        Log.info("SyncPort: Initialized")

    def apply_from_show_manager(
        self,
        action: str,
        block_id: str,
        **kwargs: Any
    ) -> Any:
        """
        Apply a ShowManager-originated change to the Editor.
        """
        editor_api = self._get_editor_api(block_id)
        if not editor_api:
            Log.warning(f"SyncPort: No EditorAPI for block {block_id}")
            return None

        if action == "sync_layer_replace":
            return editor_api.sync_layer_replace(
                layer_name=kwargs.get("layer_name", ""),
                events=kwargs.get("events", []),
                source=kwargs.get("source", "show_manager")
            )
        if action == "add_events":
            return editor_api.add_events(
                events=kwargs.get("events", []),
                source=kwargs.get("source", "show_manager")
            )
        if action == "add_event":
            return editor_api.add_event(
                time=kwargs.get("time", 0.0),
                classification=kwargs.get("classification", "event"),
                duration=kwargs.get("duration", 0.0),
                metadata=kwargs.get("metadata"),
                source=kwargs.get("source")
            )
        if action == "update_event":
            return editor_api.update_event(
                event_id=kwargs.get("event_id", ""),
                data_item_id=kwargs.get("data_item_id", ""),
                time=kwargs.get("time"),
                duration=kwargs.get("duration"),
                classification=kwargs.get("classification"),
                metadata=kwargs.get("metadata")
            )
        if action == "delete_event":
            return editor_api.delete_event(
                event_id=kwargs.get("event_id", ""),
                data_item_id=kwargs.get("data_item_id", ""),
                layer_name=kwargs.get("layer_name")
            )
        if action == "move_event":
            return editor_api.move_event(
                event_id=kwargs.get("event_id", ""),
                data_item_id=kwargs.get("data_item_id", ""),
                new_time=kwargs.get("new_time", 0.0),
                layer_name=kwargs.get("layer_name")
            )

        Log.warning(f"SyncPort: Unknown action '{action}'")
        return None

    def apply_from_editor(
        self,
        block_id: str,
        layer_name: str,
        change_type: str,
        count: int,
        events: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Apply an Editor-originated change to the ShowManager path.
        """
        editor_api = self._get_editor_api(block_id)
        if not editor_api:
            Log.warning(f"SyncPort: No EditorAPI for block {block_id}")
            return

        if change_type == "added":
            editor_api.events_added.emit(layer_name, count)
        elif change_type == "updated":
            editor_api.events_updated.emit(layer_name, count)
        elif change_type == "deleted":
            editor_api.events_deleted.emit(layer_name, count)

        if events and hasattr(editor_api, "_emit_events_change"):
            event_change_type = {
                "added": "added",
                "updated": "modified",
                "deleted": "deleted"
            }.get(change_type, "modified")
            editor_api._emit_events_change(event_change_type, layer_name, events)


    def _get_editor_api(self, block_id: str) -> Any:
        """Get or create EditorAPI for a block."""
        registry = getattr(self._facade, "editor_api_registry", None)
        if isinstance(registry, dict) and block_id in registry:
            api = registry.get(block_id)
            if api:
                return api

        from src.features.blocks.application.editor_api import create_editor_api
        api = create_editor_api(self._facade, block_id)
        if isinstance(registry, dict) and api:
            registry[block_id] = api
        return api
