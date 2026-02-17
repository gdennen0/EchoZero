"""
Migration Compatibility Layer for Old Project Files

This module patches sys.modules to redirect old import paths to new locations,
allowing old .ez project files (pickled with old paths) to load correctly.

USAGE: Import this module EARLY in main_qt.py before loading any projects:
    import src.migration_compat  # noqa: F401

DELETE THIS FILE once all old projects have been re-saved with new paths.
"""
import sys
from types import ModuleType


def _create_redirect_module(target_module_path: str, class_names: list[str]) -> ModuleType:
    """Create a fake module that redirects attribute access to the real module."""
    fake = ModuleType(target_module_path)
    
    def make_getter(name):
        def getter():
            import importlib
            real_module = importlib.import_module(target_module_path)
            return getattr(real_module, name)
        return getter
    
    # Import the real module and copy attributes
    import importlib
    try:
        real_module = importlib.import_module(target_module_path)
        for name in class_names:
            if hasattr(real_module, name):
                setattr(fake, name, getattr(real_module, name))
    except ImportError:
        pass
    
    return fake


# ============================================================================
# OLD PATH -> NEW PATH MAPPINGS
# ============================================================================
# Format: (old_path, new_path, [class_names])

_REDIRECTS = [
    # Domain entities
    ("src.domain.entities.event_data_item", "src.shared.domain.entities.event_data_item", 
     ["EventDataItem", "Event"]),
    ("src.domain.entities.audio_data_item", "src.shared.domain.entities.audio_data_item",
     ["AudioDataItem"]),
    ("src.domain.entities.data_item", "src.shared.domain.entities.data_item",
     ["DataItem", "EventDataItem"]),
    ("src.domain.entities.event_layer", "src.shared.domain.entities.event_layer",
     ["EventLayer"]),
    ("src.domain.entities.data_item_summary", "src.shared.domain.entities.data_item_summary",
     ["DataItemSummary"]),
    ("src.domain.entities.read_only_data_item", "src.shared.domain.entities.read_only_data_item",
     ["ReadOnlyDataItem"]),
    ("src.domain.entities.data_state_snapshot", "src.shared.domain.entities.data_state_snapshot",
     ["DataStateSnapshot"]),
    ("src.domain.entities.block_summary", "src.shared.domain.entities.block_summary",
     ["BlockSummary"]),
    
    # Block entity
    ("src.domain.entities.block", "src.features.blocks.domain.block",
     ["Block"]),
    ("src.domain.entities.port", "src.features.blocks.domain.port",
     ["Port", "PortDirection"]),
    ("src.domain.entities.port_type", "src.features.blocks.domain.port_type",
     ["PortType", "get_port_type"]),
    ("src.domain.entities.block_status", "src.features.blocks.domain.block_status",
     ["BlockStatus", "BlockStatusLevel"]),
    
    # Connection entity
    ("src.domain.entities.connection", "src.features.connections.domain.connection",
     ["Connection"]),
    ("src.domain.entities.connection_summary", "src.features.connections.domain.connection_summary",
     ["ConnectionSummary"]),
    
    # Project entities
    ("src.domain.entities.project", "src.features.projects.domain.project",
     ["Project"]),
    ("src.domain.entities.action_set", "src.features.projects.domain.action_set",
     ["ActionSet", "ActionItem"]),
    
    # Setlist entities
    ("src.domain.entities.setlist", "src.features.setlists.domain.setlist",
     ["Setlist"]),
    ("src.domain.entities.setlist_song", "src.features.setlists.domain.setlist_song",
     ["SetlistSong"]),
    
    # MA3 entities
    ("src.domain.entities.ma3_event", "src.features.ma3.domain.ma3_event",
     ["MA3Event"]),
    ("src.domain.entities.ma3_sync_state", "src.features.ma3.domain.ma3_sync_state",
     ["MA3SyncState"]),
    
    # Layer sync entities - now unified in SyncLayerEntity
    # Note: Old EditorLayerEntity/MA3TrackEntity are removed. Projects using them need resave.
    ("src.domain.entities.layer_sync", "src.features.show_manager.domain.sync_layer_entity",
     ["SyncLayerEntity", "SyncLayerSettings", "SyncSource", "SyncStatus"]),
    ("src.features.show_manager.domain.sync_layer_entity", "src.features.show_manager.domain.sync_layer_entity",
     ["SyncLayerEntity", "SyncLayerSettings", "SyncSource", "SyncStatus", "ConflictStrategy"]),
    
    # Value objects
    ("src.domain.value_objects.port_type", "src.features.blocks.domain.port_type",
     ["PortType", "get_port_type"]),
    ("src.domain.value_objects.execution_strategy", "src.shared.domain.value_objects.execution_strategy",
     ["ExecutionStrategy"]),
]


def _install_redirects():
    """Install all module redirects into sys.modules."""
    for old_path, new_path, class_names in _REDIRECTS:
        if old_path not in sys.modules:
            fake_module = _create_redirect_module(new_path, class_names)
            fake_module.__name__ = old_path
            fake_module.__file__ = f"<redirect to {new_path}>"
            sys.modules[old_path] = fake_module


# Install redirects on import
_install_redirects()

# Log once per process (avoids spam when module is re-imported, e.g. in child processes)
if not getattr(sys.modules.get(__name__), "_migration_compat_logged", False):
    print("[migration_compat] Old import path redirects installed - old projects should load")
    setattr(sys.modules[__name__], "_migration_compat_logged", True)
