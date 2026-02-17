"""
Layer Sync Commands - Re-export

================================================================================
DEPRECATED - DO NOT ADD FILES HERE
================================================================================

This module ONLY re-exports from src.features.show_manager.application.commands.

CANONICAL LOCATION: src/features/show_manager/application/commands/

Any .py files added to THIS folder will be IGNORED because this __init__.py
imports from the canonical location instead of from local files.

For new commands or modifications, edit files in:
    src/features/show_manager/application/commands/

This redirect exists only for backwards compatibility with old import paths.
================================================================================
"""
import warnings

# Emit deprecation warning when this module is imported
warnings.warn(
    "Importing from src.application.commands.layer_sync is deprecated. "
    "Import from src.features.show_manager.application.commands instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from canonical location
from src.features.show_manager.application.commands import (
    AddSyncedEditorLayerCommand,
    AddSyncedMA3TrackCommand,
    BatchSyncCommand,
    CreateEditorLayerFromMA3Command,
    CreateMA3TrackFromEditorCommand,
    CreateEditorLayerCommand,  # Alias
    CreateMA3TrackCommand,  # Alias
    MapLayersCommand,
    RemoveSyncedEntityCommand,
    SetTargetTimecodeCommand,
    SyncLayerCommand,
    UnmapLayersCommand,
    UpdateEntitySettingsCommand,
    SyncPair,
    PollSyncedMA3TracksCommand,
    RehookSyncedMA3TracksCommand,
    TestSyncedMA3TrackCommand,
    SetApplyUpdatesEnabledCommand,
)

__all__ = [
    'AddSyncedEditorLayerCommand',
    'AddSyncedMA3TrackCommand',
    'BatchSyncCommand',
    'CreateEditorLayerFromMA3Command',
    'CreateMA3TrackFromEditorCommand',
    'CreateEditorLayerCommand',
    'CreateMA3TrackCommand',
    'MapLayersCommand',
    'RemoveSyncedEntityCommand',
    'SetTargetTimecodeCommand',
    'SyncLayerCommand',
    'SyncPair',
    'UnmapLayersCommand',
    'UpdateEntitySettingsCommand',
    'PollSyncedMA3TracksCommand',
    'RehookSyncedMA3TracksCommand',
    'TestSyncedMA3TrackCommand',
    'SetApplyUpdatesEnabledCommand',
]
