"""
Quick Actions System

Decoupled action definitions that blocks can register for UI display.
Each block type defines its own quick actions, making the system extensible
and keeping action definitions close to block implementations.

Example usage:
    @quick_action("Separator", "Set Model", icon="model", primary=True)
    def set_separator_model(facade, block_id):
        # Action implementation
        pass
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from enum import Enum


class ActionCategory(Enum):
    """Categories for organizing quick actions"""
    EXECUTE = "execute"
    CONFIGURE = "configure"
    FILE = "file"
    EDIT = "edit"
    VIEW = "view"
    EXPORT = "export"


@dataclass
class QuickAction:
    """
    Definition of a quick action for a block type.
    
    Attributes:
        name: Display name for the action
        description: Tooltip/description text
        handler: Callable that executes the action (receives facade, block_id)
        category: Action category for grouping
        icon: Optional icon identifier
        primary: If True, action is highlighted/prominent
        dangerous: If True, action is styled as destructive
        requires_panel: If True, opens the block panel instead of direct execution
        keyboard_shortcut: Optional keyboard shortcut hint
    """
    name: str
    description: str
    handler: Callable[[Any, str], Any]
    category: ActionCategory = ActionCategory.CONFIGURE
    icon: Optional[str] = None
    primary: bool = False
    dangerous: bool = False
    requires_panel: bool = False
    keyboard_shortcut: Optional[str] = None


# Global registry of quick actions per block type
_QUICK_ACTIONS: Dict[str, List[QuickAction]] = {}

# Common actions that apply to all blocks
_COMMON_ACTIONS: List[QuickAction] = []


def register_quick_action(block_type: str, action: QuickAction) -> None:
    """
    Register a quick action for a block type.
    
    Args:
        block_type: The block type ID (e.g., "Separator", "LoadAudio")
        action: The QuickAction to register
    """
    if block_type not in _QUICK_ACTIONS:
        _QUICK_ACTIONS[block_type] = []
    _QUICK_ACTIONS[block_type].append(action)


def register_common_action(action: QuickAction) -> None:
    """
    Register an action that applies to all block types.
    
    Args:
        action: The QuickAction to register
    """
    _COMMON_ACTIONS.append(action)


def get_quick_actions(block_type: str) -> List[QuickAction]:
    """
    Get all quick actions for a block type (including common actions).
    
    Args:
        block_type: The block type ID
        
    Returns:
        List of QuickAction objects
    """
    type_actions = _QUICK_ACTIONS.get(block_type, [])
    if block_type == "Editor":
        actions = [action for action in _COMMON_ACTIONS if action.name != "Execute"]
        return actions + type_actions
    return _COMMON_ACTIONS + type_actions


def get_common_actions() -> List[QuickAction]:
    """Get actions that apply to all blocks."""
    return list(_COMMON_ACTIONS)


def is_execute_block_action(action: QuickAction) -> bool:
    """
    Return True if this action is the standard Execute block action.

    Used by UI to route Execute through the background ExecutionThread
    instead of calling the handler synchronously on the main thread.
    """
    return action.name == "Execute" and action.category == ActionCategory.EXECUTE


def get_all_registered_block_types() -> List[str]:
    """
    Get all block types that have registered quick actions.
    
    Returns:
        List of block type IDs
    """
    return list(_QUICK_ACTIONS.keys())


def quick_action(
    block_type: str,
    name: str,
    description: str = "",
    category: ActionCategory = ActionCategory.CONFIGURE,
    icon: Optional[str] = None,
    primary: bool = False,
    dangerous: bool = False,
    requires_panel: bool = False,
    keyboard_shortcut: Optional[str] = None
):
    """
    Decorator to register a function as a quick action.
    
    Usage:
        @quick_action("Separator", "Set Model", primary=True)
        def set_separator_model(facade, block_id):
            # Implementation
            pass
    """
    def decorator(func: Callable):
        action = QuickAction(
            name=name,
            description=description or f"{name} for this block",
            handler=func,
            category=category,
            icon=icon,
            primary=primary,
            dangerous=dangerous,
            requires_panel=requires_panel,
            keyboard_shortcut=keyboard_shortcut
        )
        register_quick_action(block_type, action)
        return func
    return decorator


def common_action(
    name: str,
    description: str = "",
    category: ActionCategory = ActionCategory.EDIT,
    icon: Optional[str] = None,
    primary: bool = False,
    dangerous: bool = False,
    keyboard_shortcut: Optional[str] = None
):
    """
    Decorator to register a function as a common action (applies to all blocks).
    
    Usage:
        @common_action("Execute", primary=True)
        def execute_block(facade, block_id):
            # Implementation
            pass
    """
    def decorator(func: Callable):
        action = QuickAction(
            name=name,
            description=description or f"{name}",
            handler=func,
            category=category,
            icon=icon,
            primary=primary,
            dangerous=dangerous,
            keyboard_shortcut=keyboard_shortcut
        )
        register_common_action(action)
        return func
    return decorator


# ============================================================================
# Common Actions (apply to all blocks)
# ============================================================================

@common_action(
    "Execute",
    description="Run this block and its dependencies",
    category=ActionCategory.EXECUTE,
    icon="play",
    primary=True,
    keyboard_shortcut="Ctrl+Enter"
)
def action_execute_block(facade, block_id: str, **kwargs):
    """Execute a single block"""
    return facade.execute_block(block_id)


@common_action(
    "Pull Data (Overwrite)",
    description="Wipe this block's local inputs and re-pull from incoming connections",
    category=ActionCategory.CONFIGURE,
    icon="refresh",
    primary=False
)
def action_pull_data_overwrite(facade, block_id: str, confirmed: bool = False, **kwargs):
    """
    Pull inputs from connections into persisted local state (overwrite).

    MVP behavior:
    - Requires confirmation before wipe.
    - If upstream has no persisted outputs, returns an error describing missing sources.
    """
    if not confirmed:
        return {
            "needs_confirmation": True,
            "confirm_arg": "confirmed",
            "message": "Overwrite this block's local inputs and re-pull from its connections?"
        }

    result = facade.pull_block_inputs_overwrite(block_id)
    if hasattr(result, "success") and not result.success:
        # Surface missing upstream detail if present
        missing = None
        try:
            if isinstance(result.data, dict):
                missing = result.data.get("missing")
        except Exception:
            missing = None
        if missing:
            return {
                "needs_confirmation": False,
                "success": False,
                "error": "Upstream has no data for one or more connections. Execute upstream blocks first.",
                "missing": missing,
            }
    return result

@common_action(
    "Rename",
    description="Rename this block",
    category=ActionCategory.EDIT,
    icon="edit",
    keyboard_shortcut="F2"
)
def action_rename_block(facade, block_id: str, new_name: str = None, **kwargs):
    """Rename a block - if new_name not provided, UI should prompt"""
    if new_name:
        return facade.rename_block(block_id, new_name)
    return {"needs_input": True, "input_type": "text", "prompt": "Enter new name:"}


@common_action(
    "Delete",
    description="Remove this block from the project",
    category=ActionCategory.EDIT,
    icon="trash",
    dangerous=True,
    keyboard_shortcut="Delete"
)
def action_delete_block(facade, block_id: str, confirmed: bool = False, **kwargs):
    """Delete a block - requires confirmation"""
    if confirmed:
        return facade.delete_block(block_id)
    return {"needs_confirmation": True, "message": "Delete this block?"}


@common_action(
    "Open Panel",
    description="Open the configuration panel for this block",
    category=ActionCategory.VIEW,
    icon="settings",
    keyboard_shortcut="Enter"
)
def action_open_panel(facade, block_id: str, **kwargs):
    """Signal to open the block's configuration panel"""
    return {"open_panel": True, "block_id": block_id}


@common_action(
    "Filter Data",
    description="Preview and filter data items for input/output ports",
    category=ActionCategory.VIEW,
    icon="filter"
)
def action_filter_data(facade, block_id: str, **kwargs):
    """Open the data filter dialog for this block"""
    return {"open_filter_dialog": True, "block_id": block_id}


@common_action(
    "Reset State",
    description="Reset this block's configuration to initial state",
    category=ActionCategory.EDIT,
    icon="refresh",
    dangerous=True
)
def action_reset_block_state(facade, block_id: str, confirmed: bool = False, **kwargs):
    """
    Reset block state to initial state.
    
    This clears:
    - All configuration and settings (metadata)
    - All owned data items (for Editor blocks: events, layers, audio)
    - All local state (input/output references)
    
    Returns the block to the state it had when first created.
    """
    if not confirmed:
        # Get block type for more specific warning
        block_result = facade.describe_block(block_id)
        block_type = ""
        if block_result.success and block_result.data:
            block_type = block_result.data.type
        
        message = "Reset this block's state to initial state?"
        if block_type == "Editor":
            message += "\n\nThis will clear:\n- All events and layers\n- All audio references\n- All configuration settings"
        else:
            message += "\n\nThis will clear all configuration settings and owned data."
        
        message += "\n\nThis action cannot be undone for data items."
        
        return {
            "needs_confirmation": True,
            "message": message
        }
    
    result = facade.reset_block_state(block_id)
    if result.success:
        return {
            "success": True,
            "message": f"Block state reset to initial state (cleared metadata, data items, and local state)"
        }
    else:
        return {
            "success": False,
            "error": result.message if hasattr(result, 'message') else "Failed to reset block state"
        }


# ============================================================================
# LoadAudio Actions
# ============================================================================

@quick_action(
    "LoadAudio",
    "Set Audio File",
    description="Choose an audio file to load",
    category=ActionCategory.FILE,
    icon="file-audio",
    primary=True
)
def action_set_audio_file(facade, block_id: str, file_path: str = None, **kwargs):
    """
    Set audio file path via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.load_audio_settings import LoadAudioSettingsManager
    
    if file_path:
        # Write path: set the audio file path
        try:
            settings_manager = LoadAudioSettingsManager(facade, block_id)
            settings_manager.audio_path = file_path
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            from pathlib import Path
            return {"success": True, "message": f"Audio file set: {Path(file_path).name}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    return {
        "needs_input": True, 
        "input_type": "file",
        "file_filter": "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)",
        "title": "Select Audio File"
    }


# ============================================================================
# SetlistAudioInput Actions
# ============================================================================

@quick_action(
    "SetlistAudioInput",
    "Set Audio File",
    description="Set the audio file for this setlist song",
    category=ActionCategory.FILE,
    icon="file-audio",
    primary=True
)
def action_set_setlist_audio_file(facade, block_id: str, file_path: str = None, **kwargs):
    """
    Set audio file path for SetlistAudioInput block.
    
    This is the required action for setlist processing.
    Sets audio_path in block.metadata directly (no settings manager needed).
    """
    if file_path:
        # Write path: set the audio file path directly in metadata
        try:
            from pathlib import Path
            path_obj = Path(file_path).expanduser()
            if not path_obj.is_file():
                return {"success": False, "error": f"Audio file not found: {file_path}"}
            
            # Get block and update metadata
            result = facade.get_block_metadata(block_id)
            if not result.success:
                return {"success": False, "error": f"Failed to get block: {result.error}"}
            
            block_metadata = result.data
            block_metadata["audio_path"] = str(path_obj)
            
            # Update block metadata
            update_result = facade.update_block_metadata(block_id, block_metadata)
            if not update_result.success:
                return {"success": False, "error": f"Failed to update block: {update_result.error}"}
            
            return {"success": True, "message": f"Audio file set: {path_obj.name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    return {
        "needs_input": True,
        "input_type": "file",
        "file_filter": "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)",
        "title": "Select Audio File for Setlist Song"
    }


# ============================================================================
# Separator Actions
# ============================================================================

@quick_action(
    "Separator",
    "Set Model",
    description="Choose the Demucs model for separation",
    category=ActionCategory.CONFIGURE,
    icon="model",
    primary=True
)
def action_set_separator_model(facade, block_id: str, model: str = None, **kwargs):
    """
    Set the Demucs model via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.separator_settings import SeparatorSettingsManager
    
    if model:
        # Write path: set the model
        try:
            settings_manager = SeparatorSettingsManager(facade, block_id)
            settings_manager.model = model
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Model set to {model}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = SeparatorSettingsManager(facade, block_id)
        current_model = settings_manager.model  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_model = "htdemucs"
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"],
        "default": current_model,  # Current value from database (single source of truth)
        "title": "Select Demucs Model"
    }


@quick_action(
    "Separator",
    "Set Device",
    description="Choose processing device (CPU/GPU)",
    category=ActionCategory.CONFIGURE,
    icon="cpu"
)
def action_set_separator_device(facade, block_id: str, device: str = None, **kwargs):
    """
    Set processing device via settings manager.
    
    Single source of truth: block.metadata in database.
    """
    from src.application.settings.separator_settings import SeparatorSettingsManager
    
    if device:
        # Write path: set the device
        try:
            settings_manager = SeparatorSettingsManager(facade, block_id)
            settings_manager.device = device
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Device set to {device}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = SeparatorSettingsManager(facade, block_id)
        current_device = settings_manager.device  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_device = "auto"
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["auto", "cpu", "cuda", "mps"],
        "default": current_device,  # Current value from database (single source of truth)
        "title": "Select Processing Device"
    }


@quick_action(
    "Separator",
    "Two-Stem Mode",
    description="Output only 2 stems (selected + rest combined)",
    category=ActionCategory.CONFIGURE,
    icon="split"
)
def action_set_two_stems(facade, block_id: str, stem: str = None, **kwargs):
    """
    Set two-stem mode via settings manager.
    
    Single source of truth: block.metadata in database.
    """
    from src.application.settings.separator_settings import SeparatorSettingsManager
    
    if stem is not None:  # Allow empty string to clear
        # Write path: set the two-stems mode
        try:
            settings_manager = SeparatorSettingsManager(facade, block_id)
            # Convert empty string to None
            two_stems_value = None if stem == "" else stem
            settings_manager.two_stems = two_stems_value
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            mode_text = two_stems_value if two_stems_value else "All stems"
            return {"success": True, "message": f"Separation mode set to {mode_text}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = SeparatorSettingsManager(facade, block_id)
        current_two_stems = settings_manager.two_stems  # Read from database
        # Convert None to empty string for dialog (None = "Full 4-stem")
        current_default = "" if current_two_stems is None else current_two_stems
    except Exception:
        # Fallback if settings manager fails to load
        current_default = ""
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["", "vocals", "drums", "bass", "other"],
        "labels": ["Full 4-stem", "Vocals only", "Drums only", "Bass only", "Other only"],
        "default": current_default,  # Current value from database (single source of truth)
        "title": "Two-Stem Mode"
    }


# ============================================================================
# Export Audio Actions
# ============================================================================

@quick_action(
    "ExportAudio",
    "Set Output Directory",
    description="Choose where to save exported files",
    category=ActionCategory.FILE,
    icon="folder",
    primary=True
)
def action_set_export_dir(facade, block_id: str, directory: str = None, **kwargs):
    """
    Set export output directory via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.export_audio_settings import ExportAudioSettingsManager
    
    if directory:
        # Write path: set the output directory
        try:
            settings_manager = ExportAudioSettingsManager(facade, block_id)
            settings_manager.output_dir = directory
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Output directory set to {directory}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: directory selection uses file dialog, no default needed
    return {
        "needs_input": True,
        "input_type": "directory",
        "title": "Select Output Directory"
    }


@quick_action(
    "ExportAudio",
    "Set Format",
    description="Choose audio export format",
    category=ActionCategory.CONFIGURE,
    icon="file"
)
def action_set_export_format(facade, block_id: str, fmt: str = None, **kwargs):
    """
    Set audio export format via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.export_audio_settings import ExportAudioSettingsManager
    
    if fmt:
        # Write path: set the format
        try:
            settings_manager = ExportAudioSettingsManager(facade, block_id)
            settings_manager.audio_format = fmt
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Format set to {fmt.upper()}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = ExportAudioSettingsManager(facade, block_id)
        current_format = settings_manager.audio_format  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_format = "wav"
    
    # Filter choices to only valid formats (settings manager validates)
    valid_formats = ["wav", "mp3", "flac", "ogg"]
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": valid_formats,
        "default": current_format,  # Current value from database (single source of truth)
        "title": "Select Audio Format"
    }


# ============================================================================
# Export Clips By Class Actions
# ============================================================================

@quick_action(
    "ExportClipsByClass",
    "Set Output Directory",
    description="Choose base folder for exported clips (subfolders created per class)",
    category=ActionCategory.FILE,
    icon="folder",
    primary=True
)
def action_set_clips_export_dir(facade, block_id: str, directory: str = None, **kwargs):
    """
    Set export output directory via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.export_clips_by_class_settings import ExportClipsByClassSettingsManager
    
    if directory:
        # Write path: set the output directory
        try:
            settings_manager = ExportClipsByClassSettingsManager(facade, block_id)
            settings_manager.output_dir = directory
            settings_manager.force_save()
            return {"success": True, "message": f"Output directory set to {directory}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: directory selection uses file dialog
    return {
        "needs_input": True,
        "input_type": "directory",
        "title": "Select Output Directory"
    }


@quick_action(
    "ExportClipsByClass",
    "Set Format",
    description="Choose audio export format for clips",
    category=ActionCategory.CONFIGURE,
    icon="file"
)
def action_set_clips_format(facade, block_id: str, fmt: str = None, **kwargs):
    """
    Set audio export format via settings manager.
    """
    from src.application.settings.export_clips_by_class_settings import ExportClipsByClassSettingsManager
    
    if fmt:
        try:
            settings_manager = ExportClipsByClassSettingsManager(facade, block_id)
            settings_manager.audio_format = fmt
            settings_manager.force_save()
            return {"success": True, "message": f"Format set to {fmt.upper()}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    try:
        settings_manager = ExportClipsByClassSettingsManager(facade, block_id)
        current_format = settings_manager.audio_format
    except Exception:
        current_format = "wav"
    
    valid_formats = ["wav", "mp3", "flac", "ogg"]
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": valid_formats,
        "default": current_format,
        "title": "Select Audio Format"
    }


@quick_action(
    "ExportClipsByClass",
    "Include Unclassified",
    description="Toggle whether to export events without classification",
    category=ActionCategory.CONFIGURE,
    icon="filter"
)
def action_toggle_include_unclassified(facade, block_id: str, enabled: bool = None, **kwargs):
    """
    Toggle including unclassified events.
    """
    from src.application.settings.export_clips_by_class_settings import ExportClipsByClassSettingsManager
    
    if enabled is not None:
        try:
            settings_manager = ExportClipsByClassSettingsManager(facade, block_id)
            settings_manager.include_unclassified = enabled
            settings_manager.force_save()
            status = "enabled" if enabled else "disabled"
            return {"success": True, "message": f"Include unclassified: {status}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    try:
        settings_manager = ExportClipsByClassSettingsManager(facade, block_id)
        current = settings_manager.include_unclassified
    except Exception:
        current = True
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["true", "false"],
        "default": "true" if current else "false",
        "title": "Include Unclassified Events?"
    }


# ============================================================================
# TranscribeNote Actions
# ============================================================================

@quick_action(
    "TranscribeNote",
    "Set Onset Threshold",
    description="Adjust note onset detection sensitivity",
    category=ActionCategory.CONFIGURE,
    icon="slider"
)
def action_set_onset_threshold(facade, block_id: str, value: float = None, **kwargs):
    """
    Set onset threshold via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.transcribe_note_settings import TranscribeNoteSettingsManager
    
    if value is not None:
        # Write path: set the threshold
        try:
            settings_manager = TranscribeNoteSettingsManager(facade, block_id)
            settings_manager.onset_threshold = float(value)
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Onset threshold set to {value:.2f}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = TranscribeNoteSettingsManager(facade, block_id)
        current_threshold = settings_manager.onset_threshold  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_threshold = 0.5
    
    return {
        "needs_input": True,
        "input_type": "number",
        "min": 0.0,
        "max": 1.0,
        "default": current_threshold,  # Current value from database (single source of truth)
        "step": 0.05,
        "increment_jump": 0.05,  # Step size for increment/decrement arrows
        "decimals": 2,
        "title": "Onset Threshold (0.0-1.0)"
    }


@quick_action(
    "TranscribeNote",
    "Set Min Note Length",
    description="Filter out very short notes",
    category=ActionCategory.CONFIGURE,
    icon="timer"
)
def action_set_min_note_length(facade, block_id: str, value: float = None, **kwargs):
    """
    Set minimum note length (duration) via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.transcribe_note_settings import TranscribeNoteSettingsManager
    
    if value is not None:
        # Write path: set the min duration
        try:
            settings_manager = TranscribeNoteSettingsManager(facade, block_id)
            settings_manager.min_duration = float(value)
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Min note length set to {value:.2f}s"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = TranscribeNoteSettingsManager(facade, block_id)
        current_duration = settings_manager.min_duration  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_duration = 0.05
    
    return {
        "needs_input": True,
        "input_type": "number",
        "min": 0.0,
        "max": 5.0,
        "default": current_duration,  # Current value from database (single source of truth)
        "step": 0.01,
        "increment_jump": 0.01,  # Step size for increment/decrement arrows
        "decimals": 2,
        "title": "Min Note Length (seconds)"
    }


# ============================================================================
# PlotEvents Actions
# ============================================================================

@quick_action(
    "PlotEvents",
    "Set Output Directory",
    description="Choose where to save visualizations",
    category=ActionCategory.FILE,
    icon="folder",
    primary=True
)
def action_set_plot_output_dir(facade, block_id: str, directory: str = None, **kwargs):
    """Set plot output directory"""
    if directory:
        return facade.execute_block_command(block_id, "set_output_dir", [directory], {})
    return {
        "needs_input": True,
        "input_type": "directory",
        "title": "Select Output Directory"
    }


@quick_action(
    "PlotEvents",
    "Set Plot Style",
    description="Choose visualization style",
    category=ActionCategory.CONFIGURE,
    icon="chart"
)
def action_set_plot_style(facade, block_id: str, style: str = None, **kwargs):
    """
    Set plot style via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.plot_events_settings import PlotEventsSettingsManager
    
    if style:
        # Write path: set the plot style
        try:
            settings_manager = PlotEventsSettingsManager(facade, block_id)
            settings_manager.plot_style = style
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Plot style set to {style}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = PlotEventsSettingsManager(facade, block_id)
        current_style = settings_manager.plot_style  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_style = "bars"
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["bars", "markers", "piano_roll"],
        "default": current_style,  # Current value from database (single source of truth)
        "title": "Select Plot Style"
    }


# ============================================================================
# Editor Actions
# ============================================================================

@quick_action(
    "Editor",
    "Validate Data",
    description="Check for missing audio files and stale events",
    category=ActionCategory.VIEW,
    icon="check-circle"
)
def action_validate_editor_data(facade, block_id: str, **kwargs):
    """
    Validate Editor data for stale audio files.
    
    Returns information about validation status that the UI can display.
    """
    from src.utils.tools import validate_audio_items
    from src.shared.domain.entities import AudioDataItem
    
    try:
        # Get audio data items for this editor
        audio_items = []
        
        # Check local state for audio references
        if hasattr(facade, 'block_local_state_repo'):
            local_inputs = facade.block_local_state_repo.get_inputs(block_id) or {}
            audio_ref = local_inputs.get("audio")
            
            if audio_ref:
                audio_ids = audio_ref if isinstance(audio_ref, list) else [audio_ref]
                for aid in audio_ids:
                    item = facade.data_item_repo.get(aid)
                    if isinstance(item, AudioDataItem):
                        audio_items.append(item)
        
        if not audio_items:
            return {
                "success": True,
                "message": "No audio data to validate",
                "validation_status": "no_data"
            }
        
        # Validate audio items
        validation_result = validate_audio_items(audio_items)
        
        if validation_result['all_valid']:
            return {
                "success": True,
                "message": f"✓ All {len(audio_items)} audio file(s) are accessible",
                "validation_status": "valid"
            }
        else:
            invalid_count = len(validation_result['invalid'])
            errors = [f"• {item.name}: {error}" for item, error in validation_result['invalid']]
            error_msg = "\n".join(errors[:5])  # Show first 5
            if invalid_count > 5:
                error_msg += f"\n... and {invalid_count - 5} more"
            
            return {
                "success": False,
                "message": f"⚠ {invalid_count} audio file(s) not found:\n{error_msg}",
                "validation_status": "invalid",
                "invalid_items": validation_result['invalid']
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Validation failed: {str(e)}",
            "validation_status": "error"
        }


@quick_action(
    "Editor",
    "Clear Stale Events",
    description="Remove events with missing audio files",
    category=ActionCategory.EDIT,
    icon="trash",
    dangerous=True
)
def action_clear_stale_events(facade, block_id: str, **kwargs):
    """
    Clear events that reference missing audio files.
    
    This is a dangerous operation that requires confirmation.
    """
    from src.utils.tools import validate_audio_items, validate_audio_data_item
    from src.shared.domain.entities import AudioDataItem
    from src.shared.domain.entities import EventDataItem
    
    try:
        # Get audio items
        audio_items = {}
        if hasattr(facade, 'block_local_state_repo'):
            local_inputs = facade.block_local_state_repo.get_inputs(block_id) or {}
            audio_ref = local_inputs.get("audio")
            
            if audio_ref:
                audio_ids = audio_ref if isinstance(audio_ref, list) else [audio_ref]
                for aid in audio_ids:
                    item = facade.data_item_repo.get(aid)
                    if isinstance(item, AudioDataItem):
                        audio_items[aid] = item
        
        # Get event items owned by this editor
        event_items = facade.data_item_repo.list_by_block(block_id)
        event_items = [item for item in event_items if isinstance(item, EventDataItem)]
        
        if not event_items:
            return {
                "success": True,
                "message": "No events to clear",
                "items_removed": 0
            }
        
        # Find stale events (events whose audio is missing)
        stale_event_items = []
        for event_item in event_items:
            source_audio_id = event_item.metadata.get('_source_audio_id')
            if source_audio_id and source_audio_id in audio_items:
                audio_item = audio_items[source_audio_id]
                is_valid, _ = validate_audio_data_item(audio_item)
                if not is_valid:
                    stale_event_items.append(event_item)
        
        if not stale_event_items:
            return {
                "success": True,
                "message": "No stale events found",
                "items_removed": 0
            }
        
        # This action requires confirmation - return info for UI to show dialog
        return {
            "needs_confirmation": True,
            "message": f"Remove {len(stale_event_items)} event item(s) with missing audio?",
            "items_to_remove": [item.id for item in stale_event_items],
            "confirmation_action": "delete_stale_events"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to identify stale events: {str(e)}"
        }


# ============================================================================
# DetectOnsets Actions
# ============================================================================

# ============================================================================
# ShowManager Actions
# ============================================================================

@quick_action(
    "ShowManager",
    "Set Target Timecode",
    description="Change the target timecode pool index for MA3 sync",
    category=ActionCategory.CONFIGURE,
    icon="timer",
    primary=True
)
def action_set_target_timecode(facade, block_id: str, timecode_number: int = None, **kwargs):
    """
    Set the target timecode number via ShowManager settings.
    
    Controls which timecode pool in MA3 the ShowManager fetches from.
    Supports setlist placeholders like {song_index_1} for auto-incrementing.
    """
    from src.application.settings.show_manager_settings import ShowManagerSettingsManager
    
    if timecode_number is not None:
        try:
            timecode_number = int(timecode_number)
            if timecode_number < 1:
                return {"success": False, "error": "Timecode number must be >= 1"}
            settings_manager = ShowManagerSettingsManager(facade, block_id)
            settings_manager.target_timecode = timecode_number
            settings_manager.force_save()
            return {"success": True, "message": f"Target timecode set to {timecode_number}"}
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"Invalid timecode number: {e}"}
    
    # Read current value for dialog default
    try:
        settings_manager = ShowManagerSettingsManager(facade, block_id)
        current = settings_manager.target_timecode
    except Exception:
        current = 1
    
    return {
        "needs_input": True,
        "input_type": "number",
        "min": 1,
        "max": 99999,
        "default": current,
        "step": 1,
        "increment_jump": 1,
        "decimals": 0,
        "title": "Target Timecode Number"
    }


@quick_action(
    "ShowManager",
    "Rehook Synced Tracks",
    description="Re-establish MA3 hook subscriptions for all synced tracks",
    category=ActionCategory.EXECUTE,
    icon="refresh"
)
def action_rehook_synced_tracks(facade, block_id: str, **kwargs):
    """
    Rehook all synced MA3 tracks for real-time change notifications.
    
    Sends GetTracks and HookCmdSubTrack commands to MA3 for each
    synced track. Useful before sync operations to ensure connections are live.
    """
    from src.features.show_manager.application.commands import RehookSyncedMA3TracksCommand
    
    try:
        cmd = RehookSyncedMA3TracksCommand(facade, block_id)
        if facade.command_bus:
            facade.command_bus.execute(cmd)
        else:
            cmd.redo()
        return {
            "success": True,
            "message": f"Rehook complete: {cmd.hooked_count}/{cmd.requested_count} tracks hooked"
        }
    except Exception as e:
        return {"success": False, "error": f"Rehook failed: {e}"}


@quick_action(
    "ShowManager",
    "Poll Synced Tracks",
    description="Request fresh event lists from MA3 for all synced tracks",
    category=ActionCategory.EXECUTE,
    icon="download"
)
def action_poll_synced_tracks(facade, block_id: str, **kwargs):
    """
    Poll all synced MA3 tracks by requesting their event lists.
    
    Sends GetEvents commands to MA3 for each synced track.
    Useful to refresh state before or after batch operations.
    """
    from src.features.show_manager.application.commands import PollSyncedMA3TracksCommand
    
    try:
        cmd = PollSyncedMA3TracksCommand(facade, block_id)
        if facade.command_bus:
            facade.command_bus.execute(cmd)
        else:
            cmd.redo()
        return {
            "success": True,
            "message": f"Poll complete: {cmd.polled_count}/{cmd.requested_count} tracks polled"
        }
    except Exception as e:
        return {"success": False, "error": f"Poll failed: {e}"}


@quick_action(
    "ShowManager",
    "Sync All to MA3",
    description="Push all synced Editor layers to MA3",
    category=ActionCategory.EXECUTE,
    icon="upload",
    primary=True
)
def action_sync_all_to_ma3(facade, block_id: str, **kwargs):
    """
    Push all synced Editor layers to MA3.
    
    Iterates over all synced layer entities and calls apply_to_ma3
    on each one via the SyncSystemManager.
    """
    try:
        ssm = facade.sync_system_manager(block_id)
        if not ssm:
            return {"success": False, "error": "SyncSystemManager not available"}
        
        entities = ssm.get_synced_layers()
        if not entities:
            return {"success": True, "message": "No synced layers to push"}
        
        pushed = 0
        errors = []
        for entity in entities:
            try:
                ssm.apply_to_ma3(entity.id)
                pushed += 1
            except Exception as e:
                errors.append(f"{entity.name}: {e}")
        
        if errors:
            return {
                "success": False,
                "error": f"Pushed {pushed}/{len(entities)} layers. Errors: {'; '.join(errors[:3])}"
            }
        return {"success": True, "message": f"Pushed {pushed} layer(s) to MA3"}
    except Exception as e:
        return {"success": False, "error": f"Sync to MA3 failed: {e}"}


@quick_action(
    "ShowManager",
    "Sync All from MA3",
    description="Pull all MA3 tracks to Editor",
    category=ActionCategory.EXECUTE,
    icon="download"
)
def action_sync_all_from_ma3(facade, block_id: str, **kwargs):
    """
    Pull all synced MA3 tracks into Editor layers.
    
    Iterates over all synced layer entities and calls apply_to_ez
    on each one via the SyncSystemManager.
    """
    try:
        ssm = facade.sync_system_manager(block_id)
        if not ssm:
            return {"success": False, "error": "SyncSystemManager not available"}
        
        entities = ssm.get_synced_layers()
        if not entities:
            return {"success": True, "message": "No synced layers to pull"}
        
        pulled = 0
        errors = []
        for entity in entities:
            try:
                ssm.apply_to_ez(entity.id)
                pulled += 1
            except Exception as e:
                errors.append(f"{entity.name}: {e}")
        
        if errors:
            return {
                "success": False,
                "error": f"Pulled {pulled}/{len(entities)} layers. Errors: {'; '.join(errors[:3])}"
            }
        return {"success": True, "message": f"Pulled {pulled} layer(s) from MA3"}
    except Exception as e:
        return {"success": False, "error": f"Sync from MA3 failed: {e}"}


@quick_action(
    "ShowManager",
    "Set Apply Updates",
    description="Enable or disable MA3-to-Editor live updates",
    category=ActionCategory.CONFIGURE,
    icon="toggle"
)
def action_set_apply_updates(facade, block_id: str, enabled: bool = None, **kwargs):
    """
    Enable or disable applying MA3 updates to Editor when hooked.
    
    When enabled, changes in MA3 are automatically applied to Editor layers.
    When disabled, MA3 changes are received but not applied.
    """
    from src.application.settings.show_manager_settings import ShowManagerSettingsManager
    
    if enabled is not None:
        try:
            # Handle string "true"/"false" from action args
            if isinstance(enabled, str):
                enabled = enabled.lower() in ("true", "1", "yes")
            settings_manager = ShowManagerSettingsManager(facade, block_id)
            settings_manager.apply_updates_enabled = bool(enabled)
            settings_manager.force_save()
            status = "enabled" if enabled else "disabled"
            return {"success": True, "message": f"Apply updates: {status}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to set apply updates: {e}"}
    
    try:
        settings_manager = ShowManagerSettingsManager(facade, block_id)
        current = settings_manager.apply_updates_enabled
    except Exception:
        current = True
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["true", "false"],
        "labels": ["Enabled", "Disabled"],
        "default": "true" if current else "false",
        "title": "Apply MA3 Updates to Editor?"
    }


@quick_action(
    "ShowManager",
    "Set Sync on Change",
    description="Enable or disable real-time sync on changes",
    category=ActionCategory.CONFIGURE,
    icon="toggle"
)
def action_set_sync_on_change(facade, block_id: str, enabled: bool = None, **kwargs):
    """
    Enable or disable real-time sync when Editor changes occur.
    
    When enabled, Editor changes are immediately synced to MA3.
    When disabled, changes accumulate until manual sync.
    """
    from src.application.settings.show_manager_settings import ShowManagerSettingsManager
    
    if enabled is not None:
        try:
            if isinstance(enabled, str):
                enabled = enabled.lower() in ("true", "1", "yes")
            settings_manager = ShowManagerSettingsManager(facade, block_id)
            settings_manager.sync_on_change = bool(enabled)
            settings_manager.force_save()
            status = "enabled" if enabled else "disabled"
            return {"success": True, "message": f"Sync on change: {status}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to set sync on change: {e}"}
    
    try:
        settings_manager = ShowManagerSettingsManager(facade, block_id)
        current = settings_manager.sync_on_change
    except Exception:
        current = True
    
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["true", "false"],
        "labels": ["Enabled", "Disabled"],
        "default": "true" if current else "false",
        "title": "Sync on Change?"
    }


@quick_action(
    "ShowManager",
    "Reconcile Layers",
    description="Check and resolve divergences using configured strategy",
    category=ActionCategory.EXECUTE,
    icon="check-circle"
)
def action_reconcile_layers(facade, block_id: str, strategy: str = None, **kwargs):
    """
    Check all synced layers for divergences and resolve them.
    
    Uses the configured conflict resolution strategy to automatically
    resolve any detected divergences between Editor and MA3.
    """
    from src.application.settings.show_manager_settings import ShowManagerSettingsManager
    
    try:
        ssm = facade.sync_system_manager(block_id)
        if not ssm:
            return {"success": False, "error": "SyncSystemManager not available"}
        
        # Check for divergences (returns list of entity IDs)
        diverged_ids = ssm.check_divergences()
        if not diverged_ids:
            return {"success": True, "message": "No divergences detected"}
        
        # Use provided strategy or fall back to configured default
        if not strategy:
            try:
                settings_manager = ShowManagerSettingsManager(facade, block_id)
                strategy = settings_manager.conflict_resolution_strategy or "ez_wins"
            except Exception:
                strategy = "ez_wins"
        
        # Resolve each divergence
        resolved = 0
        errors = []
        for entity_id in diverged_ids:
            try:
                ssm.resolve_divergence(entity_id, strategy=strategy)
                resolved += 1
            except Exception as e:
                errors.append(str(e))
        
        if errors:
            return {
                "success": False,
                "error": f"Resolved {resolved}/{len(diverged_ids)}. Errors: {'; '.join(errors[:3])}"
            }
        return {"success": True, "message": f"Resolved {resolved} divergence(s) using '{strategy}'"}
    except Exception as e:
        return {"success": False, "error": f"Reconciliation failed: {e}"}


@quick_action(
    "ShowManager",
    "Start Listener",
    description="Start the OSC listener for MA3 communication",
    category=ActionCategory.EXECUTE,
    icon="play"
)
def action_start_listener(facade, block_id: str, **kwargs):
    """
    Start the OSC listener to receive messages from MA3.
    
    Uses the configured listen address and port from ShowManager settings.
    """
    from src.application.settings.show_manager_settings import ShowManagerSettingsManager
    
    try:
        if not facade.show_manager_listener_service:
            return {"success": False, "error": "Listener service not available"}
        
        settings_manager = ShowManagerSettingsManager(facade, block_id)
        listen_port = settings_manager.listen_port
        listen_address = settings_manager.listen_address
        
        success, error_message = facade.show_manager_listener_service.start_listener(
            block_id=block_id,
            listen_port=listen_port,
            listen_address=listen_address
        )
        
        if success or error_message == "Listener is already running":
            return {"success": True, "message": f"Listener started on {listen_address}:{listen_port}"}
        return {"success": False, "error": f"Failed to start listener: {error_message}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to start listener: {e}"}


@quick_action(
    "ShowManager",
    "Stop Listener",
    description="Stop the OSC listener",
    category=ActionCategory.EXECUTE,
    icon="stop",
    dangerous=True
)
def action_stop_listener(facade, block_id: str, **kwargs):
    """
    Stop the OSC listener, disconnecting from MA3.
    """
    try:
        if not facade.show_manager_listener_service:
            return {"success": False, "error": "Listener service not available"}
        
        facade.show_manager_listener_service.stop_listener(block_id=block_id)
        return {"success": True, "message": "Listener stopped"}
    except Exception as e:
        return {"success": False, "error": f"Failed to stop listener: {e}"}


# ============================================================================
# DetectOnsets Actions
# ============================================================================

@quick_action(
    "DetectOnsets",
    "Tune Sensitivity",
    description="Adjust onset detection sensitivity",
    category=ActionCategory.CONFIGURE,
    icon="slider",
    primary=True
)
def action_tune_sensitivity(facade, block_id: str, value: float = None, **kwargs):
    """
    Tune onset detection sensitivity (threshold) via settings manager.
    
    Single source of truth: block.metadata in database.
    Settings manager ensures consistency with panel.
    """
    from src.application.settings.detect_onsets_settings import DetectOnsetsSettingsManager
    
    if value is not None:
        # Write path: set the threshold
        try:
            settings_manager = DetectOnsetsSettingsManager(facade, block_id)
            settings_manager.onset_threshold = float(value)
            # Force immediate save (bypasses debounce) to ensure BlockUpdated event fires right away
            # This ensures panel refreshes immediately when quick action changes setting
            settings_manager.force_save()
            return {"success": True, "message": f"Onset sensitivity set to {value:.2f}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = DetectOnsetsSettingsManager(facade, block_id)
        current_threshold = settings_manager.onset_threshold  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_threshold = 0.5
    
    return {
        "needs_input": True,
        "input_type": "number",
        "min": 0.0,
        "max": 1.0,
        "default": current_threshold,  # Current value from database (single source of truth)
        "step": 0.05,
        "increment_jump": 0.05,  # Step size for increment/decrement arrows
        "decimals": 2,
        "title": "Onset Sensitivity (0.0-1.0)"
    }


