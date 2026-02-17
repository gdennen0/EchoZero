"""
Block Settings Manager

Provides a standardized foundation for block-level settings managers.
Block settings are stored in block.metadata (via ApplicationFacade) rather than
preferences, and integrate with facade.command_bus for undo support.

Usage:
    1. Create a settings schema dataclass
    2. Inherit from BlockSettingsManager
    3. Implement property accessors for type-safe access
    
See: AgentAssets/SETTINGS_ABSTRACTION_PRESET.md for complete guide
"""
from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Any, Type, TYPE_CHECKING
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .base_settings import BaseSettings
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.application.commands.command_bus import CommandBus
    from src.application.commands.block_commands import BatchUpdateMetadataCommand

class BlockSettingsManager(QObject):
    """
    Base class for block-level settings managers.
    
    Block settings are stored in block.metadata (via ApplicationFacade) and
    integrated with facade.command_bus for undo support.
    
    Provides:
    - Type-safe property access pattern
    - Automatic persistence to block.metadata
    - Auto-save with configurable debouncing
    - Signal emission when settings change
    - Undo support via facade.command_bus
    - Backwards-compatible loading (handles missing fields)
    
    Subclasses must:
    1. Define SETTINGS_CLASS class attribute
    2. Implement property accessors for type-safe access
    
    Example:
        class MyBlockSettingsManager(BlockSettingsManager):
            SETTINGS_CLASS = MyBlockSettings
            
            @property
            def model(self) -> str:
                return self._settings.model
            
            @model.setter
            def model(self, value: str):
                if value != self._settings.model:
                    self._settings.model = value
                    self._save_setting('model')
    """
    
    # Signals
    settings_changed = pyqtSignal(str)  # Setting name that changed
    settings_loaded = pyqtSignal()
    
    # Must be defined by subclasses
    SETTINGS_CLASS: Type[BaseSettings] = BaseSettings
    
    # Configuration
    SAVE_DEBOUNCE_MS: int = 300  # Debounce delay for saves
    
    def __init__(self, facade: 'ApplicationFacade', block_id: str, parent=None):
        """
        Initialize the block settings manager.
        
        Args:
            facade: ApplicationFacade instance for updating block metadata
            block_id: ID of the block these settings belong to
            parent: Parent QObject
        """
        super().__init__(parent)
        
        if not self.SETTINGS_CLASS or self.SETTINGS_CLASS == BaseSettings:
            raise ValueError(f"{self.__class__.__name__} must define SETTINGS_CLASS")
        
        self._facade = facade
        self._block_id = block_id
        self._settings: BaseSettings = self.SETTINGS_CLASS()
        self._loaded = False
        
        # Debounce timer for saves
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(self.SAVE_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self._do_save)
        self._pending_save = False
        self._pending_changes: Dict[str, Any] = {}
        
        # Load settings from block metadata
        self._load_from_storage()
    
    # =========================================================================
    # Generic Access
    # =========================================================================
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key."""
        return getattr(self._settings, key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings as a dictionary."""
        return self._settings.to_dict()
    
    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        self._settings = self.SETTINGS_CLASS()
        self._do_save(immediate=True)
        self.settings_loaded.emit()
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _load_from_storage(self):
        """
        Load settings from block.metadata.
        
        This is the single source of truth - always reads from database.
        Call this after undo/redo or when block metadata changes externally.
        """
        try:
            result = self._facade.describe_block(self._block_id)
            if not result.success or not result.data:
                Log.warning(f"{self.__class__.__name__}: Block not found: {self._block_id}")
                self._loaded = True
                return
            
            block = result.data
            metadata = block.metadata or {}
            
            # CRITICAL: If metadata is empty, don't overwrite with defaults!
            # This happens during project load when blocks are created before metadata is set.
            # Only load if metadata actually has content (or we're doing initial load and haven't loaded yet)
            if not metadata and self._loaded:
                # Already loaded once, and metadata is now empty - don't overwrite with defaults
                Log.debug(
                    f"{self.__class__.__name__}: Metadata is empty but already loaded, "
                    f"skipping reload to avoid overwriting with defaults for block {self._block_id}"
                )
                return
            
            # Handle backwards compatibility: old blocks had settings nested under "settings" key
            # New blocks have settings at metadata top level
            if "settings" in metadata and isinstance(metadata.get("settings"), dict):
                # Old format: settings nested under "settings" key
                settings_dict = metadata["settings"]
                # Merge nested settings with top-level metadata for backwards compatibility
                merged_metadata = {**metadata}
                merged_metadata.update(settings_dict)
                # Remove the nested "settings" key to avoid confusion
                merged_metadata.pop("settings", None)
                self._settings = self.SETTINGS_CLASS.from_dict(merged_metadata)
            else:
                # New format: settings at metadata top level
                # Only load if metadata has content, otherwise keep current settings
                if metadata:
                    self._settings = self.SETTINGS_CLASS.from_dict(metadata)
                else:
                    # If metadata is empty and we haven't loaded yet, use defaults (initial load)
                    # This is fine for new blocks that don't have settings yet
                    if not self._loaded:
                        self._settings = self.SETTINGS_CLASS()
            
            self._loaded = True
            self.settings_loaded.emit()
            
            
            Log.debug(
                f"{self.__class__.__name__}: Loaded settings from block {self._block_id} metadata "
                f"({len(metadata)} keys): {list(metadata.keys())[:5]}..."
            )
            Log.debug(
                f"{self.__class__.__name__}: Settings values after load: {list(self._settings.to_dict().items())[:5]}..."
            )
            
        except Exception as e:
            Log.error(f"{self.__class__.__name__}: Failed to load settings: {e}")
            self._loaded = True
    
    def reload_from_storage(self):
        """
        Public method to reload settings from database.
        
        Use this when you know block metadata has changed externally
        (e.g., after undo/redo, or from quick actions).
        """
        self._load_from_storage()
    
    def _save_setting(self, key: str):
        """
        Queue a save operation (debounced).
        
        Tracks that settings have changed, then saves all settings at once
        for better performance and undo support.
        """
        
        # Mark that we have pending changes
        self._pending_changes[key] = True  # Just track that this key changed
        
        # Start debounce timer
        self._pending_save = True
        self._save_timer.start()
        
        
        
        # Emit signal immediately for UI responsiveness
        self.settings_changed.emit(key)
    
    def _do_save(self, immediate: bool = False):
        """
        Actually persist settings to block.metadata via facade.
        
        Uses BatchUpdateMetadataCommand for undo support.
        Saves all current settings (from _settings.to_dict()) to ensure
        proper conversion of nested structures.
        """
        
        if not self._pending_changes and not immediate:
            
            self._pending_save = False
            return
        
        try:
            # Get all settings (to_dict() handles nested structures correctly)
            all_settings = self._settings.to_dict()
            
            
            Log.debug(
                f"{self.__class__.__name__}: Saving {len(self._pending_changes)} setting(s) to block {self._block_id}: "
                f"{', '.join(self._pending_changes.keys())}"
            )
            Log.debug(
                f"{self.__class__.__name__}: Settings values: {all_settings}"
            )
            
            # Use facade.command_bus for undo support
            from src.application.commands.block_commands import BatchUpdateMetadataCommand
            
            cmd = BatchUpdateMetadataCommand(
                self._facade,
                self._block_id,
                all_settings,
                description=f"Update {len(self._pending_changes)} setting(s)"
            )
            # Get command_bus from facade (explicit dependency)
            if not self._facade.command_bus:
                Log.error(f"{self.__class__.__name__}: Cannot save settings - command_bus not initialized")
                return False
            
            result = self._facade.command_bus.execute(cmd)
            
            
            
            if result:
                Log.info(
                    f"{self.__class__.__name__}: Successfully queued save of {len(self._pending_changes)} setting(s) "
                    f"to block {self._block_id}"
                )
            else:
                Log.error(
                    f"{self.__class__.__name__}: Failed to save settings: command_bus.execute() returned False"
                )
            
            # Clear pending changes
            self._pending_changes.clear()
            self._pending_save = False
            
        except Exception as e:
            Log.error(f"{self.__class__.__name__}: Failed to save settings: {e}", exc_info=True)
            self._pending_save = False
    
    def force_save(self):
        """Force an immediate save (bypasses debounce)."""
        self._save_timer.stop()
        self._do_save(immediate=True)
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def is_loaded(self) -> bool:
        """Check if settings have been loaded from storage."""
        return self._loaded
    
    def has_pending_save(self) -> bool:
        """Check if there's a pending save operation."""
        return self._pending_save
    
    def _update_expected_outputs(self):
        """
        Update expected_outputs in block metadata when configuration changes.
        
        This should be called by subclasses when a setting changes that affects
        what outputs the block will produce (e.g., separator two_stems mode).
        
        The expected_outputs are used by:
        - Filter UI to show what outputs are expected
        - Validation to check if actual outputs match expected
        """
        if not self._facade.current_project_id:
            return
        
        try:
            # Get block with latest settings merged
            block_result = self._facade.describe_block(self._block_id)
            if not block_result.success or not block_result.data:
                return
            
            block = block_result.data
            
            # Merge current settings so get_expected_outputs() sees latest config
            # (important if _save_setting() is debounced)
            block.metadata.update(self._settings.to_dict())
            
            # Get processor from execution engine
            execution_engine = self._facade.execution_engine
            processor = execution_engine.get_processor(block)
            
            if not processor:
                return
            
            # Calculate and save expected outputs using ExpectedOutputsService
            expected_outputs_service = getattr(self._facade, 'expected_outputs_service', None)
            if expected_outputs_service:
                expected_outputs = expected_outputs_service.calculate_expected_outputs(
                    block,
                    processor,
                    facade=self._facade
                )
                block.metadata['expected_outputs'] = expected_outputs
                self._facade.block_service.update_block(
                    self._facade.current_project_id,
                    block.id,
                    block
                )
            else:
                # Fallback: use processor directly (for blocks without connection-based outputs)
                expected_outputs = processor.get_expected_outputs(block)
                block.metadata['expected_outputs'] = expected_outputs
                self._facade.block_service.update_block(
                    self._facade.current_project_id,
                    block.id,
                    block
                )
        except Exception as e:
            # Non-critical - don't fail if update fails
            Log.debug(f"BlockSettingsManager: Failed to update expected_outputs: {e}")