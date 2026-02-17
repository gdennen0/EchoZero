"""
Application Settings Manager

Global settings for the entire EchoZero application.

These settings apply across all windows, blocks, and sessions.

Usage:
    app_settings = AppSettingsManager(preferences_repo)
    
    # Read settings
    preset = app_settings.theme_preset
    
    # Write settings (auto-saves)
    app_settings.theme_preset = "nasa"
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json
import os

from .base_settings import BaseSettings, BaseSettingsManager

if TYPE_CHECKING:
    from src.infrastructure.persistence.sqlite.preferences_repository_impl import PreferencesRepository


@dataclass
class AppSettings(BaseSettings):
    """
    Global application settings schema.
    
    Add new application-wide settings here.
    All fields should have default values for backwards compatibility.
    """
    
    # Theme settings
    theme_preset: str = "default dark"  # Theme preset name (case-insensitive)
    sharp_corners: bool = False  # When True, all UI elements use 0px border-radius (no rounded edges)
    
    # Window state
    window_width: int = 1400
    window_height: int = 900
    window_x: int = 100
    window_y: int = 100
    window_maximized: bool = False
    
    # Recent files
    recent_projects: List[str] = field(default_factory=list)
    max_recent_projects: int = 10
    last_project_path: str = ""
    
    # Dialog remembered paths
    dialog_paths: Dict[str, str] = field(default_factory=dict)
    
    # Auto-save settings
    auto_save_enabled: bool = True
    auto_save_interval_seconds: int = 300  # 5 minutes
    
    # Startup behavior
    restore_last_project: bool = True
    show_welcome_on_startup: bool = True
    
    # Performance
    max_undo_steps: int = 100
    cache_enabled: bool = True
    
    # Audio settings
    default_sample_rate: int = 44100
    audio_buffer_size: int = 1024
    default_waveform_resolution: int = 20  # Points per second (very low, ~2KB per minute)
    
    # Audio device settings (empty string = system default)
    audio_output_device_id: str = ""  # QAudioDevice.id() as string
    audio_input_device_id: str = ""   # QAudioDevice.id() as string
    
    # Node editor settings
    snap_to_grid: bool = True
    grid_size: int = 20
    auto_connect_blocks: bool = True
    
    # MA3 Communication settings
    ma3_listen_enabled: bool = True
    ma3_listen_port: int = 9000
    ma3_listen_address: str = "127.0.0.1"
    ma3_send_enabled: bool = True
    ma3_send_port: int = 9001  # Port MA3 listens on for EchoZero messages
    ma3_send_address: str = "127.0.0.1"
    
    # Execution settings
    execution_stop_on_error: bool = True  # Stop execution on first block failure (fail-fast mode)
    show_execution_report: bool = True  # Show execution summary dialog after block execution
    confirm_block_deletion: bool = True
    use_subprocess_runner: bool = False  # Run blocks in separate process (no GIL; good for training)
    
    # Default directories
    default_project_directory: str = ""
    
    # Developer settings
    debug_mode: bool = False
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    filter_repetitive_logs: bool = True  # Filter out repetitive DEBUG messages (cache hits, status checks, etc.)
    clear_pycache_on_startup: bool = True  # Clear __pycache__ directories on startup (prevents stale bytecode issues)
    
    # Migration tracking (internal)
    _migrated_from_file: bool = False


class AppSettingsManager(BaseSettingsManager):
    """
    Manager for global application settings.
    
    Singleton pattern recommended - use get_app_settings_manager().
    """
    
    NAMESPACE = "app"
    SETTINGS_CLASS = AppSettings
    
    def __init__(self, preferences_repo: Optional['PreferencesRepository'] = None, parent=None):
        super().__init__(preferences_repo, parent)
    
    # =========================================================================
    # Theme Properties
    # =========================================================================
    
    @property
    def theme_preset(self) -> str:
        return self._settings.theme_preset
    
    @theme_preset.setter
    def theme_preset(self, value: str):
        if value != self._settings.theme_preset:
            self._settings.theme_preset = value.lower()
            self._save_setting('theme_preset')
    
    @property
    def sharp_corners(self) -> bool:
        return self._settings.sharp_corners
    
    @sharp_corners.setter
    def sharp_corners(self, value: bool):
        if value != self._settings.sharp_corners:
            self._settings.sharp_corners = value
            self._save_setting('sharp_corners')
    
    # =========================================================================
    # Window State Properties
    # =========================================================================
    
    @property
    def window_width(self) -> int:
        return self._settings.window_width
    
    @window_width.setter
    def window_width(self, value: int):
        if value != self._settings.window_width:
            self._settings.window_width = max(800, value)
            self._save_setting('window_width')
    
    @property
    def window_height(self) -> int:
        return self._settings.window_height
    
    @window_height.setter
    def window_height(self, value: int):
        if value != self._settings.window_height:
            self._settings.window_height = max(600, value)
            self._save_setting('window_height')
    
    @property
    def window_x(self) -> int:
        return self._settings.window_x
    
    @window_x.setter
    def window_x(self, value: int):
        if value != self._settings.window_x:
            self._settings.window_x = max(0, value)
            self._save_setting('window_x')
    
    @property
    def window_y(self) -> int:
        return self._settings.window_y
    
    @window_y.setter
    def window_y(self, value: int):
        if value != self._settings.window_y:
            self._settings.window_y = max(0, value)
            self._save_setting('window_y')
    
    @property
    def window_maximized(self) -> bool:
        return self._settings.window_maximized
    
    @window_maximized.setter
    def window_maximized(self, value: bool):
        if value != self._settings.window_maximized:
            self._settings.window_maximized = value
            self._save_setting('window_maximized')
    
    def save_window_geometry(self, x: int, y: int, width: int, height: int, maximized: bool):
        """Save all window geometry at once."""
        self._settings.window_x = max(0, x)
        self._settings.window_y = max(0, y)
        self._settings.window_width = max(800, width)
        self._settings.window_height = max(600, height)
        self._settings.window_maximized = maximized
        self._save_setting('window_geometry')
    
    # =========================================================================
    # Recent Projects Properties
    # =========================================================================
    
    @property
    def recent_projects(self) -> List[str]:
        return self._settings.recent_projects.copy()
    
    def add_recent_project(self, path: str):
        """Add a project to recent projects list."""
        projects = self._settings.recent_projects
        if path in projects:
            projects.remove(path)
        projects.insert(0, path)
        # Keep only max recent
        self._settings.recent_projects = projects[:self._settings.max_recent_projects]
        self._settings.last_project_path = path
        self._save_setting('recent_projects')
    
    def clear_recent_projects(self):
        """Clear the recent projects list."""
        self._settings.recent_projects = []
        self._save_setting('recent_projects')
    
    @property
    def last_project_path(self) -> str:
        return self._settings.last_project_path
    
    # =========================================================================
    # Dialog Paths Properties
    # =========================================================================
    
    def get_dialog_path(self, dialog_name: str) -> str:
        """
        Get the last used path for a specific dialog.
        
        Args:
            dialog_name: Unique identifier for the dialog (e.g., 'open_project', 'load_audio')
            
        Returns:
            The last used directory path, or user's home directory if not set
        """
        path = self._settings.dialog_paths.get(dialog_name, "")
        
        # Verify the path still exists, fallback to home
        if path and os.path.exists(path):
            return path
        return os.path.expanduser("~")
    
    def set_dialog_path(self, dialog_name: str, path: str):
        """
        Set the last used path for a specific dialog.
        
        Args:
            dialog_name: Unique identifier for the dialog
            path: File or directory path (directories are extracted from file paths)
        """
        if not path:
            return
        
        # If a file path was provided, get the directory
        if os.path.isfile(path):
            path = os.path.dirname(path)
        
        # Only save if the directory exists
        if os.path.exists(path):
            self._settings.dialog_paths[dialog_name] = path
            self._save_setting('dialog_paths')
    
    # =========================================================================
    # Auto-Save Properties
    # =========================================================================
    
    @property
    def auto_save_enabled(self) -> bool:
        return self._settings.auto_save_enabled
    
    @auto_save_enabled.setter
    def auto_save_enabled(self, value: bool):
        if value != self._settings.auto_save_enabled:
            self._settings.auto_save_enabled = value
            self._save_setting('auto_save_enabled')
    
    @property
    def auto_save_interval_seconds(self) -> int:
        return self._settings.auto_save_interval_seconds
    
    @auto_save_interval_seconds.setter
    def auto_save_interval_seconds(self, value: int):
        if value != self._settings.auto_save_interval_seconds:
            self._settings.auto_save_interval_seconds = max(60, min(value, 3600))
            self._save_setting('auto_save_interval_seconds')
    
    # =========================================================================
    # Startup Properties
    # =========================================================================
    
    @property
    def restore_last_project(self) -> bool:
        return self._settings.restore_last_project
    
    @restore_last_project.setter
    def restore_last_project(self, value: bool):
        if value != self._settings.restore_last_project:
            self._settings.restore_last_project = value
            self._save_setting('restore_last_project')
    
    @property
    def show_welcome_on_startup(self) -> bool:
        return self._settings.show_welcome_on_startup
    
    @show_welcome_on_startup.setter
    def show_welcome_on_startup(self, value: bool):
        if value != self._settings.show_welcome_on_startup:
            self._settings.show_welcome_on_startup = value
            self._save_setting('show_welcome_on_startup')
    
    # =========================================================================
    # Performance Properties
    # =========================================================================
    
    @property
    def max_undo_steps(self) -> int:
        return self._settings.max_undo_steps
    
    @max_undo_steps.setter
    def max_undo_steps(self, value: int):
        if value != self._settings.max_undo_steps:
            self._settings.max_undo_steps = max(10, min(value, 1000))
            self._save_setting('max_undo_steps')
    
    @property
    def cache_enabled(self) -> bool:
        return self._settings.cache_enabled
    
    @cache_enabled.setter
    def cache_enabled(self, value: bool):
        if value != self._settings.cache_enabled:
            self._settings.cache_enabled = value
            self._save_setting('cache_enabled')
    
    # =========================================================================
    # Audio Properties
    # =========================================================================
    
    @property
    def default_sample_rate(self) -> int:
        return self._settings.default_sample_rate
    
    @default_sample_rate.setter
    def default_sample_rate(self, value: int):
        valid_rates = {22050, 44100, 48000, 96000}
        if value in valid_rates and value != self._settings.default_sample_rate:
            self._settings.default_sample_rate = value
            self._save_setting('default_sample_rate')
    
    @property
    def audio_buffer_size(self) -> int:
        return self._settings.audio_buffer_size
    
    @audio_buffer_size.setter
    def audio_buffer_size(self, value: int):
        valid_sizes = {128, 256, 512, 1024, 2048}
        if value in valid_sizes and value != self._settings.audio_buffer_size:
            self._settings.audio_buffer_size = value
            self._save_setting('audio_buffer_size')
    
    @property
    def audio_output_device_id(self) -> str:
        """Selected audio output device ID (empty string = system default)."""
        return self._settings.audio_output_device_id
    
    @audio_output_device_id.setter
    def audio_output_device_id(self, value: str):
        if value != self._settings.audio_output_device_id:
            self._settings.audio_output_device_id = value
            self._save_setting('audio_output_device_id')
    
    @property
    def audio_input_device_id(self) -> str:
        """Selected audio input device ID (empty string = system default)."""
        return self._settings.audio_input_device_id
    
    @audio_input_device_id.setter
    def audio_input_device_id(self, value: str):
        if value != self._settings.audio_input_device_id:
            self._settings.audio_input_device_id = value
            self._save_setting('audio_input_device_id')
    
    # =========================================================================
    # Node Editor Properties
    # =========================================================================
    
    @property
    def snap_to_grid(self) -> bool:
        return self._settings.snap_to_grid
    
    @snap_to_grid.setter
    def snap_to_grid(self, value: bool):
        if value != self._settings.snap_to_grid:
            self._settings.snap_to_grid = value
            self._save_setting('snap_to_grid')
    
    @property
    def grid_size(self) -> int:
        return self._settings.grid_size
    
    @grid_size.setter
    def grid_size(self, value: int):
        if value != self._settings.grid_size:
            self._settings.grid_size = max(5, min(value, 50))
            self._save_setting('grid_size')
    
    @property
    def auto_connect_blocks(self) -> bool:
        return self._settings.auto_connect_blocks
    
    @auto_connect_blocks.setter
    def auto_connect_blocks(self, value: bool):
        if value != self._settings.auto_connect_blocks:
            self._settings.auto_connect_blocks = value
            self._save_setting('auto_connect_blocks')
    
    @property
    def confirm_block_deletion(self) -> bool:
        return self._settings.confirm_block_deletion
    
    @confirm_block_deletion.setter
    def confirm_block_deletion(self, value: bool):
        if value != self._settings.confirm_block_deletion:
            self._settings.confirm_block_deletion = value
            self._save_setting('confirm_block_deletion')

    @property
    def use_subprocess_runner(self) -> bool:
        return self._settings.use_subprocess_runner

    @use_subprocess_runner.setter
    def use_subprocess_runner(self, value: bool):
        if value != self._settings.use_subprocess_runner:
            self._settings.use_subprocess_runner = value
            self._save_setting('use_subprocess_runner')

    @property
    def default_project_directory(self) -> str:
        return self._settings.default_project_directory
    
    @default_project_directory.setter
    def default_project_directory(self, value: str):
        if value != self._settings.default_project_directory:
            self._settings.default_project_directory = value
            self._save_setting('default_project_directory')
    
    # =========================================================================
    # Developer Properties
    # =========================================================================
    
    @property
    def debug_mode(self) -> bool:
        return self._settings.debug_mode
    
    @debug_mode.setter
    def debug_mode(self, value: bool):
        if value != self._settings.debug_mode:
            self._settings.debug_mode = value
            self._save_setting('debug_mode')
    
    @property
    def log_level(self) -> str:
        return self._settings.log_level
    
    @log_level.setter
    def log_level(self, value: str):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if value in valid_levels and value != self._settings.log_level:
            self._settings.log_level = value
            self._save_setting('log_level')
            # Apply log level immediately
            from src.utils.message import Log
            Log.set_level(value)
    
    @property
    def filter_repetitive_logs(self) -> bool:
        return self._settings.filter_repetitive_logs
    
    @filter_repetitive_logs.setter
    def filter_repetitive_logs(self, value: bool):
        if value != self._settings.filter_repetitive_logs:
            self._settings.filter_repetitive_logs = value
            self._save_setting('filter_repetitive_logs')
            # Apply filter setting immediately
            from src.utils.message import Log
            Log.enable_repetitive_filter(value)
    
    @property
    def clear_pycache_on_startup(self) -> bool:
        return self._settings.clear_pycache_on_startup
    
    @clear_pycache_on_startup.setter
    def clear_pycache_on_startup(self, value: bool):
        if value != self._settings.clear_pycache_on_startup:
            self._settings.clear_pycache_on_startup = value
            self._save_setting('clear_pycache_on_startup')
    
    # =========================================================================
    # MA3 Communication Properties
    # =========================================================================
    
    @property
    def ma3_listen_enabled(self) -> bool:
        return self._settings.ma3_listen_enabled
    
    @ma3_listen_enabled.setter
    def ma3_listen_enabled(self, value: bool):
        if value != self._settings.ma3_listen_enabled:
            self._settings.ma3_listen_enabled = value
            self._save_setting('ma3_listen_enabled')
    
    @property
    def ma3_listen_port(self) -> int:
        return self._settings.ma3_listen_port
    
    @ma3_listen_port.setter
    def ma3_listen_port(self, value: int):
        if value != self._settings.ma3_listen_port:
            if 1 <= value <= 65535:
                self._settings.ma3_listen_port = value
                self._save_setting('ma3_listen_port')
    
    @property
    def ma3_listen_address(self) -> str:
        return self._settings.ma3_listen_address
    
    @ma3_listen_address.setter
    def ma3_listen_address(self, value: str):
        if value != self._settings.ma3_listen_address:
            self._settings.ma3_listen_address = value
            self._save_setting('ma3_listen_address')
    
    @property
    def ma3_send_enabled(self) -> bool:
        return self._settings.ma3_send_enabled
    
    @ma3_send_enabled.setter
    def ma3_send_enabled(self, value: bool):
        if value != self._settings.ma3_send_enabled:
            self._settings.ma3_send_enabled = value
            self._save_setting('ma3_send_enabled')
    
    @property
    def ma3_send_port(self) -> int:
        return self._settings.ma3_send_port
    
    @ma3_send_port.setter
    def ma3_send_port(self, value: int):
        if value != self._settings.ma3_send_port:
            if 1 <= value <= 65535:
                self._settings.ma3_send_port = value
                self._save_setting('ma3_send_port')
    
    @property
    def ma3_send_address(self) -> str:
        return self._settings.ma3_send_address
    
    @ma3_send_address.setter
    def ma3_send_address(self, value: str):
        if value != self._settings.ma3_send_address:
            self._settings.ma3_send_address = value
            self._save_setting('ma3_send_address')
    
    # =========================================================================
    # Custom Theme Persistence
    # =========================================================================
    
    _CUSTOM_THEMES_KEY = "app.custom_themes"
    
    def get_custom_themes(self) -> Dict[str, Any]:
        """Return all user-created custom themes from the DB.
        
        Returns:
            ``{theme_name: {"description": str, "colors": {attr: "#hex"}}}``
        """
        if not self._preferences_repo:
            return {}
        try:
            data = self._preferences_repo.get(self._CUSTOM_THEMES_KEY, {})
            if isinstance(data, str):
                data = json.loads(data)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    
    def save_custom_theme(self, name: str, description: str, colors_dict: Dict[str, str]):
        """Persist a custom theme to the DB.
        
        Args:
            name: Human-readable theme name.
            description: Short description.
            colors_dict: ``{attr_name: "#hex"}`` color mapping.
        """
        if not self._preferences_repo:
            return
        themes = self.get_custom_themes()
        themes[name] = {"description": description, "colors": colors_dict}
        self._preferences_repo.set(self._CUSTOM_THEMES_KEY, themes)
    
    def delete_custom_theme(self, name: str):
        """Remove a custom theme from the DB."""
        if not self._preferences_repo:
            return
        themes = self.get_custom_themes()
        lower_key = None
        for key in themes:
            if key.lower() == name.lower():
                lower_key = key
                break
        if lower_key:
            del themes[lower_key]
            self._preferences_repo.set(self._CUSTOM_THEMES_KEY, themes)


# =============================================================================
# Factory Function (for backward compatibility)
# =============================================================================

def init_app_settings_manager(preferences_repo: 'PreferencesRepository') -> AppSettingsManager:
    """
    Create an AppSettingsManager instance.
    
    Note: This is a factory function, not a singleton. The instance should be
    stored in ServiceContainer and passed as a dependency.
    """
    return AppSettingsManager(preferences_repo)








