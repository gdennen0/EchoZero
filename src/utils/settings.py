"""
Settings management for EchoZero application

This module handles persistent application settings like last opened folders,
window positions, and user preferences.
"""

import json
import os
from src.utils.message import Log
from src.utils.paths import get_settings_path

# Settings file path (uses platform-specific user config directory)
SETTINGS_FILE = str(get_settings_path())

# Default settings
DEFAULT_SETTINGS = {
    # Path settings
    "last_project_folder": os.path.expanduser("~"),  # Default to user's home directory
    "recent_folders": [],
    "dialog_paths": {},  # Per-dialog remembered paths
    "default_project_directory": os.path.expanduser("~"),
    
    # Window settings
    "window_geometry": None,
    
    # Startup settings
    "autoload_recent_project": True,  # Load most recent project on startup
    "show_welcome_on_startup": False,  # Show welcome/tips dialog
    
    # Audio settings
    "default_sample_rate": 44100,
    "audio_buffer_size": 512,
    
    # Editor settings
    "auto_connect_blocks": True,  # Auto-connect compatible ports when adding blocks
    "confirm_block_deletion": True,  # Show confirmation before deleting blocks
    "snap_to_grid": True,  # Snap nodes to grid in node editor
    "grid_size": 20,  # Grid size for node editor
    
    # Processing settings
    "max_undo_steps": 50,
    "autosave_enabled": False,
    "autosave_interval_minutes": 5,
}

class Settings:
    """Application settings manager"""
    
    def __init__(self):
        self.settings = DEFAULT_SETTINGS.copy()
        self.load_settings()
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as file:
                    saved_settings = json.load(file)
                    # Update defaults with saved settings
                    self.settings.update(saved_settings)
                    Log.info("Settings loaded successfully")
            else:
                # Create data directory if it doesn't exist
                os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
                self.save_settings()
                Log.info("Created new settings file with defaults")
        except Exception as e:
            Log.error(f"Failed to load settings: {e}")
            self.settings = DEFAULT_SETTINGS.copy()
    
    def save_settings(self):
        """Save settings to file"""
        try:
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as file:
                json.dump(self.settings, file, indent=4)
            Log.info("Settings saved successfully")
        except Exception as e:
            Log.error(f"Failed to save settings: {e}")
    
    def get(self, key, default=None):
        """Get a setting value"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Set a setting value"""
        self.settings[key] = value
        self.save_settings()
    
    def get_last_project_folder(self):
        """Get the last used project folder"""
        folder = self.get("last_project_folder", os.path.expanduser("~"))
        # Verify the folder still exists
        if not os.path.exists(folder):
            folder = os.path.expanduser("~")
            self.set("last_project_folder", folder)
        return folder
    
    def set_last_project_folder(self, folder_path):
        """Set the last used project folder"""
        if folder_path and os.path.exists(folder_path):
            # If a file path was provided, get the directory
            if os.path.isfile(folder_path):
                folder_path = os.path.dirname(folder_path)
            
            self.set("last_project_folder", folder_path)
            Log.info(f"Last project folder updated to: {folder_path}")
    
    def add_recent_folder(self, folder_path):
        """Add a folder to the recent folders list"""
        if not folder_path or not os.path.exists(folder_path):
            return
        
        # If a file path was provided, get the directory
        if os.path.isfile(folder_path):
            folder_path = os.path.dirname(folder_path)
        
        recent_folders = self.get("recent_folders", [])
        
        # Remove if already in list
        if folder_path in recent_folders:
            recent_folders.remove(folder_path)
        
        # Add to front of list
        recent_folders.insert(0, folder_path)
        
        # Keep only last 10 folders
        recent_folders = recent_folders[:10]
        
        self.set("recent_folders", recent_folders)
    
    def get_dialog_path(self, dialog_name: str) -> str:
        """Get the last used path for a specific dialog.
        
        Args:
            dialog_name: Unique identifier for the dialog (e.g., 'open_project', 'load_audio')
            
        Returns:
            The last used directory path, or user's home directory if not set
        """
        dialog_paths = self.get("dialog_paths", {})
        path = dialog_paths.get(dialog_name, "")
        
        # Verify the path still exists, fallback to home
        if path and os.path.exists(path):
            return path
        return os.path.expanduser("~")
    
    def set_dialog_path(self, dialog_name: str, path: str):
        """Set the last used path for a specific dialog.
        
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
            dialog_paths = self.get("dialog_paths", {})
            dialog_paths[dialog_name] = path
            self.set("dialog_paths", dialog_paths)

# Global settings instance
app_settings = Settings()
