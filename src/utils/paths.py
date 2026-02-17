"""
Path management for EchoZero application

Handles platform-specific user data directories following standard conventions:
- macOS: ~/Library/Application Support/EchoZero/
- Linux: ~/.local/share/echozero/
- Windows: %APPDATA%/EchoZero/

Application code should be separate from user data.
"""
import os
import sys
import shutil
from pathlib import Path
from typing import Optional


# Application name
APP_NAME = "EchoZero"
APP_AUTHOR = "EchoZero"


def get_user_data_dir() -> Path:
    """
    Get platform-specific user data directory.
    
    Returns:
        Path to user data directory where settings, databases, and user files are stored.
        
    Platform conventions:
    - macOS: ~/Library/Application Support/EchoZero/
    - Linux: ~/.local/share/echozero/
    - Windows: %APPDATA%/EchoZero/
    """
    system = sys.platform
    
    if system == "darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    elif system == "win32":  # Windows
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:  # Linux and other Unix-like
        base = Path.home() / ".local" / "share"
    
    user_data_dir = base / APP_NAME
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


def get_user_config_dir() -> Path:
    """
    Get platform-specific user config directory.
    
    Returns:
        Path to user config directory (usually same as user_data_dir on macOS/Windows,
        but separate on Linux: ~/.config/echozero/)
    """
    system = sys.platform
    
    if system == "darwin":  # macOS
        # macOS uses Application Support for both data and config
        return get_user_data_dir()
    elif system == "win32":  # Windows
        # Windows uses AppData/Roaming for both
        return get_user_data_dir()
    else:  # Linux
        config_dir = Path.home() / ".config" / "echozero"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir


def get_user_cache_dir() -> Path:
    """
    Get platform-specific user cache directory.
    
    Returns:
        Path to user cache directory for temporary files.
    """
    system = sys.platform
    
    if system == "darwin":  # macOS
        cache_dir = Path.home() / "Library" / "Caches" / APP_NAME
    elif system == "win32":  # Windows
        cache_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME / "Cache"
    else:  # Linux
        cache_dir = Path.home() / ".cache" / "echozero"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_recent_projects_path() -> Path:
    """
    Path for storing recent project metadata.
    """
    recent_file = get_user_data_dir() / "recent_projects.json"
    recent_file.parent.mkdir(parents=True, exist_ok=True)
    return recent_file


def get_logs_dir() -> Path:
    """
    Get directory for application logs.
    
    Returns:
        Path to logs directory (stored in user data directory).
    """
    logs_dir = get_user_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_database_path(db_name: str = "ez") -> Path:
    """
    Get path to SQLite database file.
    
    Args:
        db_name: Name of database file (without extension)
        
    Returns:
        Path to database file in user data directory.
    """
    return get_user_data_dir() / f"{db_name}.db"


def get_settings_path() -> Path:
    """
    Get path to settings file.
    
    Returns:
        Path to settings.json in user config directory.
    """
    return get_user_config_dir() / "settings.json"


def get_projects_dir() -> Path:
    """
    Get directory where user projects are stored.
    
    Users can override this location, but this is the default.
    
    Returns:
        Path to projects directory in user data directory.
    """
    projects_dir = get_user_data_dir() / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return projects_dir


def get_models_dir() -> Path:
    """
    Get directory where ML models are stored.
    
    Models downloaded automatically or provided by the application
    are stored here.
    
    Returns:
        Path to models directory in user data directory.
    """
    models_dir = get_user_data_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_debug_log_path() -> Path:
    """
    Get path to debug log file.
    
    Returns:
        Path to debug.log file in logs directory.
        Used for development/debugging purposes.
    """
    debug_log = get_logs_dir() / "debug.log"
    debug_log.parent.mkdir(parents=True, exist_ok=True)
    return debug_log


def get_app_install_dir() -> Optional[Path]:
    """
    Get the application installation directory.
    
    This is where the application code lives (not user data).
    Useful for finding bundled resources, templates, etc.
    
    Returns:
        Path to application installation directory, or None if not determinable.
    """
    # If running as frozen (PyInstaller or similar), use bundle root
    if getattr(sys, 'frozen', False):
        # One-file: bundled files are in sys._MEIPASS; one-folder: next to executable
        bundle_dir = getattr(sys, '_MEIPASS', None)
        if bundle_dir:
            return Path(bundle_dir)
        return Path(sys.executable).parent
    
    # If running from source, use the project root
    # This assumes the project structure: .../EchoZero/src/utils/paths.py
    current_file = Path(__file__).resolve()
    # Navigate from src/utils/paths.py to project root
    project_root = current_file.parent.parent.parent
    return project_root


def get_project_workspace_dir(project_id: str) -> Path:
    """
    Get the workspace directory for a project's runtime data files.
    
    This is where extracted project data (audio files, waveforms, etc.) lives
    during a session. The .ez file is the source of truth; this directory is
    a transient cache that is re-created from the .ez file on each load.
    
    Args:
        project_id: Unique project identifier
        
    Returns:
        Path to project workspace directory (created if needed).
        
    Platform locations:
    - macOS: ~/Library/Caches/EchoZero/projects/{project_id}/
    - Linux: ~/.cache/echozero/projects/{project_id}/
    - Windows: %LOCALAPPDATA%/EchoZero/Cache/projects/{project_id}/
    """
    workspace_dir = get_user_cache_dir() / "projects" / project_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def cleanup_project_workspace(project_id: str) -> None:
    """
    Remove the workspace directory for a project.
    
    Called before extracting a new project to ensure a clean workspace,
    or on project close to free disk space.
    
    Args:
        project_id: Unique project identifier
    """
    workspace_dir = get_user_cache_dir() / "projects" / project_id
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)


def ensure_user_directories():
    """
    Ensure all user directories exist.
    
    Creates all necessary directories if they don't exist.
    """
    get_user_data_dir()
    get_user_config_dir()
    get_user_cache_dir()
    get_logs_dir()
    get_projects_dir()
    get_models_dir()

