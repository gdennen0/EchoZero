"""
Utils module - Logging, paths, and utility functions.

Standard Python naming convention (lowercase).

Contents:
- message.py: Log class for application logging
- paths.py: Platform-specific path utilities
- tools.py: General utility functions
- settings.py: Legacy settings utilities
- recent_projects.py: Recent projects tracking
- pycache_cleaner.py: Pycache cleanup utilities
"""
from src.utils.message import Log
from src.utils.paths import (
    get_user_data_dir,
    get_user_config_dir,
    get_user_cache_dir,
    get_logs_dir,
    get_database_path,
    get_settings_path,
    get_projects_dir,
    get_models_dir,
    get_app_install_dir,
    get_project_root,
    get_recent_projects_path,
    get_debug_log_path,
    ensure_user_directories,
)

__all__ = [
    'Log',
    'get_user_data_dir',
    'get_user_config_dir',
    'get_user_cache_dir',
    'get_logs_dir',
    'get_database_path',
    'get_settings_path',
    'get_projects_dir',
    'get_models_dir',
    'get_app_install_dir',
    'get_project_root',
    'get_recent_projects_path',
    'get_debug_log_path',
    'ensure_user_directories',
]
