"""
Shared utilities module.

Provides cross-cutting utility functions and classes.

Usage:
    from src.shared.utils import Log
    Log.info("Message")
"""
# Re-export from existing location for backwards compatibility
from src.utils.message import Log
from src.utils.paths import (
    get_user_data_dir,
    get_user_config_dir,
    get_user_cache_dir,
    get_logs_dir,
    get_database_path,
    get_projects_dir,
    get_models_dir,
    get_app_install_dir,
    ensure_user_directories,
)

__all__ = [
    'Log',
    'get_user_data_dir',
    'get_user_config_dir',
    'get_user_cache_dir',
    'get_logs_dir',
    'get_database_path',
    'get_projects_dir',
    'get_models_dir',
    'get_app_install_dir',
    'ensure_user_directories',
]
