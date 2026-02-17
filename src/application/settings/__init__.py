"""
Application Settings Module

Provides standardized settings management across the application.

Classes:
    BaseSettings: Base dataclass for settings schemas
    BaseSettingsManager: Abstract base for settings managers (app/widget level)
    BlockSettingsManager: Abstract base for block-level settings managers
    AppSettingsManager: Global application settings

Functions:
    init_app_settings_manager: Create an AppSettingsManager instance (factory function)

Usage:
    # App-level settings
    from src.application.settings import AppSettingsManager, BaseSettings, BaseSettingsManager
    # AppSettingsManager should be created in bootstrap and stored in ServiceContainer
    
    # Block-level settings
    from src.application.settings import BlockSettingsManager
    
See: AgentAssets/SETTINGS_STANDARD.md for app-level settings
See: AgentAssets/SETTINGS_ABSTRACTION_PRESET.md for block-level settings
"""

from .base_settings import BaseSettings, BaseSettingsManager
from .block_settings import BlockSettingsManager
from .app_settings import (
    AppSettings,
    AppSettingsManager,
    init_app_settings_manager,
)

__all__ = [
    'BaseSettings',
    'BaseSettingsManager',
    'BlockSettingsManager',
    'AppSettings',
    'AppSettingsManager',
    'init_app_settings_manager',
]
