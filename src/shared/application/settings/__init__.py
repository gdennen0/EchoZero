"""
Shared settings module.

Provides:
- SettingsRegistry: Auto-discovery and registration of settings classes
- Base settings classes are in src/application/settings/base_settings.py
  (will be moved here during Phase 2 reorganization)
"""
from .settings_registry import (
    SettingsRegistry,
    register_settings,
    register_block_settings,
)

__all__ = [
    'SettingsRegistry',
    'register_settings',
    'register_block_settings',
]
