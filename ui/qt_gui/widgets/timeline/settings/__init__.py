"""
Timeline Settings Components

Settings UI, storage, and keyboard shortcuts.
"""

from .panel import SettingsPanel, PlayheadFollowMode, TimelineSettings
from .storage import TimelineSettingsManager, TimelineSettings as TimelineUISettings
from .storage import get_timeline_settings_manager, set_timeline_settings_manager
from .shortcuts import ShortcutsSettingsDialog, ShortcutKeySequenceEdit

__all__ = [
    'SettingsPanel',
    'PlayheadFollowMode',
    'TimelineSettings',
    'TimelineSettingsManager',
    'TimelineUISettings',
    'get_timeline_settings_manager',
    'set_timeline_settings_manager',
    'ShortcutsSettingsDialog',
    'ShortcutKeySequenceEdit',
]




