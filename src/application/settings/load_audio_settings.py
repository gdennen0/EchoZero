"""
LoadAudio Block Settings

Settings schema and manager for LoadAudio blocks.

Demonstrates the new registry decorator for auto-discovery.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.shared.application.settings import register_block_settings
from src.utils.message import Log


@register_block_settings(
    "LoadAudio",
    description="Audio file loading settings",
    tags=["audio", "input"]
)
@dataclass
class LoadAudioBlockSettings(BaseSettings):
    """
    Settings schema for LoadAudio blocks.
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Audio file path
    audio_path: Optional[str] = None
    


class LoadAudioSettingsManager(BlockSettingsManager):
    """
    Settings manager for LoadAudio blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = LoadAudioBlockSettings
    
    @property
    def audio_path(self) -> Optional[str]:
        """Get audio file path."""
        return self._settings.audio_path
    
    @audio_path.setter
    def audio_path(self, value: Optional[str]):
        """Set audio file path."""
        
        # Allow None (no file selected)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Audio path must be a string or None, got {type(value).__name__}")
        
        # Normalize empty string to None
        if value == "":
            value = None
        
        if value != self._settings.audio_path:
            self._settings.audio_path = value
            self._save_setting('audio_path')
