"""
ExportClipsByClass Block Settings

Settings schema and manager for ExportClipsByClass blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class ExportClipsByClassSettings(BaseSettings):
    """
    Settings schema for ExportClipsByClass blocks.
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Output directory (base directory, subfolders created per class)
    output_dir: Optional[str] = None
    
    # Export format
    audio_format: str = "wav"  # "wav", "mp3", "flac", "ogg"
    
    # Whether to include events without classification
    include_unclassified: bool = True
    
    # Folder name for unclassified events
    unclassified_folder: str = "unclassified"


class ExportClipsByClassSettingsManager(BlockSettingsManager):
    """
    Settings manager for ExportClipsByClass blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = ExportClipsByClassSettings
    
    @property
    def output_dir(self) -> Optional[str]:
        """Get output directory path."""
        return self._settings.output_dir
    
    @output_dir.setter
    def output_dir(self, value: Optional[str]):
        """Set output directory path."""
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Output directory must be a string or None, got {type(value).__name__}")
        
        # Normalize empty string to None
        if value == "":
            value = None
        
        if value != self._settings.output_dir:
            self._settings.output_dir = value
            self._save_setting('output_dir')
    
    @property
    def audio_format(self) -> str:
        """Get audio export format."""
        return self._settings.audio_format
    
    @audio_format.setter
    def audio_format(self, value: str):
        """Set audio export format with validation."""
        valid_formats = {"wav", "mp3", "flac", "ogg"}
        if value not in valid_formats:
            raise ValueError(
                f"Invalid audio format: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_formats))}"
            )
        
        if value != self._settings.audio_format:
            self._settings.audio_format = value
            self._save_setting('audio_format')
    
    @property
    def include_unclassified(self) -> bool:
        """Get whether to include unclassified events."""
        return self._settings.include_unclassified
    
    @include_unclassified.setter
    def include_unclassified(self, value: bool):
        """Set whether to include unclassified events."""
        if not isinstance(value, bool):
            raise ValueError(f"include_unclassified must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.include_unclassified:
            self._settings.include_unclassified = value
            self._save_setting('include_unclassified')
    
    @property
    def unclassified_folder(self) -> str:
        """Get folder name for unclassified events."""
        return self._settings.unclassified_folder
    
    @unclassified_folder.setter
    def unclassified_folder(self, value: str):
        """Set folder name for unclassified events."""
        if not isinstance(value, str):
            raise ValueError(f"unclassified_folder must be a string, got {type(value).__name__}")
        
        # Sanitize folder name
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            value = value.replace(char, '_')
        value = value.strip()[:100] or "unclassified"
        
        if value != self._settings.unclassified_folder:
            self._settings.unclassified_folder = value
            self._save_setting('unclassified_folder')


