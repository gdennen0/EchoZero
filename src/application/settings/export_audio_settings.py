"""
ExportAudio Block Settings

Settings schema and manager for ExportAudio blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class ExportAudioBlockSettings(BaseSettings):
    """
    Settings schema for ExportAudio blocks.
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Output directory
    output_dir: Optional[str] = None  # Path to output directory (also supports legacy "export_dir")
    
    # Export format
    audio_format: str = "wav"  # "wav", "mp3", "flac", "ogg"
    
    # Filename prefix
    filename_prefix: str = ""  # Prefix for exported filenames
    
    # Legacy support: map "export_dir" to "output_dir"
    @classmethod
    def from_dict(cls, data: dict):
        """
        Load settings from block metadata with backwards compatibility.
        
        Handles legacy "export_dir" key mapping to "output_dir".
        """
        merged = dict(data)
        
        # Map legacy "export_dir" to "output_dir" if present
        if "export_dir" in merged and "output_dir" not in merged:
            merged["output_dir"] = merged["export_dir"]
        
        return super().from_dict(merged)


class ExportAudioSettingsManager(BlockSettingsManager):
    """
    Settings manager for ExportAudio blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = ExportAudioBlockSettings
    
    @property
    def output_dir(self) -> Optional[str]:
        """Get output directory path."""
        return self._settings.output_dir
    
    @output_dir.setter
    def output_dir(self, value: Optional[str]):
        """Set output directory path."""
        # Allow None (no directory selected)
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
    def filename_prefix(self) -> str:
        """Get filename prefix."""
        return self._settings.filename_prefix
    
    @filename_prefix.setter
    def filename_prefix(self, value: str):
        """Set filename prefix."""
        if not isinstance(value, str):
            raise ValueError(f"Filename prefix must be a string, got {type(value).__name__}")
        
        if value != self._settings.filename_prefix:
            self._settings.filename_prefix = value
            self._save_setting('filename_prefix')
