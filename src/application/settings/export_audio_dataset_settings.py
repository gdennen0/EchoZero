"""
Export Audio Dataset Block Settings

Settings schema and manager for ExportAudioDataset blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.application.blocks.export_audio_dataset_block import (
    SUPPORTED_FORMATS, NAMING_SCHEMES,
)
from src.utils.message import Log


@dataclass
class ExportAudioDatasetSettings(BaseSettings):
    """
    Settings schema for ExportAudioDataset blocks.

    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Output directory for exported clips
    output_dir: Optional[str] = None

    # Export audio format
    audio_format: str = "wav"

    # Naming scheme for clip files
    naming_scheme: str = "index"

    # Zero-padding digits for index-based naming
    zero_pad_digits: int = 4

    # User-defined filename prefix (used with "prefix" naming scheme)
    filename_prefix: str = "clip"

    # Whether to group clips into subdirectories by classification
    group_by_class: bool = False

    # Folder name for events without classification (when group_by_class is True)
    unclassified_folder: str = "unclassified"


class ExportAudioDatasetSettingsManager(BlockSettingsManager):
    """
    Settings manager for ExportAudioDataset blocks.

    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = ExportAudioDatasetSettings

    # =========================================================================
    # Output Directory
    # =========================================================================

    @property
    def output_dir(self) -> Optional[str]:
        """Get output directory path."""
        return self._settings.output_dir

    @output_dir.setter
    def output_dir(self, value: Optional[str]):
        """Set output directory path."""
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"Output directory must be a string or None, got {type(value).__name__}"
            )
        if value == "":
            value = None
        if value != self._settings.output_dir:
            self._settings.output_dir = value
            self._save_setting("output_dir")

    # =========================================================================
    # Audio Format
    # =========================================================================

    @property
    def audio_format(self) -> str:
        """Get audio export format."""
        return self._settings.audio_format

    @audio_format.setter
    def audio_format(self, value: str):
        """Set audio export format with validation."""
        if value not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Invalid audio format: '{value}'. "
                f"Valid options: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        if value != self._settings.audio_format:
            self._settings.audio_format = value
            self._save_setting("audio_format")

    # =========================================================================
    # Naming Scheme
    # =========================================================================

    @property
    def naming_scheme(self) -> str:
        """Get the clip naming scheme."""
        return self._settings.naming_scheme

    @naming_scheme.setter
    def naming_scheme(self, value: str):
        """Set the clip naming scheme with validation."""
        if value not in NAMING_SCHEMES:
            raise ValueError(
                f"Invalid naming scheme: '{value}'. "
                f"Valid options: {', '.join(sorted(NAMING_SCHEMES.keys()))}"
            )
        if value != self._settings.naming_scheme:
            self._settings.naming_scheme = value
            self._save_setting("naming_scheme")

    # =========================================================================
    # Zero Pad Digits
    # =========================================================================

    @property
    def zero_pad_digits(self) -> int:
        """Get zero-padding digit count for index naming."""
        return self._settings.zero_pad_digits

    @zero_pad_digits.setter
    def zero_pad_digits(self, value: int):
        """Set zero-padding digit count with validation."""
        value = int(value)
        if value < 1 or value > 8:
            raise ValueError(
                f"Zero-pad digits must be between 1 and 8, got {value}"
            )
        if value != self._settings.zero_pad_digits:
            self._settings.zero_pad_digits = value
            self._save_setting("zero_pad_digits")

    # =========================================================================
    # Filename Prefix
    # =========================================================================

    @property
    def filename_prefix(self) -> str:
        """Get the user-defined filename prefix."""
        return self._settings.filename_prefix

    @filename_prefix.setter
    def filename_prefix(self, value: str):
        """Set the filename prefix with validation."""
        if not isinstance(value, str):
            raise ValueError(
                f"Filename prefix must be a string, got {type(value).__name__}"
            )
        # Sanitize: strip whitespace, limit length, default to "clip"
        value = value.strip()[:80] or "clip"
        if value != self._settings.filename_prefix:
            self._settings.filename_prefix = value
            self._save_setting("filename_prefix")

    # =========================================================================
    # Group by Classification
    # =========================================================================

    @property
    def group_by_class(self) -> bool:
        """Get whether clips are grouped into subdirectories by classification."""
        return self._settings.group_by_class

    @group_by_class.setter
    def group_by_class(self, value: bool):
        """Set whether to group clips by classification."""
        value = bool(value)
        if value != self._settings.group_by_class:
            self._settings.group_by_class = value
            self._save_setting("group_by_class")

    # =========================================================================
    # Unclassified Folder Name
    # =========================================================================

    @property
    def unclassified_folder(self) -> str:
        """Get folder name for events without classification."""
        return self._settings.unclassified_folder

    @unclassified_folder.setter
    def unclassified_folder(self, value: str):
        """Set folder name for unclassified events."""
        if not isinstance(value, str):
            raise ValueError(
                f"Unclassified folder must be a string, got {type(value).__name__}"
            )
        # Sanitize
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            value = value.replace(char, "_")
        value = value.strip()[:100] or "unclassified"

        if value != self._settings.unclassified_folder:
            self._settings.unclassified_folder = value
            self._save_setting("unclassified_folder")
