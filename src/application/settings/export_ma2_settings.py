"""
ExportMA2 Block Settings

Settings schema and manager for ExportMA2 blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class ExportMA2BlockSettings(BaseSettings):
    """
    Settings schema for ExportMA2 blocks.

    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    output_path: Optional[str] = None


class ExportMA2SettingsManager(BlockSettingsManager):
    """
    Settings manager for ExportMA2 blocks.

    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = ExportMA2BlockSettings

    @property
    def output_path(self) -> Optional[str]:
        """Get output file path."""
        return self._settings.output_path

    @output_path.setter
    def output_path(self, value: Optional[str]):
        """Set output file path."""
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Output path must be a string or None, got {type(value).__name__}")

        if value == "":
            value = None

        if value != self._settings.output_path:
            self._settings.output_path = value
            self._save_setting('output_path')
