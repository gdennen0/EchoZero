"""
Dataset Viewer Block Settings

Settings schema and manager for DatasetViewer blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings


@dataclass
class DatasetViewerSettings(BaseSettings):
    """
    Settings schema for DatasetViewer blocks.

    Stored in block.metadata. source_dir is the directory of audio clips to audit;
    removed samples are moved into source_dir/removed_subdir.
    """
    source_dir: Optional[str] = None
    removed_subdir: str = "removed"


class DatasetViewerSettingsManager(BlockSettingsManager):
    """
    Settings manager for DatasetViewer blocks.
    """

    SETTINGS_CLASS = DatasetViewerSettings

    @property
    def source_dir(self) -> Optional[str]:
        """Get source directory path."""
        return self._settings.source_dir

    @source_dir.setter
    def source_dir(self, value: Optional[str]):
        """Set source directory path."""
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"source_dir must be a string or None, got {type(value).__name__}"
            )
        if value == "":
            value = None
        if value != self._settings.source_dir:
            self._settings.source_dir = value
            self._save_setting("source_dir")

    @property
    def removed_subdir(self) -> str:
        """Get name of subdirectory for removed samples (under source_dir)."""
        return self._settings.removed_subdir

    @removed_subdir.setter
    def removed_subdir(self, value: str):
        """Set subdirectory name for removed samples."""
        if not isinstance(value, str):
            raise ValueError(
                f"removed_subdir must be a string, got {type(value).__name__}"
            )
        value = value.strip() or "removed"
        if value != self._settings.removed_subdir:
            self._settings.removed_subdir = value
            self._save_setting("removed_subdir")
