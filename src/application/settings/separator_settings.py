"""
Separator Block Settings

Settings schema and manager for Separator blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.application.blocks.separator_block import DEMUCS_MODELS
from src.utils.message import Log


@dataclass
class SeparatorBlockSettings(BaseSettings):
    """
    Settings schema for Separator blocks.
    
    All fields have default values for backwards compatibility.
    Note: Settings are stored in block.metadata at the top level, not nested.
    """
    # Model settings
    model: str = "htdemucs"  # Stored in separator_settings.model in metadata
    
    # Processing settings
    device: str = "auto"
    two_stems: Optional[str] = None  # None = all stems, "vocals"/"drums"/"bass"/"other" = 2-stem mode
    shifts: int = 1  # Number of random shifts for quality (0=fastest, 1=default, 10=paper quality)
    
    # Output settings
    output_format: str = "wav"  # "wav" or "mp3"
    mp3_bitrate: str = "320"  # "320", "192", "128"
    
    # Legacy support: map nested separator_settings.model to top-level model
    @classmethod
    def from_dict(cls, data: dict):
        """
        Load settings from block metadata with backwards compatibility.
        
        Handles both nested (separator_settings.model) and flat (model) storage.
        """
        # Check for nested separator_settings
        if "separator_settings" in data and isinstance(data["separator_settings"], dict):
            nested = data["separator_settings"]
            # Merge nested settings into top level for schema
            merged = dict(data)
            if "model" in nested:
                merged["model"] = nested["model"]
        else:
            merged = data
        
        return super().from_dict(merged)
    
    def to_dict(self) -> dict:
        """
        Convert to metadata format.
        
        Stores model in separator_settings.model for backwards compatibility
        with existing block_service.execute_block_command("set_model") code.
        Other settings stored flat in metadata.
        """
        data = super().to_dict()
        
        # Store model in nested structure for backwards compatibility
        model = data.pop("model", None)
        if model:
            if "separator_settings" not in data:
                data["separator_settings"] = {}
            data["separator_settings"]["model"] = model
        
        # Remove None values (two_stems = None means "all stems", so don't store it)
        if data.get("two_stems") is None:
            data.pop("two_stems", None)
        
        return data


class SeparatorSettingsManager(BlockSettingsManager):
    """
    Settings manager for Separator blocks.
    
    Provides type-safe property accessors with validation.
    """
    SETTINGS_CLASS = SeparatorBlockSettings
    
    def __init__(self, facade, block_id, parent=None):
        super().__init__(facade, block_id, parent)
    
    # =========================================================================
    # Model Properties
    # =========================================================================
    
    @property
    def model(self) -> str:
        return self._settings.model
    
    @model.setter
    def model(self, value: str):
        if value not in DEMUCS_MODELS:
            raise ValueError(
                f"Invalid model: '{value}'. "
                f"Valid options: {', '.join(sorted(DEMUCS_MODELS.keys()))}"
            )
        
        if value != self._settings.model:
            self._settings.model = value
            self._save_setting('model')
    
    # =========================================================================
    # Processing Properties
    # =========================================================================
    
    @property
    def device(self) -> str:
        return self._settings.device
    
    @device.setter
    def device(self, value: str):
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if value not in valid_devices:
            raise ValueError(
                f"Invalid device: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_devices))}"
            )
        
        if value != self._settings.device:
            self._settings.device = value
            self._save_setting('device')
    
    @property
    def two_stems(self) -> Optional[str]:
        return self._settings.two_stems
    
    @two_stems.setter
    def two_stems(self, value: Optional[str]):
        if value is not None:
            valid_stems = {"vocals", "drums", "bass", "other"}
            if value not in valid_stems:
                raise ValueError(
                    f"Invalid two_stems value: '{value}'. "
                    f"Valid options: {', '.join(sorted(valid_stems))} or None"
                )
        
        if value != self._settings.two_stems:
            self._settings.two_stems = value
            # Store None as empty string removal for backwards compatibility
            if value is None and hasattr(self._settings, 'to_dict'):
                # Will be handled in to_dict() conversion
                pass
            self._save_setting('two_stems')
            
            # Update expected_outputs when configuration changes
            # This ensures filter UI and validation show correct expected outputs
            self._update_expected_outputs()
    
    @property
    def shifts(self) -> int:
        """Get number of random shifts for quality."""
        return self._settings.shifts
    
    @shifts.setter
    def shifts(self, value: int):
        """Set number of random shifts with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Shifts must be a number, got {type(value).__name__}")
        
        # Convert to int (allows float input for convenience)
        int_value = int(value)
        
        if int_value < 0:
            raise ValueError(f"Shifts must be >= 0, got {int_value}")
        
        if int_value != self._settings.shifts:
            self._settings.shifts = int_value
            self._save_setting('shifts')
    
    # =========================================================================
    # Output Properties
    # =========================================================================
    
    @property
    def output_format(self) -> str:
        return self._settings.output_format
    
    @output_format.setter
    def output_format(self, value: str):
        valid_formats = {"wav", "mp3"}
        if value not in valid_formats:
            raise ValueError(
                f"Invalid output format: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_formats))}"
            )
        
        if value != self._settings.output_format:
            self._settings.output_format = value
            self._save_setting('output_format')
    
    @property
    def mp3_bitrate(self) -> str:
        return self._settings.mp3_bitrate
    
    @mp3_bitrate.setter
    def mp3_bitrate(self, value: str):
        valid_bitrates = {"320", "192", "128"}
        if value not in valid_bitrates:
            raise ValueError(
                f"Invalid MP3 bitrate: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_bitrates))}"
            )
        
        if value != self._settings.mp3_bitrate:
            self._settings.mp3_bitrate = value
            self._save_setting('mp3_bitrate')
