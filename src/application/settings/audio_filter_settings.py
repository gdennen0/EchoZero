"""
Audio Filter Block Settings

Settings schema and manager for AudioFilter blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.application.blocks.audio_filter_block import FILTER_TYPES
from src.utils.message import Log


@dataclass
class AudioFilterBlockSettings(BaseSettings):
    """
    Settings schema for AudioFilter blocks.

    All fields have default values for backwards compatibility.
    """
    # Filter type
    filter_type: str = "lowpass"

    # Frequency settings
    cutoff_freq: float = 1000.0       # Primary cutoff/center frequency (Hz)
    cutoff_freq_high: float = 8000.0  # Upper cutoff for bandpass/bandstop (Hz)

    # Butterworth filter order (1-8, for lowpass/highpass/bandpass/bandstop)
    order: int = 4

    # Gain for shelf/peak filters (dB)
    gain_db: float = 0.0

    # Q factor for shelf/peak filters
    q_factor: float = 0.707


class AudioFilterSettingsManager(BlockSettingsManager):
    """
    Settings manager for AudioFilter blocks.

    Provides type-safe property accessors with validation.
    """
    SETTINGS_CLASS = AudioFilterBlockSettings

    def __init__(self, facade, block_id, parent=None):
        super().__init__(facade, block_id, parent)

    # =========================================================================
    # Filter Type
    # =========================================================================

    @property
    def filter_type(self) -> str:
        """Get the current filter type."""
        return self._settings.filter_type

    @filter_type.setter
    def filter_type(self, value: str):
        """Set the filter type with validation."""
        if value not in FILTER_TYPES:
            raise ValueError(
                f"Invalid filter type: '{value}'. "
                f"Valid options: {', '.join(sorted(FILTER_TYPES.keys()))}"
            )
        if value != self._settings.filter_type:
            self._settings.filter_type = value
            self._save_setting("filter_type")

    # =========================================================================
    # Frequency Settings
    # =========================================================================

    @property
    def cutoff_freq(self) -> float:
        """Get the primary cutoff/center frequency in Hz."""
        return self._settings.cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, value: float):
        """Set the primary cutoff frequency with validation."""
        value = float(value)
        if value < 20.0 or value > 20000.0:
            raise ValueError(
                f"Cutoff frequency must be between 20 and 20000 Hz, got {value}"
            )
        if value != self._settings.cutoff_freq:
            self._settings.cutoff_freq = value
            self._save_setting("cutoff_freq")

    @property
    def cutoff_freq_high(self) -> float:
        """Get the upper cutoff frequency in Hz (for bandpass/bandstop)."""
        return self._settings.cutoff_freq_high

    @cutoff_freq_high.setter
    def cutoff_freq_high(self, value: float):
        """Set the upper cutoff frequency with validation."""
        value = float(value)
        if value < 20.0 or value > 20000.0:
            raise ValueError(
                f"Upper cutoff frequency must be between 20 and 20000 Hz, got {value}"
            )
        if value != self._settings.cutoff_freq_high:
            self._settings.cutoff_freq_high = value
            self._save_setting("cutoff_freq_high")

    # =========================================================================
    # Filter Order
    # =========================================================================

    @property
    def order(self) -> int:
        """Get the Butterworth filter order."""
        return self._settings.order

    @order.setter
    def order(self, value: int):
        """Set the filter order with validation."""
        value = int(value)
        if value < 1 or value > 8:
            raise ValueError(
                f"Filter order must be between 1 and 8, got {value}"
            )
        if value != self._settings.order:
            self._settings.order = value
            self._save_setting("order")

    # =========================================================================
    # Gain (Shelf / Peak)
    # =========================================================================

    @property
    def gain_db(self) -> float:
        """Get gain in dB for shelf/peak filters."""
        return self._settings.gain_db

    @gain_db.setter
    def gain_db(self, value: float):
        """Set gain in dB with validation."""
        value = float(value)
        if value < -24.0 or value > 24.0:
            raise ValueError(
                f"Gain must be between -24 and +24 dB, got {value}"
            )
        if value != self._settings.gain_db:
            self._settings.gain_db = value
            self._save_setting("gain_db")

    # =========================================================================
    # Q Factor (Shelf / Peak)
    # =========================================================================

    @property
    def q_factor(self) -> float:
        """Get Q factor for shelf/peak filters."""
        return self._settings.q_factor

    @q_factor.setter
    def q_factor(self, value: float):
        """Set Q factor with validation."""
        value = float(value)
        if value < 0.1 or value > 10.0:
            raise ValueError(
                f"Q factor must be between 0.1 and 10.0, got {value}"
            )
        if value != self._settings.q_factor:
            self._settings.q_factor = value
            self._save_setting("q_factor")
