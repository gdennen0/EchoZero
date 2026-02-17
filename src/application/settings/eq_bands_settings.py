"""
EQ Bands Block Settings

Settings schema and manager for EQBands blocks.
Stores a list of frequency bands, each with low/high frequency and gain.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.application.blocks.eq_bands_block import DEFAULT_BANDS
from src.utils.message import Log


@dataclass
class EQBandsBlockSettings(BaseSettings):
    """
    Settings schema for EQBands blocks.

    All fields have default values for backwards compatibility.
    """
    # List of EQ bands, each is {"freq_low": float, "freq_high": float, "gain_db": float}
    bands: List[Dict[str, float]] = field(default_factory=lambda: list(DEFAULT_BANDS))

    # Butterworth filter order (1-8) for bandpass filters
    order: int = 4


class EQBandsSettingsManager(BlockSettingsManager):
    """
    Settings manager for EQBands blocks.

    Provides type-safe property accessors with validation.
    Bands are stored as a list of dicts in block.metadata.
    """
    SETTINGS_CLASS = EQBandsBlockSettings

    def __init__(self, facade, block_id, parent=None):
        super().__init__(facade, block_id, parent)

    # =========================================================================
    # Bands
    # =========================================================================

    @property
    def bands(self) -> List[Dict[str, float]]:
        """Get the current list of EQ bands."""
        return self._settings.bands

    @bands.setter
    def bands(self, value: List[Dict[str, float]]):
        """Set the entire bands list with validation."""
        if not isinstance(value, list):
            raise ValueError("Bands must be a list")
        for i, band in enumerate(value):
            if not isinstance(band, dict):
                raise ValueError(f"Band {i + 1} must be a dictionary")
            self._validate_band(band, i + 1)
        self._settings.bands = value
        self._save_setting("bands")

    def add_band(self, freq_low: float = 1000.0, freq_high: float = 4000.0, gain_db: float = 0.0):
        """
        Add a new EQ band.

        Args:
            freq_low: Low frequency in Hz (20-20000)
            freq_high: High frequency in Hz (20-20000)
            gain_db: Gain in dB (-24 to +24)
        """
        band = {"freq_low": freq_low, "freq_high": freq_high, "gain_db": gain_db}
        self._validate_band(band, len(self._settings.bands) + 1)
        self._settings.bands.append(band)
        self._save_setting("bands")

    def remove_band(self, index: int):
        """
        Remove a band by index.

        Args:
            index: Zero-based index of the band to remove
        """
        if index < 0 or index >= len(self._settings.bands):
            raise ValueError(
                f"Band index {index} out of range (0-{len(self._settings.bands) - 1})"
            )
        self._settings.bands.pop(index)
        self._save_setting("bands")

    def update_band(self, index: int, freq_low: float = None, freq_high: float = None, gain_db: float = None):
        """
        Update a single band's properties.

        Args:
            index: Zero-based index of the band to update
            freq_low: New low frequency (or None to keep current)
            freq_high: New high frequency (or None to keep current)
            gain_db: New gain (or None to keep current)
        """
        if index < 0 or index >= len(self._settings.bands):
            raise ValueError(
                f"Band index {index} out of range (0-{len(self._settings.bands) - 1})"
            )

        band = self._settings.bands[index].copy()
        if freq_low is not None:
            band["freq_low"] = float(freq_low)
        if freq_high is not None:
            band["freq_high"] = float(freq_high)
        if gain_db is not None:
            band["gain_db"] = float(gain_db)

        self._validate_band(band, index + 1)
        self._settings.bands[index] = band
        self._save_setting("bands")

    def _validate_band(self, band: dict, band_number: int):
        """Validate a single band dict."""
        freq_low = float(band.get("freq_low", 0))
        freq_high = float(band.get("freq_high", 0))
        gain_db = float(band.get("gain_db", 0))

        if freq_low < 20.0 or freq_low > 20000.0:
            raise ValueError(
                f"Band {band_number}: Low frequency must be between 20 and 20000 Hz, got {freq_low}"
            )
        if freq_high < 20.0 or freq_high > 20000.0:
            raise ValueError(
                f"Band {band_number}: High frequency must be between 20 and 20000 Hz, got {freq_high}"
            )
        if freq_low >= freq_high:
            raise ValueError(
                f"Band {band_number}: Low frequency ({freq_low}) must be less than high frequency ({freq_high})"
            )
        if gain_db < -24.0 or gain_db > 24.0:
            raise ValueError(
                f"Band {band_number}: Gain must be between -24 and +24 dB, got {gain_db}"
            )

    # =========================================================================
    # Order
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
