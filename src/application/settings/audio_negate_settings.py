"""
Audio Negate Block Settings

Settings schema and manager for AudioNegate blocks.
"""
from dataclasses import dataclass

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.application.blocks.audio_negate_block import NEGATE_MODES
from src.utils.message import Log


@dataclass
class AudioNegateBlockSettings(BaseSettings):
    """
    Settings schema for AudioNegate blocks.

    All fields have default values for backwards compatibility.
    """
    # Negation mode
    mode: str = "silence"

    # Crossfade at region edges (milliseconds)
    fade_ms: float = 10.0

    # Volume reduction for attenuate mode (dB, should be negative)
    attenuation_db: float = -20.0

    # Multiplier for spectral subtraction strength (subtract mode)
    subtract_gain: float = 1.0

    # Extra emphasis on onset/transient portion of events (subtract mode)
    onset_emphasis: float = 1.0


class AudioNegateSettingsManager(BlockSettingsManager):
    """
    Settings manager for AudioNegate blocks.

    Provides type-safe property accessors with validation.
    """
    SETTINGS_CLASS = AudioNegateBlockSettings

    def __init__(self, facade, block_id, parent=None):
        super().__init__(facade, block_id, parent)

    # =========================================================================
    # Mode
    # =========================================================================

    @property
    def mode(self) -> str:
        """Get the current negation mode."""
        return self._settings.mode

    @mode.setter
    def mode(self, value: str):
        """Set the negation mode with validation."""
        if value not in NEGATE_MODES:
            raise ValueError(
                f"Invalid negation mode: '{value}'. "
                f"Valid options: {', '.join(sorted(NEGATE_MODES.keys()))}"
            )
        if value != self._settings.mode:
            self._settings.mode = value
            self._save_setting("mode")

    # =========================================================================
    # Fade Duration
    # =========================================================================

    @property
    def fade_ms(self) -> float:
        """Get the crossfade duration at region edges in milliseconds."""
        return self._settings.fade_ms

    @fade_ms.setter
    def fade_ms(self, value: float):
        """Set the crossfade duration with validation."""
        value = float(value)
        if value < 0.0 or value > 100.0:
            raise ValueError(
                f"Fade duration must be between 0 and 100 ms, got {value}"
            )
        if value != self._settings.fade_ms:
            self._settings.fade_ms = value
            self._save_setting("fade_ms")

    # =========================================================================
    # Attenuation (dB)
    # =========================================================================

    @property
    def attenuation_db(self) -> float:
        """Get the volume reduction in dB for attenuate mode."""
        return self._settings.attenuation_db

    @attenuation_db.setter
    def attenuation_db(self, value: float):
        """Set the attenuation in dB with validation."""
        value = float(value)
        if value < -60.0 or value > 0.0:
            raise ValueError(
                f"Attenuation must be between -60 and 0 dB, got {value}"
            )
        if value != self._settings.attenuation_db:
            self._settings.attenuation_db = value
            self._save_setting("attenuation_db")

    # =========================================================================
    # Subtract Gain
    # =========================================================================

    @property
    def subtract_gain(self) -> float:
        """Get the subtraction strength multiplier for subtract mode."""
        return self._settings.subtract_gain

    @subtract_gain.setter
    def subtract_gain(self, value: float):
        """Set the subtraction strength multiplier with validation."""
        value = float(value)
        if value < 1.0 or value > 10.0:
            raise ValueError(
                f"Subtract gain must be between 1.0 and 10.0, got {value}"
            )
        if value != self._settings.subtract_gain:
            self._settings.subtract_gain = value
            self._save_setting("subtract_gain")

    # =========================================================================
    # Onset Emphasis
    # =========================================================================

    @property
    def onset_emphasis(self) -> float:
        """Get the onset/transient emphasis multiplier for subtract mode."""
        return self._settings.onset_emphasis

    @onset_emphasis.setter
    def onset_emphasis(self, value: float):
        """Set the onset emphasis multiplier with validation."""
        value = float(value)
        if value < 1.0 or value > 5.0:
            raise ValueError(
                f"Onset emphasis must be between 1.0 and 5.0, got {value}"
            )
        if value != self._settings.onset_emphasis:
            self._settings.onset_emphasis = value
            self._save_setting("onset_emphasis")
