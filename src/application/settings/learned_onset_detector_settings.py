"""
LearnedOnsetDetector Block Settings

Settings schema and manager for LearnedOnsetDetector blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings, validated_field
from src.shared.application.settings import register_block_settings


@register_block_settings(
    "LearnedOnsetDetector",
    description="CNN-based onset detection settings",
    version=1,
    tags=["audio", "onset", "pytorch", "cnn"],
)
@dataclass
class LearnedOnsetDetectorBlockSettings(BaseSettings):
    """Settings schema for LearnedOnsetDetector blocks."""

    model_path: Optional[str] = None
    device: str = validated_field("cpu", choices=["cpu", "cuda", "mps"])
    threshold: float = validated_field(0.3, min_value=0.0, max_value=1.0)
    min_silence: float = validated_field(0.03, min_value=0.0)
    hop_length: int = validated_field(256, min_value=1)
    n_mels: int = validated_field(128, min_value=16, max_value=512)
    use_backtrack: bool = True
    fallback_to_spectral_flux: bool = True


class LearnedOnsetDetectorSettingsManager(BlockSettingsManager):
    """Type-safe settings manager for LearnedOnsetDetector blocks."""

    SETTINGS_CLASS = LearnedOnsetDetectorBlockSettings

    @property
    def model_path(self) -> Optional[str]:
        return self._settings.model_path

    @model_path.setter
    def model_path(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Model path must be a string or None, got {type(value).__name__}")
        if value == "":
            value = None
        if value != self._settings.model_path:
            self._settings.model_path = value
            self._save_setting("model_path")

    @property
    def device(self) -> str:
        return self._settings.device

    @device.setter
    def device(self, value: str):
        valid_devices = {"cpu", "cuda", "mps"}
        if value not in valid_devices:
            raise ValueError(f"Device must be one of {sorted(valid_devices)}, got '{value}'")
        if value != self._settings.device:
            self._settings.device = value
            self._save_setting("device")

    @property
    def threshold(self) -> float:
        return self._settings.threshold

    @threshold.setter
    def threshold(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Threshold must be a number, got {type(value).__name__}")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {value}")
        if abs(value - self._settings.threshold) > 1e-6:
            self._settings.threshold = value
            self._save_setting("threshold")

    @property
    def min_silence(self) -> float:
        return self._settings.min_silence

    @min_silence.setter
    def min_silence(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError(f"min_silence must be a number, got {type(value).__name__}")
        value = float(value)
        if value < 0.0:
            raise ValueError(f"min_silence must be >= 0.0, got {value}")
        if abs(value - self._settings.min_silence) > 1e-6:
            self._settings.min_silence = value
            self._save_setting("min_silence")

    @property
    def hop_length(self) -> int:
        return self._settings.hop_length

    @hop_length.setter
    def hop_length(self, value: int):
        if not isinstance(value, int):
            raise ValueError(f"hop_length must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"hop_length must be > 0, got {value}")
        if value != self._settings.hop_length:
            self._settings.hop_length = value
            self._save_setting("hop_length")

    @property
    def n_mels(self) -> int:
        return self._settings.n_mels

    @n_mels.setter
    def n_mels(self, value: int):
        if not isinstance(value, int):
            raise ValueError(f"n_mels must be an integer, got {type(value).__name__}")
        if value < 16 or value > 512:
            raise ValueError(f"n_mels must be between 16 and 512, got {value}")
        if value != self._settings.n_mels:
            self._settings.n_mels = value
            self._save_setting("n_mels")

    @property
    def use_backtrack(self) -> bool:
        return self._settings.use_backtrack

    @use_backtrack.setter
    def use_backtrack(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(f"use_backtrack must be a bool, got {type(value).__name__}")
        if value != self._settings.use_backtrack:
            self._settings.use_backtrack = value
            self._save_setting("use_backtrack")

    @property
    def fallback_to_spectral_flux(self) -> bool:
        return self._settings.fallback_to_spectral_flux

    @fallback_to_spectral_flux.setter
    def fallback_to_spectral_flux(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(f"fallback_to_spectral_flux must be a bool, got {type(value).__name__}")
        if value != self._settings.fallback_to_spectral_flux:
            self._settings.fallback_to_spectral_flux = value
            self._save_setting("fallback_to_spectral_flux")
