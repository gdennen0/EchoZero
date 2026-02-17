"""
LearnedOnsetTrainer Block Settings

Settings schema and manager for LearnedOnsetTrainer blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .base_settings import BaseSettings, validated_field
from .block_settings import BlockSettingsManager
from src.shared.application.settings import register_block_settings


@register_block_settings(
    "LearnedOnsetTrainer",
    description="Train a CNN-based onset detector from annotated dataset",
    version=1,
    tags=["audio", "onset", "training", "pytorch"],
)
@dataclass
class LearnedOnsetTrainerBlockSettings(BaseSettings):
    """Settings for PoC onset training."""

    dataset_root: str = validated_field("", required=True, min_length=1)
    output_model_path: Optional[str] = None
    sample_rate: int = validated_field(22050, min_value=8000, max_value=96000)
    n_mels: int = validated_field(128, min_value=16, max_value=512)
    mel_hop_length: int = validated_field(256, min_value=32, max_value=4096)
    window_seconds: float = validated_field(1.0, min_value=0.1, max_value=10.0)
    positive_radius_ms: float = validated_field(25.0, min_value=1.0, max_value=200.0)
    negative_ratio: float = validated_field(1.0, min_value=0.1, max_value=10.0)
    max_files: int = validated_field(0, min_value=0, max_value=100000)
    epochs: int = validated_field(8, min_value=1, max_value=1000)
    batch_size: int = validated_field(16, min_value=1, max_value=512)
    learning_rate: float = validated_field(0.001, min_value=1e-7, max_value=1.0)
    validation_split: float = validated_field(0.2, min_value=0.05, max_value=0.5)
    seed: int = 42
    device: str = validated_field("cpu", choices=["cpu", "cuda", "mps"])


class LearnedOnsetTrainerSettingsManager(BlockSettingsManager):
    """Type-safe settings manager for LearnedOnsetTrainer."""

    SETTINGS_CLASS = LearnedOnsetTrainerBlockSettings

    def _save_if_changed(self, key: str, value):
        if getattr(self._settings, key) != value:
            setattr(self._settings, key, value)
            self._save_setting(key)

    @property
    def dataset_root(self) -> str:
        return self._settings.dataset_root

    @dataset_root.setter
    def dataset_root(self, value: str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("dataset_root must be a non-empty string")
        self._save_if_changed("dataset_root", value.strip())

    @property
    def output_model_path(self) -> Optional[str]:
        return self._settings.output_model_path

    @output_model_path.setter
    def output_model_path(self, value: Optional[str]):
        if value == "":
            value = None
        if value is not None and not isinstance(value, str):
            raise ValueError("output_model_path must be a string or None")
        self._save_if_changed("output_model_path", value)

    @property
    def sample_rate(self) -> int:
        return self._settings.sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("sample_rate must be a positive integer")
        self._save_if_changed("sample_rate", value)

    @property
    def n_mels(self) -> int:
        return self._settings.n_mels

    @n_mels.setter
    def n_mels(self, value: int):
        if not isinstance(value, int) or not (16 <= value <= 512):
            raise ValueError("n_mels must be an integer between 16 and 512")
        self._save_if_changed("n_mels", value)

    @property
    def mel_hop_length(self) -> int:
        return self._settings.mel_hop_length

    @mel_hop_length.setter
    def mel_hop_length(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("mel_hop_length must be a positive integer")
        self._save_if_changed("mel_hop_length", value)

    @property
    def window_seconds(self) -> float:
        return self._settings.window_seconds

    @window_seconds.setter
    def window_seconds(self, value: float):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("window_seconds must be a positive number")
        self._save_if_changed("window_seconds", float(value))

    @property
    def positive_radius_ms(self) -> float:
        return self._settings.positive_radius_ms

    @positive_radius_ms.setter
    def positive_radius_ms(self, value: float):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("positive_radius_ms must be a positive number")
        self._save_if_changed("positive_radius_ms", float(value))

    @property
    def negative_ratio(self) -> float:
        return self._settings.negative_ratio

    @negative_ratio.setter
    def negative_ratio(self, value: float):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("negative_ratio must be a positive number")
        self._save_if_changed("negative_ratio", float(value))

    @property
    def max_files(self) -> int:
        return self._settings.max_files

    @max_files.setter
    def max_files(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("max_files must be >= 0")
        self._save_if_changed("max_files", value)

    @property
    def epochs(self) -> int:
        return self._settings.epochs

    @epochs.setter
    def epochs(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("epochs must be a positive integer")
        self._save_if_changed("epochs", value)

    @property
    def batch_size(self) -> int:
        return self._settings.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._save_if_changed("batch_size", value)

    @property
    def learning_rate(self) -> float:
        return self._settings.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("learning_rate must be a positive number")
        self._save_if_changed("learning_rate", float(value))

    @property
    def validation_split(self) -> float:
        return self._settings.validation_split

    @validation_split.setter
    def validation_split(self, value: float):
        if not isinstance(value, (int, float)) or not (0.0 < value < 1.0):
            raise ValueError("validation_split must be between 0 and 1")
        self._save_if_changed("validation_split", float(value))

    @property
    def seed(self) -> int:
        return self._settings.seed

    @seed.setter
    def seed(self, value: int):
        if not isinstance(value, int):
            raise ValueError("seed must be an integer")
        self._save_if_changed("seed", value)

    @property
    def device(self) -> str:
        return self._settings.device

    @device.setter
    def device(self, value: str):
        if value not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        self._save_if_changed("device", value)
