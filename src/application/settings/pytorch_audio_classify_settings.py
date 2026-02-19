"""
PyTorchAudioClassify Block Settings

Settings schema and manager for PyTorchAudioClassify blocks.
"""
from dataclasses import dataclass
from typing import Optional

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class PyTorchAudioClassifyBlockSettings(BaseSettings):
    """
    Settings schema for PyTorchAudioClassify blocks.

    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Model path (required) - must be from PyTorch Audio Trainer
    model_path: Optional[str] = None  # Path to model file created by PyTorch Audio Trainer

    # Optional processing parameters (defaults from model config if available)
    sample_rate: Optional[int] = None  # Audio sample rate in Hz (uses model config if None)
    batch_size: Optional[int] = None  # Batch size for prediction (None = auto)
    device: str = "cpu"  # Device to use ("cpu", "cuda", or "mps")
    confidence_threshold: Optional[float] = None  # Override model's optimal threshold (None = use model default)

    # Multiclass multi-label: when True, create events for ALL classes above threshold
    # (one input event can produce multiple output events across layers)
    multiclass_multi_label: bool = False
    multiclass_confidence_threshold: float = 0.4  # Min probability to include a class (when multi_label is True)

    # Layer creation: when False, do not create an EventLayer for "other" (binary/positive_vs_other)
    # Events classified as "other" are dropped from output
    create_other_layer: bool = True


class PyTorchAudioClassifySettingsManager(BlockSettingsManager):
    """
    Settings manager for PyTorchAudioClassify blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = PyTorchAudioClassifyBlockSettings
    
    @property
    def model_path(self) -> Optional[str]:
        """Get model file path."""
        return self._settings.model_path
    
    @model_path.setter
    def model_path(self, value: Optional[str]):
        """Set model file path."""
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Model path must be a string or None, got {type(value).__name__}")
        
        if value == "":
            value = None
        
        if value != self._settings.model_path:
            self._settings.model_path = value
            self._save_setting('model_path')
    
    @property
    def sample_rate(self) -> Optional[int]:
        """Get audio sample rate in Hz."""
        return self._settings.sample_rate
    
    @sample_rate.setter
    def sample_rate(self, value: Optional[int]):
        """Set audio sample rate with validation."""
        if value is not None:
            if not isinstance(value, int):
                raise ValueError(f"Sample rate must be an integer or None, got {type(value).__name__}")
            
            if value <= 0:
                raise ValueError(f"Sample rate must be > 0, got {value}")
        
        if value != self._settings.sample_rate:
            self._settings.sample_rate = value
            self._save_setting('sample_rate')
    
    @property
    def batch_size(self) -> Optional[int]:
        """Get batch size for prediction."""
        return self._settings.batch_size
    
    @batch_size.setter
    def batch_size(self, value: Optional[int]):
        """Set batch size with validation."""
        if value is not None:
            if not isinstance(value, int):
                raise ValueError(f"Batch size must be an integer or None, got {type(value).__name__}")
            
            if value <= 0:
                raise ValueError(f"Batch size must be > 0, got {value}")
        
        if value != self._settings.batch_size:
            self._settings.batch_size = value
            self._save_setting('batch_size')
    
    @property
    def device(self) -> str:
        """Get device to use."""
        return self._settings.device
    
    @device.setter
    def device(self, value: str):
        """Set device with validation."""
        if not isinstance(value, str):
            raise ValueError(f"Device must be a string, got {type(value).__name__}")
        
        valid_devices = ["cpu", "cuda", "mps"]
        if value not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}, got '{value}'")
        
        if value != self._settings.device:
            self._settings.device = value
            self._save_setting('device')
    
    @property
    def confidence_threshold(self) -> Optional[float]:
        """Get confidence threshold override (None = use model default)."""
        return self._settings.confidence_threshold
    
    @confidence_threshold.setter
    def confidence_threshold(self, value: Optional[float]):
        """Set confidence threshold with validation."""
        if value is not None:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Confidence threshold must be a number or None, got {type(value).__name__}")
            value = float(value)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {value}")

        if value != self._settings.confidence_threshold:
            self._settings.confidence_threshold = value
            self._save_setting('confidence_threshold')

    @property
    def multiclass_multi_label(self) -> bool:
        """Get whether to allow multiple classes per event (multiclass only)."""
        return self._settings.multiclass_multi_label

    @multiclass_multi_label.setter
    def multiclass_multi_label(self, value: bool):
        """Set multiclass multi-label mode."""
        if value != self._settings.multiclass_multi_label:
            self._settings.multiclass_multi_label = value
            self._save_setting('multiclass_multi_label')

    @property
    def multiclass_confidence_threshold(self) -> float:
        """Get minimum confidence to include a class in multi-label mode."""
        return self._settings.multiclass_confidence_threshold

    @multiclass_confidence_threshold.setter
    def multiclass_confidence_threshold(self, value: float):
        """Set multiclass confidence threshold with validation."""
        v = float(value)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Multiclass confidence threshold must be 0.0-1.0, got {v}")
        if v != self._settings.multiclass_confidence_threshold:
            self._settings.multiclass_confidence_threshold = v
            self._save_setting('multiclass_confidence_threshold')

    @property
    def create_other_layer(self) -> bool:
        """Get whether to create an EventLayer for 'other' classification."""
        return self._settings.create_other_layer

    @create_other_layer.setter
    def create_other_layer(self, value: bool):
        """Set whether to create an EventLayer for 'other' classification."""
        if value != self._settings.create_other_layer:
            self._settings.create_other_layer = value
            self._save_setting('create_other_layer')


