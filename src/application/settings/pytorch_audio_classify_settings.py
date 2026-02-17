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


