"""
TensorFlowClassify Block Settings

Settings schema and manager for TensorFlowClassify blocks.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class TensorFlowClassifyBlockSettings(BaseSettings):
    """
    Settings schema for TensorFlowClassify blocks.
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Model path (required)
    model_path: Optional[str] = None  # Path to model file/directory
    
    # Preprocessing configuration (optional JSON string)
    preprocessing_config: Optional[str] = None  # JSON string with preprocessing parameters
    
    # Optional processing parameters
    sample_rate: int = 22050  # Audio sample rate in Hz (for audio preprocessing)
    batch_size: Optional[int] = None  # Batch size for prediction (None = auto)
    
    # Optional TensorFlow version override
    tf_python_executable: Optional[str] = None  # Path to Python executable with specific TensorFlow version (e.g., TF 2.10.0)


class TensorFlowClassifySettingsManager(BlockSettingsManager):
    """
    Settings manager for TensorFlowClassify blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = TensorFlowClassifyBlockSettings
    
    @property
    def model_path(self) -> Optional[str]:
        """Get model file path."""
        return self._settings.model_path
    
    @model_path.setter
    def model_path(self, value: Optional[str]):
        """Set model file path."""
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Model path must be a string or None, got {type(value).__name__}")
        
        # Normalize empty string to None
        if value == "":
            value = None
        
        if value != self._settings.model_path:
            self._settings.model_path = value
            self._save_setting('model_path')
    
    @property
    def preprocessing_config(self) -> Optional[Dict[str, Any]]:
        """Get preprocessing configuration as dict."""
        if not self._settings.preprocessing_config:
            return None
        
        try:
            return json.loads(self._settings.preprocessing_config)
        except json.JSONDecodeError as e:
            Log.warning(f"Invalid preprocessing_config JSON: {e}")
            return None
    
    @preprocessing_config.setter
    def preprocessing_config(self, value: Optional[Dict[str, Any]]):
        """Set preprocessing configuration from dict."""
        if value is None:
            config_str = None
        else:
            try:
                config_str = json.dumps(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid preprocessing config: {e}")
        
        if config_str != self._settings.preprocessing_config:
            self._settings.preprocessing_config = config_str
            self._save_setting('preprocessing_config')
    
    @property
    def sample_rate(self) -> int:
        """Get audio sample rate in Hz."""
        return self._settings.sample_rate
    
    @sample_rate.setter
    def sample_rate(self, value: int):
        """Set audio sample rate with validation."""
        if not isinstance(value, int):
            raise ValueError(f"Sample rate must be an integer, got {type(value).__name__}")
        
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
    def tf_python_executable(self) -> Optional[str]:
        """Get TensorFlow Python executable path."""
        return self._settings.tf_python_executable
    
    @tf_python_executable.setter
    def tf_python_executable(self, value: Optional[str]):
        """Set TensorFlow Python executable path."""
        if value is not None and not isinstance(value, str):
            raise ValueError(f"TensorFlow Python executable must be a string or None, got {type(value).__name__}")
        
        # Normalize empty string to None
        if value == "":
            value = None
        
        # Validate path if provided
        if value is not None:
            path_obj = Path(value)
            if not path_obj.exists():
                Log.warning(f"TensorFlow Python executable path does not exist: {value}")
            elif not path_obj.is_file():
                Log.warning(f"TensorFlow Python executable path is not a file: {value}")
        
        if value != self._settings.tf_python_executable:
            self._settings.tf_python_executable = value
            self._save_setting('tf_python_executable')

