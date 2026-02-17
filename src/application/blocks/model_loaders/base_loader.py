"""
Base Model Loader

Abstract base class for model loaders.
Provides framework-agnostic interface for loading and using ML models.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from pathlib import Path
import numpy as np

from src.utils.message import Log


class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    Each framework (TensorFlow, PyTorch) implements this interface
    to provide consistent model loading and prediction.
    """
    
    @abstractmethod
    def supports_format(self, path: str) -> bool:
        """
        Check if this loader supports the given model path.
        
        Args:
            path: Path to model file or directory
            
        Returns:
            True if this loader can handle the format
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str, **kwargs) -> Any:
        """
        Load a model from the given path.
        
        Args:
            path: Path to model file or directory
            **kwargs: Framework-specific loading options
            
        Returns:
            Loaded model object (framework-specific)
            
        Raises:
            ProcessingError: If model loading fails
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run prediction on the model.
        
        Args:
            model: Loaded model object
            input_data: Input data as numpy array
            **kwargs: Framework-specific prediction options
            
        Returns:
            Prediction results as numpy array
        """
        pass
    
    @abstractmethod
    def get_input_shape(self, model: Any) -> Optional[Tuple[int, ...]]:
        """
        Get the expected input shape for the model.
        
        Args:
            model: Loaded model object
            
        Returns:
            Input shape tuple, or None if shape cannot be determined
        """
        pass
    
    @abstractmethod
    def get_output_shape(self, model: Any) -> Optional[Tuple[int, ...]]:
        """
        Get the output shape for the model.
        
        Args:
            model: Loaded model object
            
        Returns:
            Output shape tuple, or None if shape cannot be determined
        """
        pass
    
    @staticmethod
    def detect_format(path: str) -> Optional[str]:
        """
        Detect model format from path.
        
        Args:
            path: Path to model file or directory
            
        Returns:
            Format string ("tensorflow", "pytorch", "onnx", or None)
        """
        path_obj = Path(path)
        
        # Check for TensorFlow SavedModel (directory with saved_model.pb)
        if path_obj.is_dir() and (path_obj / "saved_model.pb").exists():
            return "tensorflow_savedmodel"
        
        # Check file extensions
        if path_obj.is_file():
            ext = path_obj.suffix.lower()
            
            # TensorFlow formats
            if ext in [".h5", ".keras"]:
                return "tensorflow"
            
            # PyTorch formats
            if ext in [".pth", ".pt"]:
                return "pytorch"
            
            # ONNX format
            if ext == ".onnx":
                return "onnx"
        
        return None









