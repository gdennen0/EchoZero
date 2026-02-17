"""
PyTorch Model Loader

Loads PyTorch models (.pth, .pt) and ONNX models (.onnx).
"""
from typing import Any, Optional, Tuple
from pathlib import Path
import numpy as np

from src.application.processing.block_processor import ProcessingError
from src.utils.message import Log
from .base_loader import ModelLoader

# Lazy import for PyTorch and ONNX - only import when actually needed
# This prevents slow startup times when PyTorch is not being used
_torch = None
_ort = None
_HAS_PYTORCH = None
_HAS_ONNX = None


def _get_pytorch():
    """Lazy import PyTorch - only import when actually needed"""
    global _torch, _HAS_PYTORCH
    
    if _HAS_PYTORCH is None:
        try:
            _torch = __import__('torch', globals(), locals(), [], 0)
            _HAS_PYTORCH = True
        except ImportError:
            _HAS_PYTORCH = False
            Log.warning("PyTorch not available - PyTorchModelLoader will not work")
    
    return _torch, _HAS_PYTORCH


def _get_onnx():
    """Lazy import ONNX Runtime - only import when actually needed"""
    global _ort, _HAS_ONNX
    
    if _HAS_ONNX is None:
        try:
            _ort = __import__('onnxruntime', globals(), locals(), [], 0)
            _HAS_ONNX = True
        except ImportError:
            _HAS_ONNX = False
            Log.debug("ONNX Runtime not available - ONNX models will not be supported")
    
    return _ort, _HAS_ONNX


class PyTorchModelLoader(ModelLoader):
    """
    Model loader for PyTorch models.
    
    Supports:
    - .pth files (PyTorch state dict or full model)
    - .pt files (PyTorch state dict or full model)
    - .onnx files (ONNX format, requires onnxruntime)
    """
    
    def __init__(self):
        """Initialize PyTorch model loader"""
        # Don't import PyTorch here - only check availability
        # Actual import happens lazily when load_model is called
        _, has_torch = _get_pytorch()
        if not has_torch:
            raise ImportError(
                "PyTorch is required for PyTorchModelLoader. "
                "Install with: pip install torch"
            )
        self._model_cache = None
        self._cached_model_path = None
    
    def supports_format(self, path: str) -> bool:
        """Check if path is a PyTorch or ONNX model"""
        format_type = self.detect_format(path)
        return format_type in ["pytorch", "onnx"]
    
    def load_model(self, path: str, **kwargs) -> Any:
        """
        Load PyTorch or ONNX model from path.
        
        Args:
            path: Path to model file (.pth, .pt, or .onnx)
            **kwargs: Additional loading options:
                - model_class: PyTorch model class (required for state_dict loading)
                - device: Device to load on ('cpu' or 'cuda', default: 'cpu')
                
        Returns:
            Loaded model object
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise ProcessingError(
                f"Model path not found: {path}",
                block_id="",
                block_name=""
            )
        
        # Check cache
        if self._model_cache is not None and self._cached_model_path == path:
            Log.debug(f"PyTorchModelLoader: Using cached model from {path}")
            return self._model_cache
        
        Log.info(f"PyTorchModelLoader: Loading model from {path}")
        
        try:
            ext = path_obj.suffix.lower()
            
            if ext == ".onnx":
                # Load ONNX model
                _, has_onnx = _get_onnx()
                if not has_onnx:
                    raise ProcessingError(
                        "ONNX Runtime is required for ONNX models. "
                        "Install with: pip install onnxruntime",
                        block_id="",
                        block_name=""
                    )
                model = self._load_onnx_model(path)
            else:
                # Load PyTorch model
                device = kwargs.get("device", "cpu")
                model_class = kwargs.get("model_class", None)
                model = self._load_pytorch_model(path, device=device, model_class=model_class)
            
            # Cache the model
            self._model_cache = model
            self._cached_model_path = path
            
            Log.info("PyTorchModelLoader: Model loaded successfully")
            return model
            
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Failed to load PyTorch/ONNX model: {str(e)}",
                block_id="",
                block_name=""
            ) from e
    
    def _load_pytorch_model(self, path: str, device: str = "cpu", model_class: Any = None) -> Any:
        """
        Load PyTorch model from .pth or .pt file.
        
        Args:
            path: Path to model file
            device: Device to load on ('cpu' or 'cuda')
            model_class: Model class (required if file contains state_dict)
            
        Returns:
            Loaded PyTorch model
        """
        # Lazy import PyTorch here (only when actually loading a model)
        torch, has_torch = _get_pytorch()
        if not has_torch:
            raise ProcessingError(
                "PyTorch is required. Install with: pip install torch",
                block_id="",
                block_name=""
            )
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(path, map_location=device)
            
            # Check if it's a state_dict or full model
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Checkpoint format with state_dict
                if model_class is None:
                    raise ProcessingError(
                        "Model file contains state_dict but model_class not provided. "
                        "Please provide the model class in block metadata.",
                        block_id="",
                        block_name=""
                    )
                state_dict = checkpoint["state_dict"]
                model = model_class()
                model.load_state_dict(state_dict)
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) and "." in k for k in checkpoint.keys()):
                # Looks like a state_dict (keys have module paths)
                if model_class is None:
                    raise ProcessingError(
                        "Model file appears to be a state_dict but model_class not provided. "
                        "Please provide the model class in block metadata.",
                        block_id="",
                        block_name=""
                    )
                model = model_class()
                model.load_state_dict(checkpoint)
            else:
                # Assume it's a full model
                model = checkpoint
                if not isinstance(model, torch.nn.Module):
                    raise ProcessingError(
                        "Model file does not contain a valid PyTorch model. "
                        "Expected torch.nn.Module or state_dict.",
                        block_id="",
                        block_name=""
                    )
            
            model.to(device)
            model.eval()  # Set to evaluation mode
            
            return model
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to load PyTorch model: {str(e)}",
                block_id="",
                block_name=""
            ) from e
    
    def _load_onnx_model(self, path: str) -> Any:
        """
        Load ONNX model.
        
        Args:
            path: Path to .onnx file
            
        Returns:
            ONNX Runtime InferenceSession
        """
        # Lazy import ONNX Runtime here (only when actually loading a model)
        ort, has_onnx = _get_onnx()
        if not has_onnx:
            raise ProcessingError(
                "ONNX Runtime is required. Install with: pip install onnxruntime",
                block_id="",
                block_name=""
            )
        
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(
                path,
                providers=['CPUExecutionProvider']  # Use CPU by default
            )
            return session
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to load ONNX model: {str(e)}",
                block_id="",
                block_name=""
            ) from e
    
    def predict(self, model: Any, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run prediction on PyTorch or ONNX model.
        
        Args:
            model: Loaded model (PyTorch model or ONNX session)
            input_data: Input data as numpy array
            **kwargs: Additional prediction options (device, etc.)
            
        Returns:
            Prediction results as numpy array
        """
        try:
            # Check if it's an ONNX model
            if hasattr(model, 'run'):  # ONNX Runtime session
                return self._predict_onnx(model, input_data)
            else:
                # PyTorch model
                device = kwargs.get("device", "cpu")
                return self._predict_pytorch(model, input_data, device=device)
                
        except Exception as e:
            raise ProcessingError(
                f"Prediction failed: {str(e)}",
                block_id="",
                block_name=""
            ) from e
    
    def _predict_pytorch(self, model: Any, input_data: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Run prediction on PyTorch model"""
        # Lazy import PyTorch
        torch, _ = _get_pytorch()
        
        # Convert numpy to torch tensor
        input_tensor = torch.from_numpy(input_data).float()
        input_tensor = input_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert back to numpy
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        elif isinstance(output, (list, tuple)):
            # Handle multiple outputs
            output = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
            output = np.array(output)
        
        return output
    
    def _predict_onnx(self, session: Any, input_data: np.ndarray) -> np.ndarray:
        """Run prediction on ONNX model"""
        # Get input name from session
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        # Return first output (most models have single output)
        if isinstance(outputs, list) and len(outputs) > 0:
            return outputs[0]
        return outputs
    
    def get_input_shape(self, model: Any) -> Optional[Tuple[int, ...]]:
        """Get input shape from model"""
        try:
            # Check if it's an ONNX model
            if hasattr(model, 'get_inputs'):
                inputs = model.get_inputs()
                if inputs:
                    shape = inputs[0].shape
                    if shape:
                        # Convert to tuple, handling dynamic dimensions
                        shape_tuple = tuple(s if isinstance(s, int) else None for s in shape)
                        return shape_tuple
            
            # PyTorch model
            if hasattr(model, 'input_shape'):
                return tuple(model.input_shape)
            
            # Try to get from first layer
            if hasattr(model, 'modules'):
                for module in model.modules():
                    if hasattr(module, 'in_features') or hasattr(module, 'in_channels'):
                        # This is a rough heuristic
                        break
            
            return None
        except Exception:
            return None
    
    def get_output_shape(self, model: Any) -> Optional[Tuple[int, ...]]:
        """Get output shape from model"""
        try:
            # Check if it's an ONNX model
            if hasattr(model, 'get_outputs'):
                outputs = model.get_outputs()
                if outputs:
                    shape = outputs[0].shape
                    if shape:
                        shape_tuple = tuple(s if isinstance(s, int) else None for s in shape)
                        return shape_tuple
            
            # PyTorch model - harder to determine without forward pass
            return None
        except Exception:
            return None

