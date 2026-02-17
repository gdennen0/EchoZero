"""
TensorFlow Model Loader

Loads TensorFlow/Keras models including SavedModel format (Keras 3 compatible).
"""
from typing import Any, Optional, Tuple
from pathlib import Path
import numpy as np

from src.application.processing.block_processor import ProcessingError
from src.utils.message import Log
from .base_loader import ModelLoader

# Lazy import for TensorFlow/Keras - only import when actually needed
# This prevents slow startup times when TensorFlow is not being used
_tf = None
_keras = None
_HAS_TENSORFLOW = None


def _get_tensorflow():
    """Lazy import TensorFlow - only import when actually needed"""
    global _tf, _keras, _HAS_TENSORFLOW
    
    if _HAS_TENSORFLOW is None:
        try:
            _tf = __import__('tensorflow', globals(), locals(), [], 0)
            _keras = __import__('keras', globals(), locals(), [], 0)
            _HAS_TENSORFLOW = True
        except ImportError:
            _HAS_TENSORFLOW = False
            Log.warning("TensorFlow not available - TensorFlowModelLoader will not work")
    
    return _tf, _keras, _HAS_TENSORFLOW


class TensorFlowModelLoader(ModelLoader):
    """
    Model loader for TensorFlow/Keras models.
    
    Supports:
    - .h5 files (legacy Keras format)
    - .keras files (Keras 3 format)
    - SavedModel directories (using TFSMLayer for Keras 3 compatibility)
    """
    
    def __init__(self):
        """Initialize TensorFlow model loader"""
        # Don't import TensorFlow here - only check availability
        # Actual import happens lazily when load_model is called
        _, _, has_tf = _get_tensorflow()
        if not has_tf:
            raise ImportError(
                "TensorFlow is required for TensorFlowModelLoader. "
                "Install with: pip install tensorflow"
            )
        self._model_cache = None
        self._cached_model_path = None
    
    def supports_format(self, path: str) -> bool:
        """Check if path is a TensorFlow model"""
        format_type = self.detect_format(path)
        return format_type in ["tensorflow", "tensorflow_savedmodel"]
    
    def load_model(self, path: str, tf_python_executable: Optional[str] = None, **kwargs) -> Any:
        """
        Load TensorFlow model from path.
        
        Args:
            path: Path to model file (.h5, .keras) or SavedModel directory
            **kwargs: Additional loading options
            
        Returns:
            Loaded TensorFlow/Keras model
        """
        # Lazy import TensorFlow here (only when actually loading a model)
        tf, keras, has_tf = _get_tensorflow()
        if not has_tf:
            raise ProcessingError(
                "TensorFlow is required. Install with: pip install tensorflow",
                block_id="",
                block_name=""
            )
        
        # Check cache
        if self._model_cache is not None and self._cached_model_path == path:
            Log.debug(f"TensorFlowModelLoader: Using cached model from {path}")
            return self._model_cache
        
        path_obj = Path(path)
        if not path_obj.exists():
            raise ProcessingError(
                f"Model path not found: {path}",
                block_id="",
                block_name=""
            )
        
        Log.info(f"TensorFlowModelLoader: Loading model from {path}")
        
        try:
            # Check if it's a SavedModel directory
            is_saved_model = path_obj.is_dir() and (path_obj / "saved_model.pb").exists()
            
            # If it's a SavedModel, check for .h5 version first (preferred format)
            if is_saved_model:
                h5_path = path_obj.parent / f"{path_obj.name}.h5"
                if h5_path.exists():
                    Log.info(f"TensorFlowModelLoader: Found .h5 version, using instead of SavedModel: {h5_path}")
                    compile_model = kwargs.get("compile", False)
                    model = tf.keras.models.load_model(str(h5_path), compile=compile_model)
                    # Cache the model
                    self._model_cache = model
                    self._cached_model_path = str(h5_path)
                    Log.info("TensorFlowModelLoader: Model loaded successfully from .h5")
                    return model
            
            if is_saved_model:
                # Load SavedModel directly (works with TensorFlow 2.13.0-2.19.x)
                Log.info("TensorFlowModelLoader: Detected SavedModel format, attempting direct load...")
                model = self._load_saved_model(path)
            else:
                # Load .h5 or .keras file
                Log.info("TensorFlowModelLoader: Loading Keras model file")
                compile_model = kwargs.get("compile", False)
                model = tf.keras.models.load_model(path, compile=compile_model)
            
            # Cache the model
            self._model_cache = model
            self._cached_model_path = path
            
            Log.info("TensorFlowModelLoader: Model loaded successfully")
            return model
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to load TensorFlow model: {str(e)}",
                block_id="",
                block_name=""
            ) from e
    
    def _load_saved_model(self, path: str) -> Any:
        """
        Load SavedModel using direct loading (v1 approach).
        
        Works with TensorFlow 2.13.0-2.19.x which can load SavedModels from TensorFlow 2.10.0.
        
        Args:
            path: Path to SavedModel directory
            
        Returns:
            Keras model wrapping the SavedModel
        """
        # Use direct loading with manual wrapping (v1 approach)
        return self._load_saved_model_direct(path)
    
    
    def _load_saved_model_direct(self, path: str) -> Any:
        """
        Load SavedModel using v1 approach (exact match to ez_speedy implementation).
        
        Strategy (matching v1 exactly):
        1. Try tf.keras.models.load_model() with compile=False
        2. Try tf.saved_model.load() with 'serve' tag
        3. Try tf.saved_model.load() without tags
        4. Try tf.compat.v1.saved_model.load_v2()
        5. If Keras model, use directly
        6. If SavedModel object, wrap it (avoiding signature access)
        """
        # Lazy import TensorFlow
        tf, keras, has_tf = _get_tensorflow()
        if not has_tf:
            raise ProcessingError(
                "TensorFlow is required. Install with: pip install tensorflow",
                block_id="",
                block_name=""
            )
        
        Log.info("TensorFlowModelLoader: Detected SavedModel format, attempting to load...")
        
        saved_model = None
        load_error = None
        
        # Strategy 1: Try with compile=False (sometimes works even for SavedModel) - v1 approach
        try:
            Log.info("TensorFlowModelLoader: Trying tf.keras.models.load_model() with compile=False...")
            saved_model = tf.keras.models.load_model(path, compile=False)
            Log.info("TensorFlowModelLoader: Successfully loaded using load_model()")
        except Exception as e1:
            Log.debug(f"TensorFlowModelLoader: load_model() failed: {e1}")
            load_error = e1
            
            # Strategy 2: Try tf.saved_model.load() with different options - v1 approach
            try:
                Log.info("TensorFlowModelLoader: Trying tf.saved_model.load() with 'serve' tag...")
                saved_model = tf.saved_model.load(path, tags=['serve'])
                Log.info("TensorFlowModelLoader: Successfully loaded with 'serve' tag")
            except Exception as e2:
                Log.debug(f"TensorFlowModelLoader: load() with 'serve' tag failed: {e2}")
                try:
                    Log.info("TensorFlowModelLoader: Trying tf.saved_model.load() without tags...")
                    saved_model = tf.saved_model.load(path)
                    Log.info("TensorFlowModelLoader: Successfully loaded without tags")
                except Exception as e3:
                    Log.debug(f"TensorFlowModelLoader: load() without tags failed: {e3}")
                    load_error = e3
                    
                    # Strategy 3: Try using tf.compat.v1 if available - v1 approach
                    try:
                        Log.info("TensorFlowModelLoader: Trying tf.compat.v1.saved_model.load()...")
                        saved_model = tf.compat.v1.saved_model.load_v2(path)
                        Log.info("TensorFlowModelLoader: Successfully loaded using compat.v1")
                    except Exception as e4:
                        Log.debug(f"TensorFlowModelLoader: compat.v1 load failed: {e4}")
                        load_error = e4
        
        if saved_model is None:
            raise ProcessingError(
                f"Failed to load SavedModel after trying multiple methods. "
                f"Last error: {str(load_error)}\n\n"
                f"This may be a TensorFlow/Keras version compatibility issue. "
                f"The model was saved with an older TensorFlow version. "
                f"Consider:\n"
                f"1. Converting the SavedModel to .h5 format\n"
                f"2. Using TensorFlow 2.10 or earlier\n"
                f"3. Re-saving the model with the current TensorFlow version",
                block_id="",
                block_name=""
            )
        
        # Check if we got a Keras model (from load_model()) or SavedModel (from saved_model.load())
        # Keras models can be used directly, SavedModels need wrapping - v1 approach
        if isinstance(saved_model, tf.keras.Model):
            # Successfully loaded as Keras model - use it directly
            Log.info("TensorFlowModelLoader: Model loaded as Keras model, using directly")
            return saved_model
        else:
            # It's a SavedModel object - need to wrap it - v1 approach
            Log.info("TensorFlowModelLoader: Model loaded as SavedModel, creating wrapper...")
            
            # Instead of accessing signatures (which causes the error), 
            # try to use the model as a callable directly or access via __call__
            serving_func = None
            
            # Try 1: Check if saved_model itself is callable - v1 approach
            if callable(saved_model):
                try:
                    # Test with a dummy input to verify it works
                    test_input = tf.constant(np.zeros((1, 128, 100, 3), dtype=np.float32))
                    _ = saved_model(test_input)
                    serving_func = saved_model
                    Log.info("TensorFlowModelLoader: Using saved_model as direct callable")
                except Exception as e:
                    Log.debug(f"TensorFlowModelLoader: saved_model callable test failed: {e}")
            
            # Try 2: Access via __call__ method if it exists - v1 approach
            if serving_func is None and hasattr(saved_model, '__call__'):
                try:
                    test_input = tf.constant(np.zeros((1, 128, 100, 3), dtype=np.float32))
                    _ = saved_model.__call__(test_input)
                    serving_func = saved_model.__call__
                    Log.info("TensorFlowModelLoader: Using saved_model.__call__")
                except Exception as e:
                    Log.debug(f"TensorFlowModelLoader: __call__ test failed: {e}")
            
            # Try 3: Use a lambda wrapper that calls the model - v1 approach
            # This avoids accessing signatures directly
            if serving_func is None:
                def make_predict_func(model_obj):
                    """Create a predict function that calls the model without accessing signatures (v1 approach)"""
                    def predict_func(x):
                        # Convert input to tensor
                        if isinstance(x, np.ndarray):
                            x_tensor = tf.constant(x, dtype=tf.float32)
                        else:
                            x_tensor = x
                        
                        # Try calling the model directly
                        try:
                            result = model_obj(x_tensor)
                        except Exception:
                            # If direct call fails, try with input name
                            # Some models expect named inputs
                            try:
                                result = model_obj({'input': x_tensor})
                            except Exception:
                                # Last resort: try with common input names
                                for input_name in ['input_1', 'input_0', 'x']:
                                    try:
                                        result = model_obj({input_name: x_tensor})
                                        break
                                    except Exception:
                                        continue
                                else:
                                    raise
                        
                        # Extract result from dict if needed
                        if isinstance(result, dict):
                            result = list(result.values())[0]
                        
                        # Convert to numpy
                        if hasattr(result, 'numpy'):
                            return result.numpy()
                        elif isinstance(result, tf.Tensor):
                            return result.numpy()
                        return result
                    return predict_func
                
                serving_func = make_predict_func(saved_model)
                Log.info("TensorFlowModelLoader: Using wrapped predict function")
            
            if serving_func is None:
                raise ProcessingError(
                    "Could not find a callable function in SavedModel. "
                    "The model may not be in the expected format.",
                    block_id="",
                    block_name=""
                )
            
            # Create a wrapper class to make it work like a Keras model - v1 approach
            class SavedModelWrapper:
                def __init__(self, func):
                    self._func = func
                
                def predict(self, x, verbose=0):
                    try:
                        # Convert numpy array to tensor if needed
                        if isinstance(x, np.ndarray):
                            x = tf.constant(x, dtype=tf.float32)
                        
                        # Call the serving function
                        result = self._func(x)
                        
                        # Handle different output formats
                        if isinstance(result, dict):
                            # If output is a dict, get the first value (usually 'output_0' or similar)
                            result = list(result.values())[0]
                        
                        # Convert back to numpy
                        if hasattr(result, 'numpy'):
                            return result.numpy()
                        elif isinstance(result, tf.Tensor):
                            return result.numpy()
                        return result
                    except Exception as e:
                        raise ProcessingError(
                            f"Error during model prediction: {str(e)}",
                            block_id="",
                            block_name=""
                        ) from e
            
            model = SavedModelWrapper(serving_func)
            Log.info("TensorFlowModelLoader: SavedModel wrapped successfully (v1 approach)")
            return model
    
    
    def predict(self, model: Any, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run prediction on TensorFlow model.
        
        Args:
            model: Loaded TensorFlow/Keras model
            input_data: Input data as numpy array
            **kwargs: Additional prediction options (verbose, batch_size, etc.)
            
        Returns:
            Prediction results as numpy array
        """
        verbose = kwargs.get("verbose", 0)
        batch_size = kwargs.get("batch_size", None)
        
        try:
            predictions = model.predict(
                input_data,
                verbose=verbose,
                batch_size=batch_size
            )
            
            # Lazy import TensorFlow
            tf, _, _ = _get_tensorflow()
            
            # Ensure output is numpy array
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            elif isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            
            return predictions
            
        except Exception as e:
            raise ProcessingError(
                f"Prediction failed: {str(e)}",
                block_id="",
                block_name=""
            ) from e
    
    def get_input_shape(self, model: Any) -> Optional[Tuple[int, ...]]:
        """Get input shape from model"""
        try:
            if hasattr(model, 'input_shape'):
                shape = model.input_shape
                if shape:
                    # Remove batch dimension if present
                    if isinstance(shape, list):
                        shape = shape[0]
                    if len(shape) > 1:
                        return tuple(shape[1:])  # Remove batch dim
                    return tuple(shape)
            
            if hasattr(model, 'input'):
                input_layer = model.input
                if hasattr(input_layer, 'shape'):
                    shape = input_layer.shape
                    if shape:
                        return tuple(shape[1:])  # Remove batch dim
            
            return None
        except Exception:
            return None
    
    def get_output_shape(self, model: Any) -> Optional[Tuple[int, ...]]:
        """Get output shape from model"""
        try:
            if hasattr(model, 'output_shape'):
                shape = model.output_shape
                if shape:
                    if isinstance(shape, list):
                        shape = shape[0]
                    if len(shape) > 1:
                        return tuple(shape[1:])  # Remove batch dim
                    return tuple(shape)
            
            if hasattr(model, 'output'):
                output_layer = model.output
                if hasattr(output_layer, 'shape'):
                    shape = output_layer.shape
                    if shape:
                        return tuple(shape[1:])  # Remove batch dim
            
            return None
        except Exception:
            return None

