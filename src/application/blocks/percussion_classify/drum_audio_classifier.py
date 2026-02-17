"""
Drum Audio Classifier Model

Implementation using the drum-audio-classifier library from:
https://github.com/aabalke/drum-audio-classifier

Classifies 5 drum types: Kick, Snare, Closed Hat, Open Hat, Clap
"""
import os
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import shutil
import urllib.request
import tempfile

from src.shared.domain.entities import EventDataItem, Event, EventLayer
from src.shared.domain.entities import AudioDataItem
from src.application.processing.block_processor import ProcessingError
from src.utils.message import Log
from src.utils.paths import get_models_dir

from . import PercussionClassifier, register_classifier

# Try to import required libraries
try:
    import librosa
    import tensorflow as tf
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    Log.warning("drum-audio-classifier dependencies not available (librosa, tensorflow)")


# Built-in model file name (stored in models directory)
BUILTIN_MODEL_NAME = "drum_audio_classifier_default.h5"

# Model repository information
MODEL_REPO = "gdennen0/drum-audio-classifier"
MODEL_BRANCH = "main"
MODEL_DIR = "saved_model/model_20230607_02"

# Base URL for downloading from the repository
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{MODEL_REPO}/{MODEL_BRANCH}"


class DrumAudioClassifier(PercussionClassifier):
    """
    Drum Audio Classifier using CNN model from drum-audio-classifier library.
    
    This classifier uses a TensorFlow/Keras CNN model trained on 2,700+ drum samples.
    Classifies: Kick Drum, Snare Drum, Closed Hat Cymbal, Open Hat Cymbal, Clap Drum
    
    Supports built-in models - if model_path is None or "builtin", automatically
    downloads and uses the default model.
    """
    
    def __init__(self):
        """Initialize classifier with model caching"""
        self._model_cache = None
        self._cached_model_path = None
    
    def get_model_type(self) -> str:
        """Get model type identifier"""
        return "drum_audio_classifier"
    
    def get_display_name(self) -> str:
        """Get human-readable name"""
        return "Drum Audio Classifier (CNN)"
    
    def get_required_config(self) -> List[str]:
        """Get required configuration keys"""
        return ["model_path"]
    
    def get_optional_config(self) -> List[str]:
        """Get optional configuration keys"""
        return ["sample_rate", "hop_length"]
    
    def get_builtin_model_path(self) -> str:
        """
        Get path to built-in model (downloads if necessary).
        
        This ensures the built-in model is always available in the models folder.
        Users can reference this path in their model_path setting.
        
        Returns:
            Path to built-in model file as string
        """
        return str(self._get_builtin_model_path())
    
    def _get_builtin_model_path(self) -> Path:
        """
        Get path to built-in model, downloading if necessary.
        
        Returns:
            Path to built-in model file or directory (saved_model format)
        """
        models_dir = get_models_dir()
        
        # Check for saved_model directory first (preferred format)
        saved_model_dir = models_dir / "drum_audio_classifier_saved_model"
        if saved_model_dir.exists() and (saved_model_dir / "saved_model.pb").exists():
            Log.debug(f"DrumAudioClassifier: Using existing saved_model at {saved_model_dir}")
            return saved_model_dir
        
        # Check for .h5 file (legacy format)
        model_path = models_dir / BUILTIN_MODEL_NAME
        if model_path.exists():
            Log.debug(f"DrumAudioClassifier: Using existing built-in model at {model_path}")
            return model_path
        
        # Try to find model in drum-audio-classifier package first
        try:
            import drum_audio_classifier
            package_path = Path(drum_audio_classifier.__file__).parent
            
            # The drum-audio-classifier package may have the model in various locations
            # Check the package directory structure more thoroughly
            possible_paths = []
            
            # Check root of package
            possible_paths.extend([
                package_path / "drum_model.h5",
                package_path / "model.h5",
                package_path / "model" / "drum_model.h5",
                package_path / "models" / "drum_model.h5",
            ])
            
            # Check if there's a data or resources directory
            for subdir in ["data", "resources", "assets", "model_data"]:
                possible_paths.extend([
                    package_path / subdir / "drum_model.h5",
                    package_path / subdir / "model.h5",
                ])
            
            # Also check parent directory (in case package is in a subdirectory)
            parent_path = package_path.parent
            possible_paths.extend([
                parent_path / "drum_model.h5",
                parent_path / "model.h5",
            ])
            
            for package_model in possible_paths:
                if package_model.exists():
                    Log.info(f"DrumAudioClassifier: Found model in drum-audio-classifier package at {package_model}")
                    # Copy to our models directory for caching
                    shutil.copy2(package_model, model_path)
                    return model_path
            
            # Try to use the package's own model loading mechanism
            # The drum-audio-classifier might have a function to get or load the model
            try:
                # Check for common model loading patterns
                if hasattr(drum_audio_classifier, 'get_model_path'):
                    package_model_path = drum_audio_classifier.get_model_path()
                    if package_model_path and os.path.exists(package_model_path):
                        Log.info(f"DrumAudioClassifier: Using model from package function: {package_model_path}")
                        shutil.copy2(package_model_path, model_path)
                        return model_path
                
                # Try to instantiate a classifier and see if it has model info
                if hasattr(drum_audio_classifier, 'DrumClassifier') or hasattr(drum_audio_classifier, 'Classifier'):
                    # The package might load the model internally - we can't extract it easily
                    # In this case, we'll need to rely on the package being installed
                    Log.info("DrumAudioClassifier: Package has classifier class, but model is internal")
                    # We'll fall through to download attempt
                    
            except (AttributeError, Exception) as e:
                Log.debug(f"DrumAudioClassifier: Package model function not available: {e}")
                
        except ImportError:
            Log.debug("DrumAudioClassifier: drum-audio-classifier package not installed")
        except Exception as e:
            Log.warning(f"DrumAudioClassifier: Error checking package for model: {e}")
        
        # If package not available, try to download from GitHub
        Log.info("DrumAudioClassifier: Attempting to download model from GitHub...")
        return self._download_model_from_github(model_path, models_dir)
    
    def _download_model_from_github(self, model_path: Path, models_dir: Path) -> Path:
        """
        Download the model from GitHub repository.
        
        The model is in TensorFlow saved_model format (directory structure),
        so we download the entire directory.
        
        Args:
            model_path: Target path for the downloaded model (will be a directory)
            models_dir: Directory where models are stored
            
        Returns:
            Path to downloaded model directory
            
        Raises:
            ProcessingError: If download fails
        """
        try:
            # The model is in saved_model format, so we need to download the directory
            # Create a directory for the saved_model
            saved_model_dir = models_dir / "drum_audio_classifier_saved_model"
            saved_model_dir.mkdir(exist_ok=True)
            
            # Files to download for saved_model format
            files_to_download = [
                f"{MODEL_DIR}/saved_model.pb",
                f"{MODEL_DIR}/variables/variables.data-00000-of-00001",
                f"{MODEL_DIR}/variables/variables.index",
            ]
            
            Log.info(f"DrumAudioClassifier: Downloading saved_model from {MODEL_REPO}")
            
            downloaded_files = []
            for file_path in files_to_download:
                try:
                    url = f"{GITHUB_RAW_BASE}/{file_path}"
                    # Create directory structure
                    local_file_path = saved_model_dir / file_path.replace(f"{MODEL_DIR}/", "")
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    Log.info(f"DrumAudioClassifier: Downloading {file_path}...")
                    urllib.request.urlretrieve(url, local_file_path)
                    
                    # Verify file was downloaded
                    if local_file_path.exists() and local_file_path.stat().st_size > 0:
                        downloaded_files.append(local_file_path)
                        Log.info(f"DrumAudioClassifier: Downloaded {local_file_path.name} ({local_file_path.stat().st_size} bytes)")
                    else:
                        raise ProcessingError(f"Downloaded file is empty: {file_path}", block_id="", block_name="")
                        
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        Log.warning(f"DrumAudioClassifier: File not found: {file_path}")
                        # Try alternative paths
                        alt_paths = [
                            file_path.replace("saved_model.pb", "model.h5"),
                            file_path.replace(f"{MODEL_DIR}/", ""),
                        ]
                        for alt_path in alt_paths:
                            try:
                                alt_url = f"{GITHUB_RAW_BASE}/{alt_path}"
                                urllib.request.urlretrieve(alt_url, local_file_path)
                                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                                    downloaded_files.append(local_file_path)
                                    Log.info(f"DrumAudioClassifier: Found alternative: {alt_path}")
                                    break
                            except:
                                continue
                        if not any(f.name == Path(file_path).name for f in downloaded_files):
                            raise ProcessingError(
                                f"Could not download {file_path}. Model files may not be available in the repository.",
                                block_id="",
                                block_name=""
                            )
                    else:
                        raise ProcessingError(f"HTTP error {e.code} downloading {file_path}", block_id="", block_name="")
            
            if not downloaded_files:
                raise ProcessingError(
                    "No model files were downloaded. The model may not be available in the repository.",
                    block_id="",
                    block_name=""
                )
            
            # If we downloaded saved_model.pb, use the directory; otherwise use the .h5 file
            saved_model_pb = saved_model_dir / "saved_model.pb"
            if saved_model_pb.exists():
                # Use the saved_model directory
                Log.info(f"DrumAudioClassifier: Model downloaded successfully to {saved_model_dir}")
                return saved_model_dir
            else:
                # Try to find .h5 file
                h5_files = list(saved_model_dir.glob("*.h5"))
                if h5_files:
                    h5_file = h5_files[0]
                    # Move to expected location
                    shutil.move(str(h5_file), str(model_path))
                    shutil.rmtree(saved_model_dir)  # Clean up temp directory
                    Log.info(f"DrumAudioClassifier: Model downloaded successfully to {model_path}")
                    return model_path
                else:
                    raise ProcessingError(
                        "Downloaded files but could not find model file (saved_model.pb or .h5)",
                        block_id="",
                        block_name=""
                    )
            
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Unexpected error downloading model: {str(e)}\n\n"
                f"To manually download:\n"
                f"1. Visit https://github.com/{MODEL_REPO}\n"
                f"2. Navigate to {MODEL_DIR}\n"
                f"3. Download the model files and place them in: {models_dir}\n"
                f"4. Or provide your own model_path in block settings.",
                block_id="",
                block_name=""
            ) from e
    
    def _load_model(self, model_path: str):
        """
        Load or retrieve cached TensorFlow model.
        
        Args:
            model_path: Path to saved model directory or file
        
        Returns:
            Loaded TensorFlow model
        """
        # Check cache
        if self._model_cache is not None and self._cached_model_path == model_path:
            Log.debug(f"DrumAudioClassifier: Using cached model from {model_path}")
            return self._model_cache
        
        if not HAS_DEPENDENCIES:
            raise ProcessingError(
                "Required dependencies not available. "
                "Install with: pip install librosa tensorflow",
                block_id="",
                block_name=""
            )
        
        if not os.path.exists(model_path):
            raise ProcessingError(
                f"Model path not found: {model_path}. "
                "Please check the model_path in block metadata, or use built-in model by leaving model_path empty.",
                block_id="",
                block_name=""
            )
        
        Log.info(f"DrumAudioClassifier: Loading model from {model_path}")
        
        try:
            # Check if model_path is a directory (saved_model format) or file (.h5)
            model_path_obj = Path(model_path)
            is_saved_model = model_path_obj.is_dir() and (model_path_obj / "saved_model.pb").exists()
            
            if is_saved_model:
                # Keras 3 doesn't support load_model() for SavedModel format
                # The 'add_slot' error is a known TensorFlow/Keras compatibility issue
                # Try multiple loading strategies to work around it
                Log.info("DrumAudioClassifier: Detected SavedModel format, attempting to load...")
                
                saved_model = None
                load_error = None
                
                # Strategy 1: Try with compile=False (sometimes works even for SavedModel)
                try:
                    Log.info("DrumAudioClassifier: Trying tf.keras.models.load_model() with compile=False...")
                    saved_model = tf.keras.models.load_model(model_path, compile=False)
                    Log.info("DrumAudioClassifier: Successfully loaded using load_model()")
                except Exception as e1:
                    Log.debug(f"DrumAudioClassifier: load_model() failed: {e1}")
                    load_error = e1
                    
                    # Strategy 2: Try tf.saved_model.load() with different options
                    try:
                        Log.info("DrumAudioClassifier: Trying tf.saved_model.load() with 'serve' tag...")
                        # Disable eager execution temporarily to avoid some compatibility issues
                        saved_model = tf.saved_model.load(model_path, tags=['serve'])
                        Log.info("DrumAudioClassifier: Successfully loaded with 'serve' tag")
                    except Exception as e2:
                        Log.debug(f"DrumAudioClassifier: load() with 'serve' tag failed: {e2}")
                        try:
                            Log.info("DrumAudioClassifier: Trying tf.saved_model.load() without tags...")
                            saved_model = tf.saved_model.load(model_path)
                            Log.info("DrumAudioClassifier: Successfully loaded without tags")
                        except Exception as e3:
                            Log.debug(f"DrumAudioClassifier: load() without tags failed: {e3}")
                            load_error = e3
                            
                            # Strategy 3: Try using tf.compat.v1 if available
                            try:
                                Log.info("DrumAudioClassifier: Trying tf.compat.v1.saved_model.load()...")
                                saved_model = tf.compat.v1.saved_model.load_v2(model_path)
                                Log.info("DrumAudioClassifier: Successfully loaded using compat.v1")
                            except Exception as e4:
                                Log.debug(f"DrumAudioClassifier: compat.v1 load failed: {e4}")
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
                # Keras models can be used directly, SavedModels need wrapping
                if isinstance(saved_model, tf.keras.Model):
                    # Successfully loaded as Keras model - use it directly
                    Log.info("DrumAudioClassifier: Model loaded as Keras model, using directly")
                    model = saved_model
                else:
                    # It's a SavedModel object - need to wrap it
                    Log.info("DrumAudioClassifier: Model loaded as SavedModel, creating wrapper...")
                    
                    # Instead of accessing signatures (which causes the error), 
                    # try to use the model as a callable directly or access via __call__
                    serving_func = None
                
                # Try 1: Check if saved_model itself is callable
                if callable(saved_model):
                    try:
                        # Test with a dummy input to verify it works
                        test_input = tf.constant(np.zeros((1, 128, 100, 3), dtype=np.float32))
                        _ = saved_model(test_input)
                        serving_func = saved_model
                        Log.info("DrumAudioClassifier: Using saved_model as direct callable")
                    except Exception as e:
                        Log.debug(f"DrumAudioClassifier: saved_model callable test failed: {e}")
                
                # Try 2: Access via __call__ method if it exists
                if serving_func is None and hasattr(saved_model, '__call__'):
                    try:
                        test_input = tf.constant(np.zeros((1, 128, 100, 3), dtype=np.float32))
                        _ = saved_model.__call__(test_input)
                        serving_func = saved_model.__call__
                        Log.info("DrumAudioClassifier: Using saved_model.__call__")
                    except Exception as e:
                        Log.debug(f"DrumAudioClassifier: __call__ test failed: {e}")
                
                # Try 3: Use a lambda wrapper that calls the model
                # This avoids accessing signatures directly
                if serving_func is None:
                    def make_predict_func(model_obj):
                        """Create a predict function that calls the model without accessing signatures"""
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
                    Log.info("DrumAudioClassifier: Using wrapped predict function")
                
                    if serving_func is None:
                        raise ProcessingError(
                            "Could not find a callable function in SavedModel. "
                            "The model may not be in the expected format.",
                            block_id="",
                            block_name=""
                        )
                    
                    # Create a wrapper class to make it work like a Keras model
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
            else:
                # Load .h5 file format
                Log.info("DrumAudioClassifier: Loading .h5 model file")
                model = tf.keras.models.load_model(model_path)
            
            # Cache the model
            self._model_cache = model
            self._cached_model_path = model_path
            
            Log.info("DrumAudioClassifier: Model loaded successfully")
            return model
            
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Failed to load model: {str(e)}",
                block_id="",
                block_name=""
            ) from e
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio for CNN model.
        
        The drum-audio-classifier expects:
        - Mel spectrogram representation
        - Shape: (time, frequency) -> converted to (frequency, time, 3) for CNN
        - Audio is duplicated across 3 channels to make CNN "think" it's a color image
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Preprocessed array ready for model input
        """
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=0)
        else:
            audio_mono = audio_data
        
        # Compute mel spectrogram
        # Using parameters similar to drum-audio-classifier
        mel_spec = librosa.feature.melspectrogram(
            y=audio_mono,
            sr=sample_rate,
            n_mels=128,
            fmax=8000
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1 range
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
        
        # Reshape for CNN: (frequency, time) -> (frequency, time, 3)
        # Duplicate across 3 channels to make CNN think it's a color image
        if len(mel_spec_normalized.shape) == 2:
            mel_spec_rgb = np.stack([mel_spec_normalized] * 3, axis=-1)
        else:
            mel_spec_rgb = mel_spec_normalized
        
        # Add batch dimension: (1, frequency, time, 3)
        mel_spec_batch = np.expand_dims(mel_spec_rgb, axis=0)
        
        return mel_spec_batch
    
    def _predict_drum_type(self, model, audio_data: np.ndarray, sample_rate: int) -> tuple[str, float, Dict[str, float]]:
        """
        Predict drum type for audio sample.
        
        Args:
            model: Loaded TensorFlow model
            audio_data: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Tuple of (predicted_label, confidence, probabilities_dict)
            - predicted_label: Predicted drum type label
            - confidence: Confidence score for predicted class (0-1)
            - probabilities_dict: Dictionary mapping class labels to probabilities
        """
        # Preprocess audio
        preprocessed = self._preprocess_audio(audio_data, sample_rate)
        
        # Predict
        predictions = model.predict(preprocessed, verbose=0)
        
        # Get class labels (from drum-audio-classifier)
        class_labels = [
            "Kick Drum",
            "Snare Drum",
            "Closed Hat Cymbal",
            "Open Hat Cymbal",
            "Clap Drum"
        ]
        
        # Get predicted class
        predicted_idx = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        # Build probabilities dictionary
        probabilities = {
            label: float(predictions[0][i])
            for i, label in enumerate(class_labels)
        }
        
        Log.debug(
            f"DrumAudioClassifier: Predicted '{predicted_label}' "
            f"with confidence {confidence:.3f}"
        )
        
        return predicted_label, confidence, probabilities
    
    def classify_events(
        self,
        events: EventDataItem,
        model_config: Dict
    ) -> EventDataItem:
        """
        Classify events using the drum-audio-classifier model.
        
        Args:
            events: Input EventDataItem with events to classify
            model_config: Configuration dict with:
                - model_path: Path to saved model
                - sample_rate: Optional, default 22050
                - hop_length: Optional, default 512
                
        Returns:
            EventDataItem with classifications added
        """
        if not HAS_DEPENDENCIES:
            raise ProcessingError(
                "Required dependencies not available. "
                "Install with: pip install librosa tensorflow",
                block_id="",
                block_name=""
            )
        
        # Get configuration
        model_path = model_config.get("model_path")
        if not model_path:
            raise ProcessingError(
                "model_path required in block metadata for drum_audio_classifier. "
                "Use get_builtin_model_path() to get the built-in model path.",
                block_id="",
                block_name=""
            )
        
        sample_rate = model_config.get("sample_rate", 22050)
        
        # Load model
        model = self._load_model(model_path)
        
        # Get input events
        input_events = events.get_events()
        if not input_events:
            Log.warning("DrumAudioClassifier: No events to classify")
            return events
        
        total_events = len(input_events)
        Log.info(f"DrumAudioClassifier: Classifying {total_events} events")
        
        # Collect events by classification for EventLayers
        # Structure: EventDataItem -> EventLayers -> Events
        from collections import defaultdict
        from datetime import datetime
        events_by_classification = defaultdict(list)
        
        # Process each event
        classified_count = 0
        failed_count = 0
        skipped_count = 0
        
        for event in input_events:
            # Check if event has associated audio data
            # Events from DetectOnsets may have audio clips in metadata
            audio_path = (
                event.metadata.get("audio_path") or
                event.metadata.get("file_path") or
                event.metadata.get("audio_file")
            )
            
            if audio_path and os.path.exists(audio_path):
                # Load and classify audio
                try:
                    audio_data, sr = librosa.load(audio_path, sr=sample_rate)
                    classification, confidence, probabilities = self._predict_drum_type(model, audio_data, sr)
                    
                    classified_event = Event(
                        time=event.time,
                        classification=classification,
                        duration=event.duration,
                        metadata={
                            **event.metadata,  # This preserves audio_name, _original_source_item_id, etc.
                            "classified_by": "drum_audio_classifier",
                            "model_path": model_path,
                            "original_classification": event.classification,
                            "classification_confidence": confidence,
                            "classification_probabilities": probabilities,
                            "classification_timestamp": datetime.now().isoformat(),
                        }
                    )
                    events_by_classification[classification].append(classified_event)
                    classified_count += 1
                    
                except Exception as e:
                    Log.warning(
                        f"DrumAudioClassifier: Failed to classify event at {event.time}s: {e}"
                    )
                    # Pass through with original classification
                    error_classification = event.classification or "unknown"
                    error_event = Event(
                        time=event.time,
                        classification=error_classification,
                        duration=event.duration,
                        metadata={
                            **event.metadata,
                            "classification_error": str(e)
                        }
                    )
                    events_by_classification[error_classification].append(error_event)
                    failed_count += 1
            else:
                # No audio data - pass through unchanged
                Log.debug(
                    f"DrumAudioClassifier: Event at {event.time}s has no audio data, "
                    "passing through unchanged"
                )
                skipped_classification = event.classification or "unknown"
                skipped_event = Event(
                    time=event.time,
                    classification=skipped_classification,
                    duration=event.duration,
                    metadata={
                        **event.metadata,
                        "note": "No audio data available for classification"
                    }
                )
                events_by_classification[skipped_classification].append(skipped_event)
                skipped_count += 1
        
        # Create EventLayers from grouped events
        layers = []
        for classification, layer_events in events_by_classification.items():
            if layer_events:
                layer = EventLayer(
                    name=classification,
                    events=layer_events,
                    metadata={
                        "source": "drum_audio_classifier",
                        "event_count": len(layer_events)
                    }
                )
                layers.append(layer)
        
        # Create output EventDataItem with EventLayers
        output_events = EventDataItem(
            id="",
            block_id=events.block_id,
            name=f"{events.name}_classified",
            type="Event",
            metadata={
                "classified_by": "drum_audio_classifier",
                "classification_summary": {
                    "total": total_events,
                    "classified": classified_count,
                    "failed": failed_count,
                    "skipped": skipped_count
                }
            },
            layers=layers  # SINGLE SOURCE OF TRUTH: EventLayers
        )
        
        Log.info(
            f"DrumAudioClassifier: Classification complete - "
            f"Total: {total_events}, Classified: {classified_count}, "
            f"Failed: {failed_count}, Skipped: {skipped_count}, "
            f"Layers: {len(layers)}"
        )
        
        return output_events


# Auto-register this classifier
register_classifier(DrumAudioClassifier)




