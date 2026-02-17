"""
Percussion Classify Block - Modular Classifier System

Supports multiple classification models as sub-modules.
Each model type is implemented as a separate classifier class.
"""

from typing import Dict, Optional, List
from abc import ABC, abstractmethod

from src.shared.domain.entities import EventDataItem
from src.utils.message import Log


class PercussionClassifier(ABC):
    """
    Abstract base class for percussion classification models.
    
    Each model type (drum-audio-classifier, etc.) implements this interface.
    """
    
    @abstractmethod
    def get_model_type(self) -> str:
        """
        Get the identifier for this model type.
        
        Returns:
            Model type identifier (e.g., "drum_audio_classifier")
        """
        pass
    
    @abstractmethod
    def get_display_name(self) -> str:
        """
        Get human-readable name for this model.
        
        Returns:
            Display name (e.g., "Drum Audio Classifier")
        """
        pass
    
    @abstractmethod
    def classify_events(
        self,
        events: EventDataItem,
        model_config: Dict
    ) -> EventDataItem:
        """
        Classify events using this model.
        
        Args:
            events: Input EventDataItem with events to classify
            model_config: Model-specific configuration from block.metadata
            
        Returns:
            EventDataItem with classified events (classifications added/updated)
            
        Raises:
            Exception: If classification fails
        """
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """
        Get list of required configuration keys for this model.
        
        Returns:
            List of required metadata keys (e.g., ["model_path"])
        """
        pass
    
    @abstractmethod
    def get_optional_config(self) -> List[str]:
        """
        Get list of optional configuration keys for this model.
        
        Returns:
            List of optional metadata keys with defaults
        """
        pass


# Registry of available classifiers
_CLASSIFIER_REGISTRY: Dict[str, type[PercussionClassifier]] = {}


def register_classifier(classifier_class: type[PercussionClassifier]) -> None:
    """
    Register a classifier class.
    
    Args:
        classifier_class: PercussionClassifier subclass to register
    """
    instance = classifier_class()
    model_type = instance.get_model_type()
    _CLASSIFIER_REGISTRY[model_type] = classifier_class
    Log.info(f"Registered percussion classifier: {model_type} ({instance.get_display_name()})")


def get_classifier(model_type: str) -> Optional[PercussionClassifier]:
    """
    Get a classifier instance by model type.
    
    Args:
        model_type: Model type identifier
        
    Returns:
        PercussionClassifier instance or None if not found
    """
    if model_type not in _CLASSIFIER_REGISTRY:
        return None
    
    return _CLASSIFIER_REGISTRY[model_type]()


def list_available_classifiers() -> List[str]:
    """
    List all available classifier model types.
    
    Returns:
        List of model type identifiers
    """
    return list(_CLASSIFIER_REGISTRY.keys())


def get_builtin_model_path(model_type: str = "drum_audio_classifier") -> Optional[str]:
    """
    Get path to built-in model for a given model type.
    
    This ensures the built-in model is always available in the models folder.
    Downloads the model from GitHub if it doesn't exist.
    
    Args:
        model_type: Model type identifier (default: "drum_audio_classifier")
        
    Returns:
        Path to built-in model file, or None if model type doesn't support built-in models
    """
    classifier = get_classifier(model_type)
    if not classifier:
        return None
    
    # Check if classifier has get_builtin_model_path method
    if hasattr(classifier, 'get_builtin_model_path'):
        try:
            return classifier.get_builtin_model_path()
        except Exception as e:
            Log.warning(f"Failed to get built-in model path for {model_type}: {e}")
            return None
    
    return None


def download_builtin_model(model_type: str = "drum_audio_classifier") -> Optional[str]:
    """
    Force download the built-in model from GitHub.
    
    This will download the model even if it already exists (useful for updates).
    
    Args:
        model_type: Model type identifier (default: "drum_audio_classifier")
        
    Returns:
        Path to downloaded model file, or None if download fails
    """
    classifier = get_classifier(model_type)
    if not classifier:
        Log.error(f"Classifier {model_type} not found")
        return None
    
    # Check if classifier has download capability
    if hasattr(classifier, '_download_model_from_github'):
        try:
            from src.utils.paths import get_models_dir
            from pathlib import Path
            models_dir = get_models_dir()
            model_path = models_dir / "drum_audio_classifier_default.h5"
            
            # Force download by removing existing file if present
            if model_path.exists():
                Log.info(f"Removing existing model to force re-download: {model_path}")
                model_path.unlink()
            
            # Call the download method directly
            downloaded_path = classifier._download_model_from_github(model_path, models_dir)
            return str(downloaded_path)
        except Exception as e:
            Log.error(f"Failed to download model for {model_type}: {e}")
            return None
    
    Log.warning(f"Classifier {model_type} does not support direct download")
    return None


# Import and register all classifier modules
# Each classifier module registers itself when imported
try:
    from . import drum_audio_classifier
except ImportError as e:
    Log.debug(f"drum_audio_classifier not available: {e}")




