"""
Built-in Model Management

Manages built-in models that can be automatically downloaded and used.
"""
from typing import Optional, Dict, List
from pathlib import Path
import urllib.request
import urllib.error
import shutil

from src.utils.paths import get_models_dir
from src.utils.message import Log
from src.application.processing.block_processor import ProcessingError


# Built-in model definitions
BUILTIN_MODELS: Dict[str, Dict] = {
    "drum_audio_classifier": {
        "name": "Drum Audio Classifier",
        "description": "CNN-based classifier for drum types (Kick, Snare, Closed Hat, Open Hat, Clap)",
        "framework": "tensorflow",
        "format": "savedmodel",
        "repo": "gdennen0/drum-audio-classifier",
        "branch": "main",
        "model_dir": "saved_model/model_20230607_02",
        "local_dir": "drum_audio_classifier_saved_model",
        "files": [
            "saved_model.pb",
            "variables/variables.data-00000-of-00001",
            "variables/variables.index",
        ],
        # Optional: Path to .h5 file on GitHub (preferred over SavedModel)
        "h5_file": None,  # Set to path like "models/drum_audio_classifier.h5" if available
    }
}


def get_builtin_model_path(model_id: str) -> Optional[str]:
    """
    Get path to built-in model, downloading if necessary.
    Prefers .h5 format over SavedModel for better compatibility.
    
    Args:
        model_id: Built-in model identifier (e.g., "drum_audio_classifier")
        
    Returns:
        Path to model file/directory, or None if model not found or download failed
    """
    if model_id not in BUILTIN_MODELS:
        Log.error(f"Unknown built-in model: {model_id}")
        return None
    
    model_info = BUILTIN_MODELS[model_id]
    models_dir = get_models_dir()
    
    # First, check for .h5 file (preferred format)
    h5_file = models_dir / f"{model_id}.h5"
    if h5_file.exists():
        Log.debug(f"Built-in model '{model_id}' .h5 already exists at {h5_file}")
        return str(h5_file)
    
    # Check for SavedModel directory (fallback)
    if model_info["format"] == "savedmodel":
        local_dir = models_dir / model_info["local_dir"]
        if local_dir.exists() and (local_dir / "saved_model.pb").exists():
            Log.debug(f"Built-in model '{model_id}' SavedModel already exists at {local_dir}")
            return str(local_dir)
    else:
        # For file-based models
        model_file = models_dir / f"{model_id}.{model_info.get('extension', 'h5')}"
        if model_file.exists():
            Log.debug(f"Built-in model '{model_id}' already exists at {model_file}")
            return str(model_file)
    
    # Model doesn't exist, download it
    Log.info(f"Built-in model '{model_id}' not found, downloading...")
    try:
        downloaded_path = download_builtin_model(model_id)
        return downloaded_path
    except Exception as e:
        Log.error(f"Failed to download built-in model '{model_id}': {e}")
        return None


def download_builtin_model(model_id: str, force: bool = False) -> Optional[str]:
    """
    Download a built-in model from GitHub.
    Prefers .h5 format over SavedModel for better compatibility.
    
    Args:
        model_id: Built-in model identifier
        force: If True, re-download even if model exists
        
    Returns:
        Path to downloaded model, or None if download failed
    """
    if model_id not in BUILTIN_MODELS:
        Log.error(f"Unknown built-in model: {model_id}")
        return None
    
    model_info = BUILTIN_MODELS[model_id]
    models_dir = get_models_dir()
    
    # GitHub raw content base URL
    github_base = f"https://raw.githubusercontent.com/{model_info['repo']}/{model_info['branch']}"
    
    # First, try to download .h5 file if specified (preferred format)
    h5_file_path = models_dir / f"{model_id}.h5"
    if model_info.get("h5_file"):
        if h5_file_path.exists() and not force:
            Log.info(f"Built-in model '{model_id}' .h5 already exists at {h5_file_path}")
            return str(h5_file_path)
        
        try:
            url = f"{github_base}/{model_info['h5_file']}"
            Log.info(f"Downloading {model_id} .h5 from {url}...")
            urllib.request.urlretrieve(url, h5_file_path)
            
            if h5_file_path.exists() and h5_file_path.stat().st_size > 0:
                Log.info(f"Built-in model '{model_id}' .h5 downloaded successfully to {h5_file_path}")
                return str(h5_file_path)
            else:
                Log.warning(f"Downloaded .h5 file is empty, falling back to SavedModel")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                Log.info(f".h5 file not found on GitHub (expected if not yet uploaded), falling back to SavedModel")
            else:
                Log.warning(f"HTTP error {e.code} downloading .h5, falling back to SavedModel: {e}")
        except Exception as e:
            Log.warning(f"Failed to download .h5 file, falling back to SavedModel: {e}")
    
    # Fallback to SavedModel download
    if model_info["format"] == "savedmodel":
        # Download SavedModel directory structure
        local_dir = models_dir / model_info["local_dir"]
        
        if local_dir.exists() and not force:
            Log.info(f"Built-in model '{model_id}' already exists at {local_dir}")
            return str(local_dir)
        
        # Create directory
        local_dir.mkdir(exist_ok=True)
        
        # Download each file
        downloaded_files = []
        for file_name in model_info["files"]:
            try:
                # Construct URL
                url = f"{github_base}/{model_info['model_dir']}/{file_name}"
                
                # Construct local path
                local_file_path = local_dir / file_name
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                Log.info(f"Downloading {file_name}...")
                urllib.request.urlretrieve(url, local_file_path)
                
                # Verify download
                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                    downloaded_files.append(local_file_path)
                    Log.info(f"Downloaded {file_name} ({local_file_path.stat().st_size} bytes)")
                else:
                    raise ProcessingError(
                        f"Downloaded file is empty: {file_name}",
                        block_id="",
                        block_name=""
                    )
                    
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise ProcessingError(
                        f"Model file not found: {file_name}\n"
                        f"URL: {url}\n"
                        f"The model may not be available in the repository.",
                        block_id="",
                        block_name=""
                    )
                else:
                    raise ProcessingError(
                        f"HTTP error {e.code} downloading {file_name}",
                        block_id="",
                        block_name=""
                    )
            except Exception as e:
                raise ProcessingError(
                    f"Failed to download {file_name}: {str(e)}",
                    block_id="",
                    block_name=""
                )
        
        # Verify saved_model.pb exists
        saved_model_pb = local_dir / "saved_model.pb"
        if not saved_model_pb.exists():
            raise ProcessingError(
                "Downloaded files but saved_model.pb not found",
                block_id="",
                block_name=""
            )
        
        Log.info(f"Built-in model '{model_id}' downloaded successfully to {local_dir}")
        return str(local_dir)
    
    else:
        # Download single file model
        model_file = models_dir / f"{model_id}.{model_info.get('extension', 'h5')}"
        
        if model_file.exists() and not force:
            Log.info(f"Built-in model '{model_id}' already exists at {model_file}")
            return str(model_file)
        
        # Download file
        url = f"{github_base}/{model_info.get('model_path', model_file.name)}"
        
        try:
            Log.info(f"Downloading {model_id} from {url}...")
            urllib.request.urlretrieve(url, model_file)
            
            if model_file.exists() and model_file.stat().st_size > 0:
                Log.info(f"Built-in model '{model_id}' downloaded successfully to {model_file}")
                return str(model_file)
            else:
                raise ProcessingError(
                    f"Downloaded file is empty",
                    block_id="",
                    block_name=""
                )
        except Exception as e:
            raise ProcessingError(
                f"Failed to download built-in model '{model_id}': {str(e)}",
                block_id="",
                block_name=""
            )


def list_builtin_models(framework: Optional[str] = None) -> List[str]:
    """
    List available built-in models.
    
    Args:
        framework: Optional filter by framework ("tensorflow" or "pytorch")
        
    Returns:
        List of model IDs
    """
    if framework:
        return [
            model_id for model_id, info in BUILTIN_MODELS.items()
            if info.get("framework") == framework
        ]
    return list(BUILTIN_MODELS.keys())


def get_builtin_model_info(model_id: str) -> Optional[Dict]:
    """
    Get information about a built-in model.
    
    Args:
        model_id: Built-in model identifier
        
    Returns:
        Model info dictionary, or None if not found
    """
    return BUILTIN_MODELS.get(model_id)

