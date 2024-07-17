import os
import json
from message import Log

def generate_metadata(stem):
    """
    Generates metadata for a stem.
    """
    stem_metadata = {
        "name": stem.name,
        "path": stem.path,
    }
    metadata_path = os.path.splitext(stem.path)[0] + "_metadata.json"
    try:
        with open(metadata_path, 'w') as f:
            json.dump(stem_metadata, f, indent=4)
            Log.info(f"Metadata for stem '{stem.name}' saved to {metadata_path}")
    except Exception as e:
        Log.error(f"Error during metadata generation: {e}")