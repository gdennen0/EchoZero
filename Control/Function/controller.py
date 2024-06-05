from Model.tools import prompt, path_exists, file_exists, Log, is_valid_audio_format
from Model.load_audio import load_audio
# Main Controller Functions

class Ingest:
    # Get user input
    path = prompt("Please Enter Path:")
    # validity check
    if not path_exists(path): # Check if the path is valid
        Log.error(f"Invalid Path: '{path}'")
        pass
    if not file_exists(path):   # Check if the file exists at specified path
        Log.error(f"File does not exist at specified path")
        pass
    if not is_valid_audio_format(path): # Check if audio is in a usable format
        Log.error(f"Invalid audio format")
        pass
    
    # Load the audio into the model  
    
class Digest:
    def process_data(self):
        # Placeholder for digest data process implementation
        pass

    