import os
from Model.tools import prompt, path_exists, file_exists, Log, is_valid_audio_format
from Model.load_audio import load_audio
# Main Controller Functions
class Control:
    def __init__(self, model, command):
        self.model = model          # i can see everything
        self.command = command      # we have all the power now muahaa

    def ingest(self):
        # Get user input
        str_path = str(prompt("Please Enter Path:"))
        abs_path = os.path.abspath(str_path)
        # validity check
        if not path_exists(abs_path): # Check if the path is valid
            Log.error(f"Invalid Path: '{abs_path}'")
            pass
        if not file_exists(abs_path):   # Check if the file exists at specified path
            Log.error(f"File does not exist at specified path")
            pass
        if not is_valid_audio_format(abs_path): # Check if audio is in a usable format
            Log.error(f"Invalid audio format")
            pass
        
        # Load the audio into the model
        a = load_audio(abs_path) #creates audio object 
        # add the audio into the model
        self.command.add_audio(a)
        Log.debug(f"Audio Ingested! yum..")

        
    def digest():
        def process_data(self):
            # Placeholder for digest data process implementation
            pass

    