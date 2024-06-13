import os
from message import Log
from tools import prompt, yes_no_prompt, path_exists, file_exists, is_valid_audio_format

"""
Responsible for Validating and directing input streams to execute the proper control actions

"""
class Command:
    def __init__(self, model, control):
        self.model = model
        self.control = control
        Log.info("Initialized Command Module")
        self.commands = {
            "ingest" : self.ingest,
            "digest" : self.digest,
            "list_audio_objects" : self.list_audio_objects,
            "delete_audio_object" : self.delete_audio_object,
        }
        pass


    def list_audio_objects(self):
        objects = self.model.audio.objects
        for index, a in enumerate(objects):
            Log.info(f"[{index}] {a.name}")

    def delete_audio_object(self, index):
        self.model.audio.delete(index)

    def ingest(self ,path=None):
        # BEGIN INPUT VALIDATION
        if not path:
            # Get user input if path is not specified
            str_path = str(prompt("Please Enter Path: "))
            abs_path = os.path.abspath(str_path)
        if path:
            abs_path = os.path.abspath(path)
        # validity check
        if not path_exists(abs_path): # Check if the path is valid
            Log.error(f"Invalid Path: '{abs_path}'")
            return
        if not file_exists(abs_path):   # Check if the file exists at specified path
            Log.error(f"File does not exist at specified path")
            return
        if not is_valid_audio_format(abs_path): # Check if audio is in a usable format
            Log.error(f"Invalid audio format")
            return
        # BEGIN LOAD INTO PROGRAM
        # add the audio into the model
        self.control.load_audio(abs_path)

    def digest(self, a=None):
        # apply the pre transformation
        if yes_no_prompt("Apply pre transformation?"):
            Log.info(f"Pre transformation applied to {a.name}")
        # run onset detection
        if yes_no_prompt("run offset transformation?"):
            Log.info("Run offset detection")
        # apply post transformation
        if yes_no_prompt("run post transformation?"):
            Log.info("Apply post transformation")
    