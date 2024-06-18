import os
from message import Log
from tools import prompt, yes_no_prompt, check_audio_path
from Control.load_audio import load_audio
from Control.audio_transformation import stem_separation

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
            "select_audio": self.select_audio,
            "ingest_to_stems": self.ingest_to_stems,
        }
        self.stems = None
        
    def list_audio_objects(self):
        self.model.audio.list()
        Log.command("list_audio_objects")

    def delete_audio_object(self, index):
        self.model.audio.delete(index)
        Log.command(f"delete_audio_object at index {index}")
                    
    def select_audio(self):
        index = prompt("Please enter the index for the audio object you'd like to select")
        self.select_audio(index)

    def ingest(self, path=None, opath=None):
        # BEGIN INPUT VALIDATION
        if not path:
            # Get user input if path is not specified
            path = str(prompt("Please Enter Path: "))
        
        if check_audio_path(path):
            abs_path = os.path.abspath(path)
            # add the audio into the model
            self.load_audio(abs_path)
            # CALL THE ingest_to_stems function here..

    def load_audio(self, abs_path):
        a, sr = load_audio(abs_path)
        self.model.audio.add(a)

    def select_audio(self, index):
        self.model.audio.select(index)

    def generate_stems(self, a_index=a_index):
        Log.command(f"Command initiated: 'generate_stems'")
        if not a_index:
            self.list_audio_objects()
            a_index = prompt("Please enter index for audio object you would like to generate stems for")

        audio_file_path = self.model.audio.get_audio_file_path(a_index)
        stems_path = self.model.audio.get_stems_file_path(a_index)

        stems = stem_separation(None, None, audio_file_path, stems_path, "Demucs")
        self.model.audio.add_stems(stems)     

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
    


















