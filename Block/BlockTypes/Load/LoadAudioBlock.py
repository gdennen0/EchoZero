from Block.block import Block
from message import Log
import os
from tools import prompt_selection
from Model.Audio.audio_data import AudioData                  

class LoadAudioBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "LoadAudio"
        self.selected_file = None
        self.audio_dir = "./audio/sources"
        self.add_command("select_file", self.select_file)

    def select_file(self):
        try:
            audio_files = [f for f in os.listdir(self.audio_dir) if os.path.isfile(os.path.join(self.audio_dir, f))]
            Log.list("Audio Files", audio_files)
            self.selected_file = prompt_selection("Please select an audio file: ", audio_files)
            Log.info(f"Selected file: {self.selected_file}")
            
        except Exception as e:
            Log.error(f"Error listing audio files: {e}")
            return

    def start(self):
        if self.selected_file:
            audio = AudioData.generate_from_path(os.path.join(self.audio_dir, self.selected_file))
            return audio
        else:
            Log.error("No file selected. Please select a file first.")
            return
