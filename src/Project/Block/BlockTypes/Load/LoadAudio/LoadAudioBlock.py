from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Output.Types.audio_output import AudioOutput

from src.Utils.tools import prompt_selection
import librosa
from src.Utils.message import Log
import os
from pathlib import Path
import json
import numpy as np
class LoadAudioBlock(Block):
    name = "LoadAudio"
    type = "LoadAudio"
    
    def __init__(self):
        super().__init__()
        self.name = "LoadAudio" 
        self.type = "LoadAudio"

        self.audio_source_dir = str(Path(__file__).resolve().parents[6] / "sources" / "audio")
        self.selected_file_path = None

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        self.command.add("select_file", self.select_file)

    def select_file(self):
        audio_files = [f for f in os.listdir(self.audio_source_dir) if os.path.isfile(os.path.join(self.audio_source_dir, f))]
        Log.list("Audio Files", audio_files)
        selected_file = prompt_selection("Please select an audio file: ", audio_files)
        self.set_selected_file_path(os.path.join(self.audio_source_dir, selected_file))
        Log.info(f"Selected file: {self.selected_file_path}")

        self.reload()
        return
    
    def set_selected_file_path(self, path):
        self.selected_file_path = path
        Log.info(f"Set selected file path: {self.selected_file_path}")

    def load_file(self):
        if self.selected_file_path:
            SAMPLE_RATE = 44100
            try:
                audio_data = AudioData()
                audio_data.path = self.selected_file_path
                audio_data.sample_rate = SAMPLE_RATE
                y, sr = librosa.load(audio_data.path, sr=audio_data.sample_rate)
                audio_data.set_data(y)
                audio_data.name = os.path.basename(audio_data.path)
                audio_data.length_ms = len(audio_data.data) / audio_data.sample_rate * 1000
                audio_data.set_source(self)
                Log.info(f"Audio data generated from path: {self.selected_file_path}")
                return audio_data
            except Exception as e:
                Log.error(f"Error generating audio data from path: {e}")

        else:
            Log.error("No file selected. Please select a file first.")
            return

    def process(self, input_data):
        #input_data is not used in this block
        audio_data = []
        audio_data.append(self.load_file())
        return audio_data

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "selected_file_path": self.selected_file_path,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_selected_file_path(block_metadata.get("selected_file_path"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())



