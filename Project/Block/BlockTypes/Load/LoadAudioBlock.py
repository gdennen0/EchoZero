from Project.Block.block import Block
from Project.Data.Types.audio_data import AudioData
from Project.Block.Output.Types.audio_output import AudioOutput

from Utils.tools import prompt_selection
import librosa
from Utils.message import Log
import os
from pathlib import Path

class LoadAudioBlock(Block):
    name = "LoadAudio"
    def __init__(self):
        super().__init__()
        self.name = "LoadAudio" 
        self.type = "LoadAudio"

        self.audio_source_dir = str(Path(__file__).resolve().parents[4] / "sources" / "audio")
        self.selected_file_path = None

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        self.command.add("select_file", self.select_file)

    def select_file(self):
        audio_files = [f for f in os.listdir(self.audio_source_dir) if os.path.isfile(os.path.join(self.audio_source_dir, f))]
        Log.list("Audio Files", audio_files)
        selected_file = prompt_selection("Please select an audio file: ", audio_files)
        self.selected_file_path = os.path.join(self.audio_source_dir, selected_file)
        Log.info(f"Selected file: {self.selected_file_path}")

        self.reload()
        return

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


    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "selected_file_path": self.selected_file_path,
            "data": self.data.save(),
            "input": self.input.save(),
            "output": self.output.save()
        }

    def load(self, data):
        self.selected_file_path = data.get("selected_file_path")
        self.reload()
