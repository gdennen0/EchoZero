from Block.block import Block
from message import Log
import os
from tools import prompt_selection
from DataTypes.audio_data_type import AudioData
import librosa
from Connections.port_types.audio_port import AudioPort

class LoadAudioBlock(Block):


    def __init__(self):
        super().__init__()
        self.AUDIO_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sources", "audio")
        
        self.selected_file_path = None
        self.selected_file_data = None
        self.set_name("LoadAudio")
        self.add_command("select_file", self.select_file)
        self.add_output_type(AudioData())

        self.add_port_type(AudioPort)
        self.add_output_port(port_name="AudioPort")
        self.link_port_attribute("output", "AudioPort", "data")

    def select_file(self):
        try:
            audio_files = [f for f in os.listdir(self.AUDIO_SOURCE_DIR) if os.path.isfile(os.path.join(self.AUDIO_SOURCE_DIR, f))]
            Log.list("Audio Files", audio_files)
            selected_file, _ = prompt_selection("Please select an audio file: ", audio_files)
            self.selected_file_path = os.path.join(self.AUDIO_SOURCE_DIR, selected_file)
            Log.info(f"Selected file: {self.selected_file_path}")
            self.load_file()
        except Exception as e:
            Log.error(f"Error listing audio files: {e}")
            return

    def load_file(self):
        if self.selected_file_path:
            SAMPLE_RATE = 44100
            try:
                audio_data = AudioData()
                audio_data.set_path(self.selected_file_path)
                audio_data.set_sample_rate(SAMPLE_RATE)
                y, sr = librosa.load(audio_data.path, sr=audio_data.sample_rate)
                audio_data.set_data(y)
                audio_data.set_name = os.path.basename(audio_data.path)
                audio_data.set_length_ms = len(audio_data.data) / audio_data.sample_rate * 1000
                Log.info(f"Audio data generated from path: {self.selected_file_path}")
                self.set_data(audio_data)
            except Exception as e:
                Log.error(f"Error generating audio data from path: {e}")

        else:
            Log.error("No file selected. Please select a file first.")
            return
