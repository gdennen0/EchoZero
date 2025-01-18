from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.audio_output import AudioOutput

from src.Utils.message import Log
import librosa
import numpy as np
import os


# Extracts harmonic components from audio using librosa.effects.harmonic
class ExtractHarmonicBlock(Block):
    name = "ExtractHarmonic"
    type = "ExtractHarmonic"

    def __init__(self):
        super().__init__()
        self.name = "ExtractHarmonic"
        self.type = "ExtractHarmonic"

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

    def process(self, input_data):
        processed_data = []
        Log.info(f"Extract Harmonic Start Sequence Executed")
        Log.info(f"Processing {len(input_data)} Audio Objects data")

        for audio_object in input_data:
            if isinstance(audio_object, AudioData):
                Log.info(f"Processing Audio Object: {audio_object.name}")
                Log.info(f"This might take a while...")
                y = audio_object.data
                sr = audio_object.sample_rate

                # Extract harmonic component
                harmonic = librosa.effects.harmonic(y)

                # Create a new AudioData object for harmonic data
                harmonic_audio = AudioData()
                harmonic_audio.set_name(f"{audio_object.name}_harmonic")
                harmonic_audio.set_sample_rate(sr)
                harmonic_audio.set_data(harmonic)

                processed_data.append(harmonic_audio)
                Log.info(f"Harmonic component extracted for: {audio_object.name}")

        return processed_data

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # Load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        # Load sub-components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # Push the results to the output ports
        self.output.push_all(self.data.get_all())