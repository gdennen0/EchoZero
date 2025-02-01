from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.audio_output import AudioOutput

from src.Utils.message import Log
from src.Utils.tools import prompt_selection
import os
from src.Utils.message import Log
import json
from lib.larsnet.larsnet import LarsNet  # Ensure LarsNet is imported
import torch

DEFAULT_LOG_LEVEL = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
DEFAULT_NORMALIZATION_THRESHOLD = 1.0

# Separates audio file into stems
class LarsNetBlock(Block):
    name = "LarsNet"
    type = "LarsNet"
    
    def __init__(self):
        super().__init__()
        self.name = "LarsNet"
        self.type = "LarsNet"
        self.log_level = DEFAULT_LOG_LEVEL
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.normalization_threshold = DEFAULT_NORMALIZATION_THRESHOLD
        self.model = LarsNet(device='cpu')  # Initialize LarsNet model with device

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        self.command.add("set_output_dir", self.set_output_dir)
        self.command.add("set_normalization_threshold", self.set_normalization_threshold)
        self.command.add("set_model", self.set_model)

        Log.info(f"ExtractDrums initialized")

    def process(self, input_data):
        processed_data = []
        Log.info(f"Extract Drums Start Sequence Executed")
        Log.info(f"Processing {len(input_data)} Audio Objects data: {input_data}")
        for object in input_data:
            if object.type == "AudioData":  
                audio_object = object
                Log.info(f"Processing Audio Object: {audio_object.name}")
                                
                # Convert data to tensor and ensure it matches the model's device and dtype
                audio_tensor = torch.tensor(audio_object.data, dtype=torch.float32, device=self.model.device)
                
                # Use LarsNet to separate the audio into stems
                stems = self.model.separate(audio_tensor)

                for stem_name, stem_data in stems.items():
                    Log.debug(f"Stem: {stem_name}, Length: {len(stem_data)}")
                    audio_stem = AudioData()
                    audio_stem.set_name(f"{stem_name}")
                    audio_stem.set_sample_rate(object.sample_rate)
                    audio_stem.set_frame_rate(object.frame_rate)
                    audio_stem.set_length_ms(object.length_ms)
                    audio_stem.set_path(object.path)
                    audio_stem.set_data(stem_data)
                    processed_data.append(audio_stem)
                    Log.info(f"{stem_name} stem found and set")
        return processed_data

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        Log.info(f"Output directory set to {self.output_dir}")

    def set_normalization_threshold(self, normalization_threshold):
        self.normalization_threshold = normalization_threshold
        Log.info(f"Normalization threshold set to {self.normalization_threshold}")

    def set_model(self, model):
        self.model = model
        Log.info(f"Model set to {self.model}")

    def prompt_set_model(self):
        ai_model, _ = prompt_selection("Available models:", model_dict)
        ai_category, _ = prompt_selection("Available Categories:", model_dict[ai_model])
        ai_training, _  = prompt_selection("Available Models:", model_dict[ai_model][ai_category])
        model_value = model_dict[ai_model][ai_category][ai_training]
        return model_value
    

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "output_dir": self.output_dir,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def save(self, save_dir):
        self.data.save(save_dir)

    # def load(self, metadata, block_dir):
    #     self.log_level = data.get("log_level")
    #     self.output_dir = data.get("output_dir")
    #     self.normalization_threshold = data.get("normalization_threshold")
    #     self.model = data.get("model")
    #     self.input.load(data.get("input")) # just need to reconnect the inputs

    #     self.reload()
    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_output_dir(block_metadata.get("output_dir"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())


