from message import Log
from audio_separator.separator import *
from message import Log
import json
import time

# ===================
# Audio Object Class
# ===================


# Separates audio file into stems
class stem_generation:
    def __init__(self, audio_tensor, sr, input_filepath, output_filepath, ai_model):
        Log.info(f"created stem_generation instance")
        self.at = audio_tensor # torch audio tensor
        self.sr = sr # sample rate
        self.input_filepath = input_filepath
        self.output_file = output_filepath
        self.model_type = ai_model
        self.model_yaml = None
        self.model = None

        self.separator = Separator(log_level=3, output_dir=self.output_file, normalization_threshold=1.0) # Initializes an instance of the separator

    def load_model(self, model_name):
        Log.info(f"Running function Model > load_model")
        self.separator.load_model(model_filename=model_name)

    def separate_stems(self):
        Log.info(f"Running function Model > separate_stems")
        stem_paths = self.separator.separate(self.input_filepath)
        return stem_paths

    # Dictionary (for user selection) of available training models
    models = {
        "mdx" : {
            "vocal": {
                 "kuielab vocals a" : "kuielab_a_vocals.onnx",
                 "kuielab vocals b" : "kuielab_b_vocals.onnx",
            },
            "instr": {
                 "kuielab other a" : "kuielab_a_other.onnx",
                 "kuielab other b" : "kuielab_b_other.onnx",
            },
            "bass": {
                "kuielab bass a" : "kuielab_a_bass.onnx",
                "kuielab bass b" : "kuielab_b_bass.onnx",
            },
            "drum": {
                "kuielab drums a" : "kuielab_a_drums.onnx",
                "kuielab drums b" : "kuielab_b_drums.onnx",
            },
            "util": {},
        },
        "vr arch": {
            "vocal": {},
            "instr": {},
            "bass": {},
            "drum": {},
            "util": {},
        },
        "demucs": {
            "vocal": {},
            "instr": {},
            "bass": {},
            "drum": {},
            "util": {},
        },
    }
    