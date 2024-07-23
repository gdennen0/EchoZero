from message import Log
from audio_separator.separator import *
from message import Log
import json
import time
from Command.Digest.PointTypes.point import Point
from tools import prompt_selection, prompt
import os
# ===================
# Audio Object Class
# ===================


# Separates audio file into stems
class GenerateStems(Point):
    def __init__(self):
        self.name = "generate_stems"
        self.type = "generate_stems"
        self.seperator = None

    def apply(self, audio_object):
        Log.command(f"Command initiated: 'generate_stems'")
        output_dir = os.path.join(audio_object.directory, "stems")
        self.seperator = Separator(log_level=3, output_dir=output_dir, normalization_threshold=1.0)
        self.seperator.load_model(model_filename=_select_model())
        stem_filenames = self.seperator.separate(audio_object.path)

        for filename in stem_filenames:
            path = os.path.join(output_dir, filename)
            Log.info(f"Adding stem with path {filename}")
            audio_object.add_stem(path)
        self.update_audio_metadata(audio_object) # refreshes audio metadata


    def update_audio_metadata(self, a):
        metadata_file_path = os.path.join(a.directory, f"{a.name}_metadata.json")
        metadata = a.get_audio_metadata()
        with open(metadata_file_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=4)
        Log.info(f"Audio metadata written to: {metadata_file_path}")

    
    def generate_stem_metadata(self, metadata, audio_file_path):
        metadata_file_path = os.path.join(audio_file_path, "_metadata.json")
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f)
        Log.info(f"Metadata saved to {metadata_file_path}")
        pass

    def load_model(self, model_name):
        Log.info(f"Running function Model > load_model")
        self.separator.load_model(model_filename=model_name)


def _select_model():
    ai_model, _ = prompt_selection("Available models:", models)
    ai_category, _ = prompt_selection("Available Categories:", models[ai_model])
    ai_training, _  = prompt_selection("Available Models:", models[ai_model][ai_category])
    model_value = models[ai_model][ai_category][ai_training]
    return model_value
 
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