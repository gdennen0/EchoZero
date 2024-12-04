from message import Log
from audio_separator.separator import Separator
from tools import prompt_selection
import os
from Data.Types.audio_data import AudioData
from message import Log
import soundfile as sf


DEFAULT_LOG_LEVEL = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
DEFAULT_NORMALIZATION_THRESHOLD = 1.0
DEFAULT_MODEL = "kuielab_a_drums.onnx"

# Separates audio file into stems
class ExtractDrums(Part):
    name = "ExtractDrums"
    def __init__(self):
        super().__init__()
        self.set_name("extract_drums")  # maybe uncessary now?
        self.set_type("Transform")

        self.command.add("set_log_level", self.set_log_level)

        self.log_level = DEFAULT_LOG_LEVEL
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.normalization_threshold = DEFAULT_NORMALIZATION_THRESHOLD
        self.model = DEFAULT_MODEL

        self.separator = Separator(log_level=self.log_level, output_dir=self.output_dir, normalization_threshold=self.normalization_threshold)
        self.separator.load_model(model_filename=self.model)

        Log.info(f"ExtractDrums initialized")

    def start(self, data):
        results = []
        Log.info(f"Extract Drums Start Sequence Executed")
        Log.info(f"Processing {len(data)} Audio Objects data: {data}")
        for audio_object in data:
            Log.info(f"Processing Audio Object: {audio_object.name}")
            if not isinstance(audio_object, AudioData):
                Log.error(f"Input is not an AudioData object")
                return
            
            stem_sources = self.separator.separate(audio_object) #this should return a dict of numpy arrays and stem names for keys
            stem_filenames = []
            sample_rate = audio_object.sample_rate

            for stem_name, stem_source in stem_sources.items():
                filename = f'{stem_name}.wav'
                filepath = os.path.join(self.output_dir, filename)
                # sf.write(filepath, stem_source, sample_rate) # write the stem to a file
                # Log.info(f"Stem {stem_name} written to {filepath}")
                stem_filenames.append(filename)
                Log.info(f"Stem source: {stem_name}")
                if stem_name == "Drums":
                    audio_object.set_data(stem_source) # update the audio object to the drum stem
                    Log.info(f"Drums stem found and set")
        return audio_object
    

# Core Methods
    def set_log_level(self, log_level):
        self.log_level = log_level
        Log.info(f"Log level set to {self.log_level}")

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        self.separator.output_dir = output_dir
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
    
# Dictionary (for user selection) of available training models
model_dict = {
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



"""
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

"""