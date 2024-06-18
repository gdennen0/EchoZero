from message import Log
from audio_separator.separator import Separator
from message import Log
import json

# ===================
# Audio Object Class
# ===================


# Separates audio file into stems
class stem_generation:
    def __init__(self, audio_tensor, sr, input_filepath, output_filepath, ai_model):
        Log.info(f"__init__ class stem_generation")
        self.at = audio_tensor # torch audio tensor
        self.sr = sr # sample rate
        self.input_filepath = input_filepath
        self.output_file = output_filepath
        self.model_type = ai_model
        self.model_yaml = None
        self.model = None

        # Log.info(f"Audio tensor set: {self.at}")
        # Log.info(f"Sample rate set: {self.sr}")
        # Log.info(f"Input filepath set: {self.input_filepath}")
        # Log.info(f"Output filepath set: {self.output_file}")
        # Log.info(f"Model type set: {self.model_type}")


        self.separator = Separator(log_level=3, output_dir=self.output_file, normalization_threshold=0.9) # Initializes an instance of the separator
        self.stems = self.separate_stems()


    def load_model(self):
        Log.info(f"Running function Model > load_model")
        #self.separator.load_model(model_filename="f7e0c4bc-ba3fe64a.th")
        self.separator.load_model(model_filename="kuielab_a_vocals.onnx")

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

        
    def separate_stems(self):

        
        Log.info(f"Running function Model > separate_stems")
        self.load_model()

        # Check if separator is initialized and has the 'separate' method
        if not self.separator or not hasattr(self.separator, 'separate'):
            Log.error("Separator instance is not initialized or 'separate' method is missing.")
            return            

        # Proceed if separator is initialized
        output_file_name = f"output_file.wav"
        Log.info(output_file_name)
        try:
            # Log.info(f"trying to seperate file at path {self.input_filepath}")
            self.separator.separate(self.input_filepath)
        except Exception as e:
            Log.error(f"Error separating stems: {str(e)}")