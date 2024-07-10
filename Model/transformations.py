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

# class EZ_Separator(Separator):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def separate(self, audio_file_path):
#         """
#         Separates the audio file into different stems (e.g., vocals, instruments) using the loaded model.

#         This method takes the path to an audio file, processes it through the loaded separation model, and returns
#         the separated audio data instead of file paths.

#         Parameters:
#         - audio_file_path (str): The path to the audio file to be separated.

#         Returns:
#         - output_data (dict): A dictionary containing the separated audio data.
#         """
#         self.logger.info(f"Starting separation process for audio_file_path: {audio_file_path}")
#         separate_start_time = time.perf_counter()

#         self.logger.debug(f"Normalization threshold set to {self.normalization_threshold}, waveform will lowered to this max amplitude to avoid clipping.")

#         # Run separation method for the loaded model
#         output_data = self.model_instance.separate(audio_file_path)

#         # Clear GPU cache to free up memory
#         self.model_instance.clear_gpu_cache()

#         # Unset more separation params to prevent accidentally re-using the wrong source files or output paths
#         self.model_instance.clear_file_specific_paths()

#         # Remind the user one more time if they used a VIP model, so the message doesn't get lost in the logs
#         self.print_uvr_vip_message()

#         self.logger.debug("Separation process completed.")
#         self.logger.info(f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - separate_start_time)))}')

#         return output_data
    