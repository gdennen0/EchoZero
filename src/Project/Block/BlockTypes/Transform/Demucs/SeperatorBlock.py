from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.audio_output import AudioOutput

from src.Utils.message import Log
from lib.audio_separator.separator.separator import Separator
from src.Utils.tools import prompt_selection, prompt
import os
from src.Utils.message import Log
import json

DEFAULT_LOG_LEVEL = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
DEFAULT_NORMALIZATION_THRESHOLD = 1.0
# DEFAULT_MODEL = "kuielab_a_drums.onnx"

# Separates audio file into stems
class SeperatorBlock(Block):
    name = "Seperator"
    type = "Seperator"
    
    def __init__(self):
        super().__init__()
        self.name = "Seperator"
        self.type = "Seperator"
        self.log_level = DEFAULT_LOG_LEVEL
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.normalization_threshold = DEFAULT_NORMALIZATION_THRESHOLD
        self.model = None

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        self.separator = Separator(log_level=self.log_level, output_dir=self.output_dir, normalization_threshold=self.normalization_threshold)
        # self.separator.load_model(model_filename=self.model)
        
        self.command.add("set_output_dir", self.set_output_dir)
        self.command.add("set_normalization_threshold", self.set_normalization_threshold)
        self.command.add("set_model", self.set_model)
        self.command.add("list_supported_model_files", self.list_supported_model_files)
        self.command.add("load_model", self.load_model)
        self.command.add("select_model", self.select_model)


        Log.info(f"MDXSeperator initialized")

    def load_model(self):
        self.separator.load_model(model_filename=self.model)
        Log.info(f"Model loaded: {self.model}")

    def select_model(self):
        # List supported model files
        supported_files = self.separator.list_supported_model_files()
        
        # Create a list of tuples (readable_name, model_filename) with only the last entry from each model type
        model_choices = []
        for model_type, models in supported_files.items():
            for readable_name, model_info in models.items():
                if isinstance(model_info, dict):
                    # Get the last item in the dictionary
                    last_key, last_value = list(model_info.items())[-1]
                    model_choices.append((readable_name, last_key))
                else:
                    model_choices.append((readable_name, model_info))
        
        # Prompt the user to select a model
        selected_readable_name = prompt_selection(
            "Select a model by its readable name:",
            [name for name, _ in model_choices]
        )
        
        # Find the corresponding model filename
        for readable_name, model_filename in model_choices:
            if readable_name == selected_readable_name:
                Log.info(f"Selected model: {readable_name} ({model_filename})")
                self.model = model_filename
                break
        
        # Load the selected model
        self.load_model()

    def set_selected_model(self, model):
        self.model = model
        Log.info(f"Selected model set to {self.model}")
        self.load_model()

    def list_supported_model_files(self):
        supported_files = self.separator.list_supported_model_files()
        formatted_files = json.dumps(supported_files, indent=4)
        Log.info(f"Supported model files:\n{formatted_files}")

    def process(self, input_data):
        if self.model is None:
            Log.error("No model selected please select a model with 'select_model' command")
            return
        processed_data = []
        Log.info(f"Extract Drums Start Sequence Executed")
        Log.info(f"Processing {len(input_data)} Audio Objects data: {input_data}")
        for object in input_data:
            if object.type == "AudioData":  
                audio_object = object
                Log.info(f"Processing Audio Object: {audio_object.name}")
                stems = self.separator.separate(audio_object) 
                Log.info(f"Separation process returned {len(stems)} stems.")
                for stem_name, stem_data in stems.items():
                    Log.debug(f"Stem: {stem_name}, stem data: {len(stem_data)}")
                    if stem_name == "Drums":
                        audio_object = AudioData()
                        audio_object.set_sample_rate(object.sample_rate)
                        audio_object.set_frame_rate(object.frame_rate)
                        audio_object.set_length_ms(object.length_ms)
                        audio_object.set_path(object.path)
                        audio_object.set_data(stem_data.mean(axis=1))
                        processed_data.append(audio_object) # update the audio object to the drum stem
                        Log.info(f"Drums stem found and set")
            return processed_data

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        self.separator.output_dir = output_dir
        Log.info(f"Output directory set to {self.output_dir}")

    def set_normalization_threshold(self, normalization_threshold):
        self.normalization_threshold = normalization_threshold
        self.separator.normalization_threshold = normalization_threshold
        Log.info(f"Normalization threshold set to {self.normalization_threshold}")

    def set_model(self, model):
        self.model = model
        self.separator.model = model
        Log.info(f"Model set to {self.model}")
    
    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "model": self.model,
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
        self.set_selected_model(block_metadata.get("model"))
        self.set_output_dir(block_metadata.get("output_dir"))
        self.set_normalization_threshold(block_metadata.get("normalization_threshold"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())

