from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.audio_output import AudioOutput

from src.Utils.message import Log
from src.Utils.tools import prompt_selection, prompt
import os
import json

from src.Project.Block.BlockTypes.Transform.Stems.Splitter.Generic.GenericSplitter import GenericSplitter

class StemsBlock(Block):
    name = "Stems"
    type = "Stems"
    
    def __init__(self):
        super().__init__()
        self.name = "Stems"
        self.type = "Stems"
        self.log_level = 3
        self.output_dir = os.path.join(os.getcwd(), "tmp")
        self.normalization_threshold = 1.0
        self.model = None
        
        # Available splitter types
        self.splitter_types = {
            "Generic Splitter": GenericSplitter,
        }
        
        # Default to the standard separator
        self.current_splitter_type = None
        self.splitter = None

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        # Add commands
        self.command.add("select_splitter", self.select_splitter)
        self.command.add("select_model", self.select_model)
        self.command.add("set_output_dir", self.set_output_dir)
        self.command.add("set_normalization_threshold", self.set_normalization_threshold)
        self.command.add("list_supported_models", self.list_supported_models)

    def select_splitter(self, splitter_type=None):
        """Allow user to select which type of splitter to use"""
        if splitter_type is None:
            selected_splitter = prompt_selection(
                "Select a splitter type:",
                list(self.splitter_types.keys())
            )
        else:
            if splitter_type in self.splitter_types:
                selected_splitter = splitter_type
            else:
                Log.error(f"Invalid splitter type: {splitter_type}")
                return

        self.current_splitter_type = selected_splitter
        self.splitter = self.splitter_types[selected_splitter]()
        
        Log.info(f"Selected splitter: {selected_splitter}")
        Log.info(f"Supported input types: {self.splitter.input_types}")
        Log.info(f"Output stem types: {self.splitter.output_stems}")

    def select_model(self):
        """Allow user to select a model for the current splitter"""
        if not self.splitter:
            Log.error("No splitter selected. Please select a splitter first.")
            return
            
        supported_models = self.splitter.list_supported_model_files()
        
        # Create a list of tuples (readable_name, model_filename)
        model_choices = []
        for model_type, models in supported_models.items():
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

    def load_model(self):
        """Load the selected model into the current splitter"""
        if not self.model:
            Log.error("No model selected")
            return
            
        self.splitter.load_model(self.model)
        Log.info(f"Model {self.model} loaded into {self.current_splitter_type}")

    def list_supported_models(self):
        """List all models supported by the current splitter"""
        if not self.splitter:
            Log.error("No splitter selected")
            return
            
        supported_files = self.splitter.list_supported_model_files()
        formatted_files = json.dumps(supported_files, indent=4)
        Log.info(f"Supported model files for {self.current_splitter_type}:\n{formatted_files}")

    def process(self, input_data):
        """Process audio using the current splitter and model"""
        if not self.splitter:
            Log.error("No splitter selected. Please select a splitter with 'select_splitter' command")
            return
            
        if not self.model:
            Log.error("No model selected. Please select a model with 'select_model' command")
            return
            
        processed_data = []
        Log.info(f"Stems separation started using {self.current_splitter_type}")
        Log.info(f"Processing {len(input_data)} Audio Objects")
        
        for obj in input_data:
            if obj.type in self.splitter.input_types:
                Log.info(f"Processing Audio Object: {obj.name}")
                stems = self.splitter.separate(obj)
                Log.info(f"Separation process returned {len(stems)} stems")
                
                # Add all stem audio objects to the result
                for stem_name, stem_audio in stems.items():
                    Log.debug(f"Adding stem: {stem_name}")
                    processed_data.append(stem_audio)
            else:
                Log.warning(f"Skipped object of type {obj.type} - not supported by {self.current_splitter_type}")
                
        return processed_data

    def set_output_dir(self, output_dir):
        """Set the output directory for the splitter"""
        self.output_dir = output_dir
        if self.splitter:
            self.splitter.output_dir = output_dir
        Log.info(f"Output directory set to {self.output_dir}")

    def set_normalization_threshold(self, normalization_threshold):
        """Set the normalization threshold for the splitter"""
        self.normalization_threshold = normalization_threshold
        if self.splitter:
            self.splitter.normalization_threshold = normalization_threshold
        Log.info(f"Normalization threshold set to {self.normalization_threshold}")
    
    def get_metadata(self):
        """Get metadata for saving the block"""
        return {
            "name": self.name,
            "type": self.type,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "model": self.model,
            "current_splitter_type": self.current_splitter_type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def save(self, save_dir):
        """Save the block data"""
        self.data.save(save_dir)

    def load(self, block_dir):
        """Load the block from saved data"""
        block_metadata = self.get_metadata_from_dir(block_dir)

        # Load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_output_dir(block_metadata.get("output_dir"))
        self.set_normalization_threshold(block_metadata.get("normalization_threshold"))
        
        # Load splitter type and model
        splitter_type = block_metadata.get("current_splitter_type")
        if splitter_type in self.splitter_types:
            self.current_splitter_type = splitter_type
            self.splitter = self.splitter_types[splitter_type](
                log_level=self.log_level,
                output_dir=self.output_dir,
                normalization_threshold=self.normalization_threshold
            )
            
            # Load model if specified
            model = block_metadata.get("model")
            if model:
                self.model = model
                self.load_model()

        # Load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # Push the results to the output ports
        self.output.push_all(self.data.get_all())