from src.Project.Block.BlockTypes.Transform.Stems.Splitter.SplitterBase import SplitterBase
from lib.audio_separator.separator.separator import Separator
from src.Project.Data.Types.audio_data import AudioData
from src.Utils.message import Log
import os
import json

DEFAULT_LOG_LEVEL = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
DEFAULT_NORMALIZATION_THRESHOLD = 1.0
DEFAULT_MODEL = "kuielab_a_drums.onnx"

class GenericSplitter(SplitterBase):
    """Implementation of SplitterBase using the audio_separator library"""
    name = "Demucs"
    type = "Demucs"

    def __init__(self):
        super().__init__()
        self.name = "ExtractDrums"
        self.type = "ExtractDrums"
        self.log_level = DEFAULT_LOG_LEVEL
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.normalization_threshold = DEFAULT_NORMALIZATION_THRESHOLD
        self.model = DEFAULT_MODEL

        self.separator = Separator(
            log_level=self.log_level, 
            output_dir=self.output_dir, 
            normalization_threshold=self.normalization_threshold
        )

        self.options.add("output_dir", self.output_dir)
        self.options.add("normalization_threshold", self.normalization_threshold)
        self.options.add("model", self.model)

    def set_option(self, name, value):
        self.options.set(name, value)

    def get_option(self, name):
        return self.options.get(name)
    
    def list_options(self):
        return self.options.list()

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

    @property
    def supported_models(self):
        return self.separator.list_supported_model_files()
        
    def load_model(self, model_filename):
        self.model = model_filename
        self.separator.load_model(model_filename=model_filename)
        
    def separate(self, audio_object):
        """
        Separate audio into stems using the separator library
        Returns a dictionary of stem_name: audio_data
        """
        raw_stems = self.separator.separate(audio_object)
        processed_stems = {}
        
        for stem_name, stem_data in raw_stems.items():
            stem_audio = AudioData()
            stem_audio.set_sample_rate(audio_object.sample_rate)
            stem_audio.set_frame_rate(audio_object.frame_rate)
            stem_audio.set_length_ms(audio_object.length_ms)
            stem_audio.set_path(audio_object.path)
            stem_audio.set_data(stem_data.mean(axis=1))
            stem_audio.set_name(f"{audio_object.name}_{stem_name}")
            processed_stems[stem_name] = stem_audio
            
        return processed_stems
        
    @property
    def input_types(self):
        return ["AudioData"]
        
    @property
    def output_stems(self):
        # This will depend on the loaded model, but typically includes:
        return ["Vocals", "Drums", "Bass", "Other"]
    
    @property
    def supported_models(self):
        return self.separator.list_supported_model_files()
    

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