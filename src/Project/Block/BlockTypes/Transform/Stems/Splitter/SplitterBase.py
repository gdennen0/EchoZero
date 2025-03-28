from abc import ABC, abstractmethod
from src.Utils.message import Log
from src.Command.command_controller import CommandController
from src.Project.Data.data_controller import DataController
from src.Project.Block.Input.input_controller import InputController
from src.Project.Block.Output.output_controller import OutputController

class OptionsController:
    def __init__(self, block):
        self.block = block
        self.options = {}

    def add(self, name, value):
        if name in self.options:
            Log.error(f"Option {name} already exists")
            return
        self.options[name] = value

    def get(self, name):
        if name in self.options:
            return self.options.get(name)
        else:
            Log.error(f"Option {name} not found")
            return None

    def set(self, name, value):
        if name in self.options:
            self.options[name] = value
        else:
            Log.error(f"Option {name} not found")

    def list(self):
        return self.options

class SplitterBase(ABC):
    """
    Abstract base class for audio stem splitters.
    All stem splitter implementations must inherit from this class.
    """
    
    def __init__(self, ):
        self.name = None
        self.type = None
        self.inputs = None
        self.outputs = None
        self.command = CommandController()
        self.data = DataController(self)        
        self.input = InputController(self)
        self.output = OutputController(self)
        self.options = OptionsController(self)

    @property
    @abstractmethod
    def supported_models(self):
        """Returns a dictionary of supported models for this splitter"""
        pass
        
    @abstractmethod
    def load_model(self, model_filename):
        """Load a specific model"""
        pass
        
    @abstractmethod
    def separate(self, audio_object):
        """
        Separate the audio into stems
        Returns: dict of stem_name: audio_data
        """
        pass
        
    @property
    @abstractmethod
    def input_types(self):
        """Returns list of input types this splitter accepts"""
        pass
        
    @property
    @abstractmethod
    def output_stems(self):
        """Returns list of output stem types this splitter produces"""
        pass
        
    def list_supported_model_files(self):
        """Returns a dictionary of supported model files"""
        return self.supported_models