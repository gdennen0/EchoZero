import torch
import torch.nn as nn
import numpy as np
from src.Project.Block.block import Block
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Utils.message import Log
import os
from dataclasses import dataclass, field

@dataclass
class Config:
    input_size: int = 44100  # Must match the training configuration
    hidden_layers: list = field(default_factory=lambda: [256, 128, 64])
    output_size: int = None  # To be set based on label_map
    save_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "models"))
    model_name: str = "EZ_model.pt"  # Ensure it points to the best model

class PyTorchClassify(Block):
    """
    Loads a trained PyTorch model to classify audio events as specific percussion types.
    """
    name = "PyTorchClassify"
    type = "PyTorchClassify"

    def __init__(self):
        super().__init__()
        self.name = "PyTorchClassify"
        self.type = "PyTorchClassify"

        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.command.add("list_classifications", self.list_classifications)

        self.config = Config()
        self.model = None
        self.label_map_reverse = {}

        self.load_model()

    def build_model(self):
        """
        Builds the model architecture based on the configuration.
        """
        layers = []
        input_dim = self.config.input_size
        for hidden_dim in self.config.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.config.output_size))
        return nn.Sequential(*layers)

    def load_model(self):
        """
        Loads the model architecture and weights from the checkpoint.
        """

        Log.info(f"Loading model from {self.config.save_dir} with name {self.config.model_name}")
        checkpoint_path = os.path.join(self.config.save_dir, self.config.model_name)
        if not os.path.exists(checkpoint_path):
            Log.error(f"Checkpoint not found at {checkpoint_path}")
            return

        saved_data = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Log.info(f"Model saved data: {saved_data}")
        # Load label_map and set output_size
        label_map = saved_data.get("label_map")
        if not label_map:
            Log.error("label_map not found in checkpoint.")
            return
        self.label_map_reverse = {idx: label for label, idx in label_map.items()}
        self.config.output_size = len(label_map)

        # Build and load model
        self.model = self.build_model()
        self.model.load_state_dict(saved_data["state_dict"])
        self.model.eval()
        Log.info(f"Model loaded successfully from {checkpoint_path}")

    def process(self, event_data_list):
        """
        For each incoming EventData, classify each EventItem using the trained model.
        Set the 'classification' on the EventItem accordingly and return updated EventData.
        """
        if not self.model:
            Log.error("Model not loaded. Cannot perform classification.")
            return event_data_list

        updated_event_data_list = []

        for event_data in event_data_list:
            if not isinstance(event_data, EventData):
                updated_event_data_list.append(event_data)
                continue

            for item in event_data.items:
                # Prepare waveform
                audio_data = item.data
                if not audio_data or audio_data.data is None:
                    continue

                waveform = audio_data.data
                prepared = self._prepare_audio_samples(waveform)
                x_tensor = torch.tensor(prepared, dtype=torch.float)

                # Model inference
                with torch.no_grad():
                    outputs = self.model(x_tensor)
                    predicted_class = torch.argmax(outputs, dim=1).item()

                # Update the classification
                item.classification = self.label_map_reverse.get(predicted_class, "unclassified")
            
            updated_event_data_list.append(event_data)

        return updated_event_data_list

    def _prepare_audio_samples(self, audio_samples, desired_length=44100):
        """
        Pad or trim the waveform to match the model's expected input length.
        """
        length = len(audio_samples)
        if length > desired_length:
            return audio_samples[:desired_length][np.newaxis, :]
        elif length < desired_length:
            pad_width = desired_length - length
            return np.pad(audio_samples, (0, pad_width)).reshape(1, -1)
        else:
            return audio_samples.reshape(1, -1)

    def list_classifications(self):
        Log.info(f"Listing classifications for block '{self.name}'")
        for event_data_item in self.data.get_all():
            Log.info(f"EventData: {event_data_item.name}")
            for item in event_data_item.get_all():
                Log.info(f"Classification: {item.name}, {item.classification}")

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }  
    
    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        
        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push loaded data to output 
        self.output.push_all(self.data.get_all())    