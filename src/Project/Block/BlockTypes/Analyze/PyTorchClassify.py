import torch
import torch.nn as nn
import numpy as np
from src.Project.Block.block import Block
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput

class PyTorchClassify(Block):
    """
    Loads a trained PyTorch model to classify audio events as specific percussion types.
    """
    name = "PyTorchClassify "
    type = "PyTorchClassify"

    def __init__(self, model_path="pytorch_percussion_model.pt"):
        super().__init__()
        self.name = "PyTorchClassify"
        self.type = "PyTorchClassify"

        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Architecture should match what was used for training.
        self.model = nn.Sequential(
            nn.Linear(44100, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.label_map_reverse = {0: "kick", 1: "snare", 2: "other"}

    def process(self, event_data_list):
        """
        For each incoming EventData, classify each EventItem using the trained model.
        Set the 'classification' on the EventItem accordingly and return updated EventData.
        """
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
                item.classification = self.label_map_reverse[predicted_class]

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
            return np.pad(audio_samples, (0, pad_width))[np.newaxis, :]
        else:
            return audio_samples[np.newaxis, :]