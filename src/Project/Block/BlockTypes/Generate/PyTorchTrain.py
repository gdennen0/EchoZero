import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.Project.Block.block import Block
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput

class PyTorchTrain(Block):
    """
    A block that trains a simple PyTorch model to detect specific percussion types
    (e.g., kick, snare, hi-hat, etc.) based on manual classifications.
    """
    name = "PyTorchTrain"
    type = "PyTorchTrain"

    def __init__(self):
        super().__init__()
        self.name = "PyTorchTrain"
        self.type = "PyTorchTrain"
        
        self.input.add_type(EventInput)
        self.input.add("EventInput")
        
        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Simple two-layer network; adjust as needed.
        self.model = nn.Sequential(
            nn.Linear(44100, 32),  # example dimension if each audio slice is 1 second at 44.1kHz
            nn.ReLU(),
            nn.Linear(32, 3)      # e.g., 3-class classification (kick, snare, other)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.label_map = {"kick": 0, "snare": 1, "other": 2}  # example set of classes

        # Add commands for flexibility (optional)
        self.command.add("train_model", self.train_model)
        self.command.add("save_model", self.save_model)

    def process(self, event_data_list):
        """
        By default, just pass the data through (no modification).
        Usually training is triggered by invoking the "train_model" command separately.
        """
        return event_data_list

    def train_model(self):
        """
        Loads the labeled events from self.data,
        extracts waveforms and labels, and runs a small training loop.
        """
        labeled_waveforms = []
        labels = []

        # Gather all labeled event items from the input
        for event_data in self.data:
            if not isinstance(event_data, EventData):
                continue
            for item in event_data.items:
                # Only use items having a classification from the manual block
                if item.classification:
                    # Convert classification string to a numerical label (e.g. "kick" -> 0)
                    label = self.label_map.get(item.classification, self.label_map["other"])
                    audio_samples = item.data.data  # raw waveform (numpy array)
                    # Ensure we have a fixed-size input; if not, pad or trim
                    audio_samples = self._prepare_audio_samples(audio_samples)
                    labeled_waveforms.append(audio_samples)
                    labels.append(label)

        if not labeled_waveforms:
            print("No labeled data found. Cannot train.")
            return

        # Convert to tensors
        x_data = torch.tensor(np.vstack(labeled_waveforms), dtype=torch.float)
        y_data = torch.tensor(labels, dtype=torch.long)

        # Basic training loop
        epochs = 5
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x_data)
            loss = self.criterion(outputs, y_data)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        print("Training complete.")

    def save_model(self, path="pytorch_percussion_model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def _prepare_audio_samples(self, audio_samples, desired_length=44100):
        """
        Pad or trim the waveform to a fixed length for the model.
        """
        length = len(audio_samples)
        if length > desired_length:
            return audio_samples[:desired_length][np.newaxis, :]  # shape: (1, 44100)
        elif length < desired_length:
            pad_width = desired_length - length
            return np.pad(audio_samples, (0, pad_width))[np.newaxis, :]
        else:
            return audio_samples[np.newaxis, :]