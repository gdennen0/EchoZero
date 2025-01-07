import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from dataclasses import dataclass, field
import os

from src.Project.Block.block import Block
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Utils.message import Log

@dataclass
class Config:
    input_size: int = 44100  # Example: 1 second at 44.1kHz
    hidden_layers: list = field(default_factory=lambda: [256, 128, 64])
    output_size: int = None
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    patience: int = 15
    validation_split: float = 0.3
    save_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "models"))
    model_name: str = "EZ_model"

class PyTorchTrain(Block):
    """
    A block that trains a PyTorch model to detect specific percussion types
    based on manual classifications.
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
        
        self.config = Config()
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.label_map = None
        self.optimizer = None

        self.command.add("train_model", self.train_model)
        self.command.add("save_model", self.save_model)

    def load_model(self):
        self.config.output_size = len(self.label_map)
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def build_model(self):
        layers = []
        input_dim = self.config.input_size
        for hidden_dim in self.config.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.config.output_size))
        return nn.Sequential(*layers)

    def process(self, event_data_list):
        self.prepare_label_map(event_data_list)
        self.load_model()

        desired_length = self.config.input_size

        for event_data in event_data_list:
            for event_item in event_data.items:

                length = len(event_item.data.get())
                if length > self.config.input_size:
                    event_item.data.set(event_item.data.get()[:self.config.input_size][np.newaxis, :])  # shape: (1, 44100)
                elif length < self.config.input_size:
                    pad_width = self.config.input_size - length
                    event_item.data.set(np.pad(event_item.data.get(), (0, pad_width))[np.newaxis, :])
                else:
                    event_item.data.set(event_item.data.get()[np.newaxis, :])

        return event_data_list

    def prepare_label_map(self, event_data_list):
        classifications = {item.classification for event_data in event_data_list
                           if isinstance(event_data, EventData)
                           for item in event_data.items
                           if item.classification
                           and item.classification not in ["unclassified", "unknown", "other"]}
        self.label_map = {label: idx for idx, label in enumerate(sorted(classifications))}
        Log.info(f"Generated label map: {self.label_map}")

    def train_model(self):
        """
        Loads the labeled events from self.data,
        extracts waveforms and labels, and runs a training loop.
        """

        if not self.model:
            Log.error("Model not loaded. Cannot train. Please reload PyTorchTrain block.")
            return

        try:
            labeled_waveforms, labels = self.extract_training_data()
            if not labeled_waveforms:
                Log.warning("No labeled data found. Cannot train.")
                return

            dataset = self.create_datasets(labeled_waveforms, labels)
            self.run_training_loop(dataset)
            Log.info("Training complete.")
        except Exception as e:
            Log.error(f"An error occurred during training: {e}")
            raise

    def extract_training_data(self):
        labeled_waveforms = []
        labels = []

        for event_data in self.data.get_all():
            if not isinstance(event_data, EventData):
                continue
            for item in event_data.items:
                if item.classification:
                    if item.classification in ["unclassified", "unknown", "other"]:
                        continue
                    if item.classification not in self.label_map:
                        continue

                    label = self.label_map.get(item.classification)
                    labeled_waveforms.append(item.data.get())
                    labels.append(label)

        return labeled_waveforms, labels

    def create_datasets(self, labeled_waveforms, labels):
        x_data = torch.tensor(np.vstack(labeled_waveforms), dtype=torch.float)
        y_data = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(x_data, y_data)

    def run_training_loop(self, dataset):
        train_size = int((1 - self.config.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

        best_val_loss = float('inf')
        trigger_times = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_loss = self.train_one_epoch(train_loader)

            val_loss = self.validate(val_loader)
            scheduler.step(val_loss)

            Log.info(f"Epoch {epoch}/{self.config.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # self.save_checkpoint()
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= self.config.patience:
                    Log.warning("Early stopping triggered")
                    break

    def train_one_epoch(self, loader):
        running_loss = 0.0
        for inputs, targets in loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(loader)

    def save_checkpoint(self):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'label_map': self.label_map
        }
        os.makedirs(self.config.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.save_dir, f"{self.config.model_name}.pt")
        torch.save(checkpoint, checkpoint_path)
        Log.info(f"Checkpoint saved at {checkpoint_path}")

    def save_model(self):
        save_data = {
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'label_map': self.label_map
        }
        os.makedirs(self.config.save_dir, exist_ok=True)
        model_path = os.path.join(self.config.save_dir, f"{self.config.model_name}.pt")
        torch.save(save_data, model_path)
        Log.info(f"Model and label map saved to {model_path}")

    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "label_map": self.label_map,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # Load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_label_map(block_metadata.get("label_map"))
        
        # Load sub-components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # Push loaded data to output 
        self.output.push_all(self.data.get_all())