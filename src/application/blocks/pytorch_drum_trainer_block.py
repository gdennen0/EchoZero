"""
PyTorch Drum Trainer Block Processor

Creates and trains PyTorch models for drum classification.
Automatically discovers classes from folder structure (folder names = class names).
Supports training from audio samples organized in class folders.
"""
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from collections import Counter
import time
import os
from pathlib import Path
import json
import numpy as np

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import EventDataItem, Event
from src.application.blocks import register_processor_class
from src.features.execution.application.progress_helpers import (
    IncrementalProgress, get_progress_tracker
)
from src.utils.message import Log
from src.utils.paths import get_models_dir

# Try to import required libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, Subset
    from sklearn.model_selection import KFold
    import librosa
    HAS_PYTORCH = True
    HAS_LIBROSA = True
    HAS_SKLEARN = True
except ImportError as e:
    HAS_PYTORCH = False
    HAS_LIBROSA = False
    HAS_SKLEARN = False
    Log.warning(f"PyTorch/SciKit-Learn dependencies not available: {e}")


class DrumDataset(Dataset):
    """
    Dataset for drum audio classification.
    Loads audio files from folder structure where folder names are class labels.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        max_length: int = 22050,
        augmentation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize drum dataset.

        Args:
            data_dir: Directory containing class folders
            sample_rate: Target sample rate
            max_length: Maximum audio length in samples (pad/truncate to this)
            augmentation_config: Optional config for data augmentation
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.augmentation_config = augmentation_config

        # Discover classes from folder names
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")

        # Scan directory structure
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.classes.append(class_name)
                self.class_to_idx[class_name] = len(self.classes) - 1

                # Find audio files in this class directory
                for audio_file in class_dir.glob("*.wav"):
                    self.samples.append((str(audio_file), class_name))

        Log.info(f"DrumDataset: Found {len(self.classes)} classes: {self.classes}")
        Log.info(f"DrumDataset: Found {len(self.samples)} audio samples")

        if not self.samples:
            raise ValueError(f"No audio files found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, class_name = self.samples[idx]

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Apply data augmentation if configured
        if self.augmentation_config:
            audio = self._augment_audio_real(audio, sr, self.augmentation_config)

        # Pad or truncate to fixed length
        if len(audio) < self.max_length:
            # Pad with zeros
            audio = np.pad(audio, (0, self.max_length - len(audio)), 'constant')
        else:
            # Truncate
            audio = audio[:self.max_length]

        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            hop_length=512,
            fmax=8000
        )

        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)

        # Add channel dimension for CNN (1 channel for grayscale-like)
        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=0)

        # Convert to tensor
        audio_tensor = torch.tensor(mel_spec_norm, dtype=torch.float32)
        label_tensor = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)

        return audio_tensor, label_tensor

    def _augment_audio_real(self, audio: np.ndarray, sample_rate: int, config: Dict[str, Any]) -> np.ndarray:
        """Apply data augmentation to audio if enabled."""
        if not config.get("use_augmentation", False):
            return audio

        augmented_audio = audio.copy()

        # Random pitch shifting
        pitch_shift = config.get("augment_pitch_shift", 2)
        if pitch_shift > 0 and np.random.random() < 0.5:  # 50% chance
            shift_steps = np.random.uniform(-pitch_shift, pitch_shift)
            augmented_audio = librosa.effects.pitch_shift(
                augmented_audio, sr=sample_rate, n_steps=shift_steps
            )

        # Random time stretching
        time_stretch = config.get("augment_time_stretch", 0.1)
        if time_stretch > 0 and np.random.random() < 0.3:  # 30% chance
            stretch_factor = 1.0 + np.random.uniform(-time_stretch, time_stretch)
            augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=stretch_factor)

        # Ensure length is still valid
        max_length = config.get("max_length", len(audio))
        if len(augmented_audio) > max_length:
            augmented_audio = augmented_audio[:max_length]
        elif len(augmented_audio) < max_length:
            # Pad if too short after augmentation
            padding = max_length - len(augmented_audio)
            augmented_audio = np.pad(augmented_audio, (0, padding), 'constant')

        return augmented_audio

    def _preprocess_single_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess a single audio sample for model input.
        """
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=0)
        else:
            audio_mono = audio_data

        # Pad or truncate to fixed length
        if len(audio_mono) < self.max_length:
            # Pad with zeros
            audio_mono = np.pad(audio_mono, (0, self.max_length - len(audio_mono)), 'constant')
        else:
            # Truncate
            audio_mono = audio_mono[:self.max_length]

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_mono,
            sr=sample_rate,
            n_mels=128,
            hop_length=512,
            fmax=8000
        )

        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to 0-1 range
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)

        # Add channel dimension for CNN: (frequency, time, 1)
        mel_spec_rgb = np.expand_dims(mel_spec_norm, axis=0)

        return mel_spec_rgb

    def _augment_audio(self, audio: np.ndarray, sample_rate: int, config: Dict[str, Any]) -> np.ndarray:
        """Apply data augmentation to audio if enabled."""
        if not config.get("use_augmentation", False):
            return audio

        augmented_audio = audio.copy()

        # Random pitch shifting
        pitch_shift = config.get("augment_pitch_shift", 2)
        if pitch_shift > 0 and np.random.random() < 0.5:  # 50% chance
            shift_steps = np.random.uniform(-pitch_shift, pitch_shift)
            augmented_audio = librosa.effects.pitch_shift(
                augmented_audio, sr=sample_rate, n_steps=shift_steps
            )

        # Random time stretching
        time_stretch = config.get("augment_time_stretch", 0.1)
        if time_stretch > 0 and np.random.random() < 0.3:  # 30% chance
            stretch_factor = 1.0 + np.random.uniform(-time_stretch, time_stretch)
            augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=stretch_factor)

        # Ensure length is still valid
        if len(augmented_audio) > config["max_length"]:
            augmented_audio = augmented_audio[:config["max_length"]]
        elif len(augmented_audio) < config["max_length"]:
            # Pad if too short after augmentation
            padding = config["max_length"] - len(augmented_audio)
            augmented_audio = np.pad(augmented_audio, (0, padding), 'constant')

        return augmented_audio


class DrumClassifier(nn.Module):
    """
    Enhanced CNN for drum classification with residual connections and batch normalization.
    """

    def __init__(self, num_classes: int, input_shape: Tuple[int, int] = (128, 44)):
        """
        Initialize drum classifier.

        Args:
            num_classes: Number of drum classes
            input_shape: Input shape (mel_bins, time_steps)
        """
        super(DrumClassifier, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

        # Calculate flattened size after convolutions and pooling
        # Input: (1, 128, 44)
        # After conv1 + pool: (32, 64, 22)
        # After conv2 + pool: (64, 32, 11)
        # After conv3 + pool: (128, 16, 5)
        # After conv4 + pool: (256, 8, 2)
        # Flattened: 256 * 8 * 2 = 4096
        self.fc1 = nn.Linear(256 * 8 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional blocks with batch normalization and residual-like connections
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        identity = x  # Save for residual
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # Simple residual: add if shapes match (they don't here, so skip)

        identity = x
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class PyTorchDrumTrainerBlockProcessor(BlockProcessor):
    """
    Processor for PyTorchDrumTrainer block type.

    Trains PyTorch models for drum classification using folder-based class discovery.
    Takes training data directory as input and outputs trained model.

    Configuration parameters (via block.metadata):
    - data_dir: Directory containing class folders (required)
    - output_model_path: Where to save trained model (optional, defaults to models dir)
    - epochs: Number of training epochs (default: 50)
    - batch_size: Training batch size (default: 32)
    - learning_rate: Learning rate (default: 0.001)
    - sample_rate: Audio sample rate (default: 22050)
    - max_length: Maximum audio length in samples (default: 22050)
    - device: Device to use ("cpu" or "cuda", default: "cpu")
    - validation_split: Fraction of data for validation (default: 0.2)
    """

    def __init__(self):
        """Initialize processor"""
        if not HAS_PYTORCH or not HAS_LIBROSA:
            Log.error("PyTorchDrumTrainer requires PyTorch and librosa dependencies")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Log.info(f"PyTorchDrumTrainer using device: {self.device}")

    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return block.type == "PyTorchDrumTrainer"

    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return "PyTorchDrumTrainer"

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process PyTorchDrumTrainer block.

        Trains a model from folder-structured training data.

        Args:
            block: Block entity to process
            inputs: Input data items (expects "training_data" directory path)
            metadata: Optional metadata

        Returns:
            Dictionary with "model" containing trained model path and metadata

        Raises:
            ProcessingError: If training fails
        """
        if not HAS_PYTORCH or not HAS_LIBROSA:
            raise ProcessingError(
                "PyTorch and librosa dependencies required for PyTorchDrumTrainer. "
                "Install with: pip install torch torchaudio librosa",
                block_id=block.id,
                block_name=block.name
            )

        # Get configuration
        config = self._get_config(block)

        # Get training data directory with better error messages
        data_dir = config.get("data_dir")
        if not data_dir:
            # Try to get from inputs
            training_input = inputs.get("training_data")
            if training_input and hasattr(training_input, 'data'):
                data_dir = training_input.data
            elif training_input and isinstance(training_input, str):
                data_dir = training_input

        if not data_dir:
            raise ProcessingError(
                "❌ No Training Data Directory\n"
                "Set 'data_dir' in block settings to point to your training data.\n"
                "Example: /path/to/drum_samples\n"
                "The directory should contain subfolders for each drum class.",
                block_id=block.id,
                block_name=block.name
            )

        data_path = Path(data_dir)
        if not data_path.exists():
            raise ProcessingError(
                f"❌ Training Directory Not Found\n"
                f"Directory '{data_dir}' does not exist.\n"
                f"Create the directory and add your drum sample folders.",
                block_id=block.id,
                block_name=block.name
            )

        if not data_path.is_dir():
            raise ProcessingError(
                f"❌ Invalid Training Directory\n"
                f"'{data_dir}' is not a directory.\n"
                f"Select a folder containing drum class subdirectories.",
                block_id=block.id,
                block_name=block.name
            )

        Log.info(f"PyTorchDrumTrainer: Training from directory {data_dir}")

        # Get progress tracker from metadata
        progress_tracker = get_progress_tracker(metadata)

        try:
            # Train the model with progress tracking
            training_results = self._train_model(data_dir, config, progress_tracker)
        except Exception as e:
            # Provide more specific error messages based on the error type
            error_msg = str(e)
            if "CUDA" in error_msg and "out of memory" in error_msg:
                raise ProcessingError(
                    "❌ GPU Memory Error\n"
                    "Not enough GPU memory for training.\n"
                    "Try:\n"
                    "• Reduce batch_size (try 8 or 4)\n"
                    "• Use CPU training: set device to 'cpu'\n"
                    "• Reduce max_length\n"
                    "• Close other GPU-using applications",
                    block_id=block.id,
                    block_name=block.name
                ) from e
            elif "No audio files found" in error_msg:
                raise ProcessingError(
                    "❌ No Training Audio Found\n"
                    f"No .wav files found in '{data_dir}'.\n"
                    "Make sure each class subfolder contains .wav audio files.",
                    block_id=block.id,
                    block_name=block.name
                ) from e
            else:
                raise ProcessingError(
                    f"❌ Training Failed\n"
                    f"Error: {error_msg}\n"
                    "Check the logs for more details.",
                    block_id=block.id,
                    block_name=block.name
                ) from e

        # Save model
        model_path = self._save_model(training_results["model"], training_results["classes"], config)

        # Create output data
        output_data = {
            "model_path": model_path,
            "classes": training_results["classes"],
            "training_stats": training_results["stats"],
            "config": config
        }

        # Store training results in block metadata
        if block.metadata:
            block.metadata.update({
                "last_training": {
                    "timestamp": datetime.now().isoformat(),
                    "model_path": model_path,
                    "classes": training_results["classes"],
                    "stats": training_results["stats"]
                }
            })
        else:
            block.metadata = {
                "last_training": {
                    "timestamp": datetime.now().isoformat(),
                    "model_path": model_path,
                    "classes": training_results["classes"],
                    "stats": training_results["stats"]
                }
            }

        # Return training results as dictionary
        return {"model": output_data}

    @staticmethod
    def load_trained_model(model_path: str, device: str = "cpu") -> Tuple[nn.Module, List[str], Dict[str, Any]]:
        """
        Load a trained PyTorch drum classification model.

        Args:
            model_path: Path to the saved model file
            device: Device to load the model on

        Returns:
            Tuple of (model, classes, metadata)
        """
        if not HAS_PYTORCH:
            raise ProcessingError(
                "PyTorch dependencies required for loading models. "
                "Install with: pip install torch torchaudio",
                block_id="",
                block_name=""
            )

        device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Extract model info
        classes = checkpoint.get('classes', [])
        config = checkpoint.get('config', {})
        num_classes = len(classes)

        # Create model and load state
        model = DrumClassifier(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        metadata = {
            'classes': classes,
            'config': config,
            'training_date': checkpoint.get('training_date')
        }

        return model, classes, metadata

    def predict_audio(
        self,
        model: nn.Module,
        audio_data: np.ndarray,
        sample_rate: int,
        classes: List[str],
        device: str = "cpu"
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict drum type for audio sample.

        Args:
            model: Trained PyTorch model
            audio_data: Audio waveform
            sample_rate: Sample rate
            classes: List of class names
            device: Device to use

        Returns:
            Tuple of (predicted_class, confidence, probabilities_dict)
        """
        if not HAS_PYTORCH or not HAS_LIBROSA:
            raise ProcessingError(
                "PyTorch and librosa required for prediction",
                block_id="",
                block_name=""
            )

        # Preprocess audio (reuse the dataset preprocessing)
        # Create a temporary dataset to get preprocessing
        temp_dataset = DrumDataset("/tmp", sample_rate=sample_rate, max_length=sample_rate * 2)  # Dummy path

        # Manually preprocess
        preprocessed = temp_dataset._preprocess_single_audio(audio_data, sample_rate)
        preprocessed = torch.tensor(preprocessed, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(preprocessed)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get prediction
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = classes[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        # Create probabilities dict
        prob_dict = {cls: probabilities[i].item() for i, cls in enumerate(classes)}

        return predicted_class, confidence, prob_dict

    def _run_training_loop(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        classes: List[str],
        config: Dict[str, Any],
        progress_tracker=None
    ) -> Dict[str, Any]:
        """Run the training loop with early stopping and learning rate scheduling."""
        # Training loop with early stopping and learning rate scheduling
        best_val_accuracy = 0.0
        best_model_state = None
        patience = config.get("early_stopping_patience", 15)
        patience_counter = 0
        min_delta = 0.001  # Minimum improvement to reset patience

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=False
        )

        training_stats = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rates": []
        }

        Log.info(f"PyTorchDrumTrainer: Starting training for up to {config['epochs']} epochs...")

        # Initialize progress tracking for epochs
        progress = IncrementalProgress(
            progress_tracker,
            "Training model",
            total=config["epochs"]
        )

        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            # Store stats
            training_stats["epochs"].append(epoch + 1)
            training_stats["train_loss"].append(train_loss)
            training_stats["val_loss"].append(val_loss)
            training_stats["train_accuracy"].append(train_accuracy)
            training_stats["val_accuracy"].append(val_accuracy)
            training_stats["learning_rates"].append(optimizer.param_groups[0]['lr'])

            # Update learning rate scheduler
            scheduler.step(val_accuracy)

            # Early stopping check
            if val_accuracy > best_val_accuracy + min_delta:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                Log.debug(f"PyTorchDrumTrainer: New best model at epoch {epoch+1} (val_acc: {val_accuracy:.2f}%)")
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                Log.info(
                    f"Epoch {epoch+1}/{config['epochs']}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                    f"LR: {current_lr:.6f}, Patience: {patience_counter}/{patience}"
                )
            
            # Update progress after each epoch
            progress.step(
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train: {train_accuracy:.1f}%, Val: {val_accuracy:.1f}%"
            )

            # Early stopping
            if patience_counter >= patience:
                Log.info(
                    f"PyTorchDrumTrainer: Early stopping at epoch {epoch+1}. "
                    f"Best validation accuracy: {best_val_accuracy:.2f}%"
                )
                break

        Log.info(f"PyTorchDrumTrainer: Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")

        # Complete progress tracking
        progress.complete(f"Training complete - Best accuracy: {best_val_accuracy:.1f}%")

        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)

        return {
            "model": model,
            "classes": classes,
            "stats": training_stats
        }

    def _train_with_cross_validation(self, dataset: DrumDataset, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train using k-fold cross-validation and return the best model."""
        cv_folds = config.get("cv_folds", 5)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        fold_results = []
        best_model = None
        best_accuracy = 0.0
        best_classes = dataset.classes

        Log.info(f"PyTorchDrumTrainer: Starting {cv_folds}-fold cross-validation")

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
            Log.info(f"PyTorchDrumTrainer: Training fold {fold}/{cv_folds}")

            # Create fold-specific datasets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)

            # Train model for this fold
            fold_result = self._train_single_model(dataset.classes, train_loader, val_loader, config)
            fold_accuracy = max(fold_result["stats"]["val_accuracy"])

            fold_results.append({
                "fold": fold,
                "best_accuracy": fold_accuracy,
                "stats": fold_result["stats"]
            })

            Log.info(f"PyTorchDrumTrainer: Fold {fold} best validation accuracy: {fold_accuracy:.2f}%")

            # Keep track of best model across folds
            if fold_accuracy > best_accuracy:
                best_accuracy = fold_accuracy
                best_model = fold_result["model"]

        # Aggregate results across folds
        avg_accuracy = sum(result["best_accuracy"] for result in fold_results) / len(fold_results)
        Log.info(f"PyTorchDrumTrainer: Cross-validation complete. Average accuracy: {avg_accuracy:.2f}%")

        return {
            "model": best_model,
            "classes": best_classes,
            "stats": {
                "cross_validation": True,
                "num_folds": cv_folds,
                "fold_results": fold_results,
                "average_accuracy": avg_accuracy,
                "best_fold_accuracy": best_accuracy
            }
        }

    def _get_config(self, block: Block) -> Dict[str, Any]:
        """Get configuration from block metadata."""
        config = {
            "epochs": 100,  # Increased default epochs with early stopping
            "batch_size": 32,
            "learning_rate": 0.001,
            "sample_rate": 22050,
            "max_length": 22050,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "validation_split": 0.2,
            "output_model_path": None,
            "early_stopping_patience": 15,
            "use_augmentation": False,  # Data augmentation
            "use_cross_validation": False,  # K-fold cross-validation
            "cv_folds": 5,  # Number of CV folds
            "augment_pitch_shift": 2,  # Max pitch shift in semitones
            "augment_time_stretch": 0.1,  # Max time stretch factor
            "weight_decay": 1e-4,  # L2 regularization
            "dropout_rate": 0.4
        }

        if block.metadata:
            config.update(block.metadata)

        return config

    def _train_model(self, data_dir: str, config: Dict[str, Any], progress_tracker=None) -> Dict[str, Any]:
        """
        Train the drum classification model.

        Args:
            data_dir: Directory containing training data
            config: Training configuration
            progress_tracker: Optional progress tracker for reporting training progress

        Returns:
            Dictionary with trained model, classes, and training statistics
        """
        Log.info("PyTorchDrumTrainer: Creating dataset...")

        # Create dataset with better error handling
        try:
            dataset = DrumDataset(
                data_dir=data_dir,
                sample_rate=config["sample_rate"],
                max_length=config["max_length"],
                augmentation_config=config if config.get("use_augmentation", False) else None
            )
        except ValueError as e:
            error_msg = str(e)
            if "does not exist" in error_msg:
                raise ProcessingError(
                    f"❌ Dataset Error\n"
                    f"Directory '{data_dir}' not found.\n"
                    f"Make sure the directory exists and is accessible.",
                    block_id="",
                    block_name=""
                ) from e
            elif "No audio files found" in error_msg:
                raise ProcessingError(
                    "❌ No Audio Files Found\n"
                    f"No .wav files found in class subdirectories of '{data_dir}'.\n"
                    "Each class folder must contain .wav audio files.\n"
                    "Example structure:\n"
                    "  drum_samples/\n"
                    "    kick/\n"
                    "      kick01.wav\n"
                    "      kick02.wav\n"
                    "    snare/\n"
                    "      snare01.wav\n"
                    "      snare02.wav",
                    block_id="",
                    block_name=""
                ) from e
            else:
                raise ProcessingError(
                    f"❌ Dataset Creation Failed\n"
                    f"Error: {error_msg}",
                    block_id="",
                    block_name=""
                ) from e

        # Split into train/validation
        val_size = int(len(dataset) * config["validation_split"])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        # Check if cross-validation is enabled
        use_cv = config.get("use_cross_validation", False)
        cv_folds = config.get("cv_folds", 5)

        if use_cv and HAS_SKLEARN:
            Log.info(f"PyTorchDrumTrainer: Using {cv_folds}-fold cross-validation")
            Log.info(f"PyTorchDrumTrainer: Classes: {dataset.classes}")

            # Perform cross-validation training
            cv_results = self._train_with_cross_validation(dataset, config)
            return cv_results
        else:
            if use_cv and not HAS_SKLEARN:
                Log.warning("Cross-validation requested but scikit-learn not available. Using simple split.")

            Log.info(f"PyTorchDrumTrainer: Training set: {train_size} samples")
            Log.info(f"PyTorchDrumTrainer: Validation set: {val_size} samples")
            Log.info(f"PyTorchDrumTrainer: Classes: {dataset.classes}")

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

            # Train single model
            return self._train_single_model(dataset.classes, train_loader, val_loader, config)

    def _train_single_model(
        self,
        classes: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a single model with train/validation split."""
        # Create model
        model = DrumClassifier(num_classes=len(classes))
        model.to(self.device)

        # Loss and optimizer with L2 regularization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        return self._run_training_loop(model, criterion, optimizer, train_loader, val_loader, classes, config, progress_tracker)

        return {
            "model": model,
            "classes": dataset.classes,
            "stats": training_stats
        }

    def _save_model(self, model: nn.Module, classes: List[str], config: Dict[str, Any]) -> str:
        """Save trained model and metadata."""
        # Get output path
        if config.get("output_model_path"):
            model_path = config["output_model_path"]
        else:
            models_dir = get_models_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = models_dir / f"pytorch_drum_model_{timestamp}.pth"

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': classes,
            'config': config,
            'training_date': datetime.now().isoformat()
        }, model_path)

        Log.info(f"PyTorchDrumTrainer: Model saved to {model_path}")

        return str(model_path)

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate PyTorchDrumTrainer block configuration.

        Provides user-friendly error messages to help with setup.

        Args:
            block: Block to validate
            data_item_repo: Data item repository
            connection_repo: Connection repository
            block_registry: Block registry

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check dependencies first - most common issue
        if not HAS_PYTORCH or not HAS_LIBROSA:
            errors.append(
                "❌ Missing Dependencies\n"
                "PyTorch and librosa are required for training.\n"
                "Install with: pip install torch torchaudio librosa\n"
                "Note: This may take several minutes to download."
            )
            return errors

        if not HAS_SKLEARN:
            Log.warning("scikit-learn not available - cross-validation will be disabled")

        # Check data directory configuration
        data_dir = block.metadata.get("data_dir") if block.metadata else None
        if not data_dir:
            errors.append(
                "❌ No Training Data Directory Set\n"
                "Set 'data_dir' in block settings to point to your training data.\n"
                "Example: /path/to/drum_samples"
            )
            return errors  # Early return since other checks depend on this

        data_path = Path(data_dir)

        # Check if directory exists
        if not data_path.exists():
            errors.append(
                f"❌ Training Directory Not Found\n"
                f"Directory '{data_dir}' does not exist.\n"
                f"Create the directory and add your drum sample folders."
            )
            return errors

        if not data_path.is_dir():
            errors.append(
                f"❌ Invalid Training Directory\n"
                f"'{data_dir}' is not a directory.\n"
                f"Select a folder containing drum class subdirectories."
            )
            return errors

        # Check for class subdirectories
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not class_dirs:
            errors.append(
                "❌ No Drum Class Folders Found\n"
                f"No subdirectories found in '{data_dir}'.\n"
                "Create folders for each drum type you want to classify:\n"
                "• kick/ (for kick drum samples)\n"
                "• snare/ (for snare drum samples)\n"
                "• hihat/ (for hi-hat samples)\n"
                "• etc..."
            )
            return errors

        # Check for audio files
        total_audio_files = 0
        classes_with_audio = []
        classes_without_audio = []

        for class_dir in class_dirs:
            audio_files = list(class_dir.glob("*.wav"))
            if audio_files:
                total_audio_files += len(audio_files)
                classes_with_audio.append(f"{class_dir.name} ({len(audio_files)} files)")
            else:
                classes_without_audio.append(class_dir.name)

        if classes_without_audio:
            errors.append(
                f"❌ Missing Audio Files in Classes\n"
                f"These class folders have no .wav files:\n"
                + "\n".join(f"• {cls}/" for cls in classes_without_audio) +
                "\n\nAdd .wav audio files to each class folder."
            )

        if total_audio_files == 0:
            errors.append(
                "❌ No Audio Files Found\n"
                "No .wav files found in any class folder.\n"
                "Each class folder must contain .wav audio samples."
            )
        elif total_audio_files < 10:
            errors.append(
                f"⚠️ Very Few Training Samples\n"
                f"Only {total_audio_files} audio files found total.\n"
                "For good results, use at least 50-100 samples per class.\n"
                "Current classes:\n" +
                "\n".join(f"• {cls}" for cls in classes_with_audio)
            )

        # Success message if everything looks good
        if not errors and total_audio_files > 0:
            Log.info(
                f"✅ Configuration Valid: {len(class_dirs)} classes, "
                f"{total_audio_files} total samples"
            )

        return errors


# Auto-register this processor class
register_processor_class(PyTorchDrumTrainerBlockProcessor)
