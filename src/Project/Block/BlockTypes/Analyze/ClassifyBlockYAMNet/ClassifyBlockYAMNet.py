import os
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
import torchaudio
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Data.Types.event_data import EventData
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Utils.message import Log
import pandas as pd
import librosa
import json

class ClassifyBlockYAMNet(Block):
    """
    This block uses YAMNet to classify audio events, focusing solely on percussion.
    """
    name = "ClassifyYAMNet"
    type = "ClassifyYAMNet"
    
    DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
    DEFAULT_SAMPLE_RATE = 16000  # YAMNet expects 16kHz
    DEFAULT_THRESHOLD = 0        # Adjusted threshold for YAMNet probabilities

    CLASS_NAMES_FILE = os.path.join(os.getcwd(), "data", "yamnet_class_map.csv")

    def __init__(self):
        super().__init__()
        self.name = "ClassifyYAMNet"
        self.type = "ClassifyYAMNet"

        self.input.add_type(EventInput)
        self.input.add("EventInput")
        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.output_dir = self.DEFAULT_OUTPUT_DIR
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.threshold = self.DEFAULT_THRESHOLD

        # Load class names from CSV
        self.classes = pd.read_csv(os.path.join(os.getcwd(), "data", "yamnet_class_map.csv"), header=None)
        self.class_names = self.classes[2].tolist()
        
        self.allowed_percussion_classes = self.class_names
        # Only allow percussion-related categories
        percussion_classes = [
            "Percussion", "Drum kit", "Drum machine", "Drum", "Snare drum",
            "Rimshot", "Drum roll", "Bass drum", "Timpani", "Tabla",
            "Cymbal", "Hi-hat", "Wood block", "Tambourine", "Rattle (instrument)",
            "Maraca", "Gong", "Tubular bells", "Mallet percussion", "Marimba, xylophone",
            "Glockenspiel", "Vibraphone", "Steelpan"
        ]

        # Load the YAMNet model
        self.model = None  # DONT LOAD ON INIT, LOAD ON RELOAD 
        Log.info("Classify initialized with YAMNet model.")

        # Register commands
        self.command.add("set_output_dir", self.set_output_dir)
        self.command.add("set_threshold", self.set_threshold)
    
    def load_model(self):
        """
        Load the YAMNet model from a local path, so you don't need to re-download it each time.
        """
        Log.info("Loading YAMNet model from a local directory.")
        # Make sure the local_model_path contains saved_model.pb, variables/, etc.
        local_model_path = os.path.join(os.getcwd(), "models", "yamnet", "1")
        if self.model is None:
            self.model = hub.load(local_model_path)
        else:
            Log.warning("YAMNet model already loaded")


    def process(self, input_data):
        """
        Process each EventData object and classify using YAMNet, focusing on percussive classes.
        """
        Log.info("Starting YAMNet classification...")
        os.makedirs(self.output_dir, exist_ok=True)

        if self.model is None:
            self.load_model()

        for event_data in input_data:
            for event in event_data.items:
                # Convert/Resample audio to YAMNet's expected sample rate
                audio_tensor = self.prepare_audio_tensor(event)
                
                # Run YAMNet inference
                try:
                    scores, _, _ = self.model(audio_tensor)
                except Exception as exc:
                    Log.error(f"Error during YAMNet inference: {exc}")
                    raise

                # Identify the best valid percussion class
                best_score, best_class = self.identify_percussion_class(scores)
                event.set_confidence(best_score)
                event.set_classification(best_class)

                Log.info(f"{event.name} classified as '{best_class}' with confidence {best_score}%")

        return input_data


    def prepare_audio_tensor(self, event):
        """
        Convert event audio to a single-channel float32 TensorFlow tensor. Resample if needed.
        """
        if event.data.sample_rate != self.sample_rate:
            return self.resampler(event)
        elif isinstance(event.data, torch.Tensor):
            return tf.convert_to_tensor(event.data.numpy(), dtype=tf.float32)
        return tf.convert_to_tensor(event.data, dtype=tf.float32)


    def identify_percussion_class(self, scores):
        """
        Determine the best-scoring percussion class above the threshold.
        """
        mean_scores = np.mean(scores, axis=0)
        best_score, best_class = 0.0, "Unclassified"

        for index, score_value in enumerate(mean_scores):
            class_label = self.class_names[index + 1]
            if (score_value >= self.threshold and 
                class_label in self.allowed_percussion_classes and 
                score_value > best_score):
                best_score = score_value
                best_class = class_label

        return best_score, best_class

    def resampler(self, event):
        """
        Resample audio data to the target sample rate and return as a TensorFlow tensor.

        Args:
            event: A single event object containing audio data.

        Returns:
            tf.Tensor: Resampled audio as float32 tensor.
        """
        try:
            original_audio = event.data.data.copy()

            if original_audio.size == 0:
                Log.warning(f"Empty audio array for event {event.name}")
                return tf.zeros([0], dtype=tf.float32)

            # Ensure mono
            if original_audio.ndim > 1:
                original_audio = np.mean(original_audio, axis=-1)

            # Resample using librosa
            resampled_audio = librosa.resample(
                y=original_audio,
                orig_sr=event.data.sample_rate,
                target_sr=self.sample_rate
            ).astype(np.float32)

            # Normalize if needed
            max_val = np.max(np.abs(resampled_audio))
            if resampled_audio.size > 0 and max_val > 1.0:
                resampled_audio /= max_val

            return tf.convert_to_tensor(resampled_audio, dtype=tf.float32)

        except Exception as exc:
            Log.error(f"Error during resampling: {exc}")
            raise

    def set_output_dir(self, output_dir):
        """
        Set the directory where outputs will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        Log.info(f"Output directory set to {self.output_dir}")

    def set_threshold(self, threshold):
        """
        Set the classification confidence threshold.
        """
        if not (0.0 <= threshold <= 1.0):
            Log.error("Threshold must be between 0.0 and 1.0.")
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        Log.info(f"Threshold set to {self.threshold}")

    def set_sample_rate(self, sample_rate):
        """
        Set the sample rate for the block.
        """
        self.sample_rate = sample_rate
        Log.info(f"Sample rate set to {self.sample_rate}")

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "output_dir": self.output_dir,
            "sample_rate": self.sample_rate,
            "threshold": self.threshold,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        # just saves the data
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)  

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_output_dir(block_metadata.get("output_dir", self.DEFAULT_OUTPUT_DIR))
        self.set_sample_rate(block_metadata.get("sample_rate", self.DEFAULT_SAMPLE_RATE))
        self.set_threshold(block_metadata.get("threshold", self.DEFAULT_THRESHOLD))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())

