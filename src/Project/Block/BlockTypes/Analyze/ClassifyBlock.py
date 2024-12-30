import os
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
import torchaudio
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Utils.message import Log
import pandas as pd
import librosa

class ClassifyBlock(Block):
    name = "Classify"
    
    DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
    DEFAULT_SAMPLE_RATE = 16000  # YAMNet expects 16kHz
    DEFAULT_THRESHOLD = 0  # Adjusted threshold for YAMNet probabilities

    CLASS_NAMES_FILE = os.path.join(os.getcwd(), "data", "yamnet_class_map.csv")

    def __init__(self):
        super().__init__()
        self.name = "Classify"
        self.type = "Classify"

        self.input.add_type(EventInput)
        self.input.add("EventInput")
        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.output_dir = self.DEFAULT_OUTPUT_DIR
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.threshold = self.DEFAULT_THRESHOLD
        self.classes = pd.read_csv(os.path.join(os.getcwd(), "data", "yamnet_class_map.csv"), header=None)
        self.class_names = self.classes[2].tolist()

        self.accepted_classes = [
            # Percussion classes
            "Percussion", "Drum kit", "Drum machine", "Drum", "Snare drum",
            "Rimshot", "Drum roll", "Bass drum", "Timpani", "Tabla",
            "Cymbal", "Hi-hat", "Wood block", "Tambourine", "Rattle (instrument)",
            "Maraca", "Gong", "Tubular bells", "Mallet percussion", "Marimba, xylophone",
            "Glockenspiel", "Vibraphone", "Steelpan",
            # Music classes
            "Music", "Musical instrument", "Pop music", "Hip hop music", "Rock music",
            "Heavy metal", "Punk rock", "Grunge", "Progressive rock", "Rock and roll",
            "Psychedelic rock", "Rhythm and blues", "Soul music", "Reggae", "Country",
            "Swing music", "Bluegrass", "Funk", "Folk music", "Middle Eastern music",
            "Jazz", "Disco", "Classical music", "Opera", "Electronic music",
            "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
            "Electronic dance music", "Ambient music", "Trance music", 
            "Music of Latin America", "Salsa music", "Flamenco", "Blues",
            "Music for children", "New-age music", "Vocal music", "A capella",
            "Music of Africa", "Afrobeat", "Christian music", "Gospel music",
            "Music of Asia", "Carnatic music", "Music of Bollywood", "Ska",
            "Traditional music", "Independent music", "Song", "Background music",
            "Theme music", "Jingle (music)", "Soundtrack music", "Lullaby",
            "Video game music", "Christmas music", "Dance music", "Wedding music",
            "Happy music", "Sad music", "Tender music", "Exciting music",
            "Angry music", "Scary music"
        ]

        # Load the YAMNet model
        self.model = self.load_model()

        Log.info(f"Classify initialized with YAMNet model.")

        # Register commands
        self.command.add("set_output_dir", self.set_output_dir)
        self.command.add("set_threshold", self.set_threshold)

    def load_model(self):
        """
        Load the YAMNet model from TensorFlow Hub.
        """
        Log.info("Loading YAMNet model from TensorFlow Hub.")
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        model = hub.load(yamnet_model_handle)
        return model
    
    # go through passed EventData and classify each event

    def process(self, input_data):
        """
        Process the EventData/ add classification for each event if its confidence is above the threshold.

        Args:
            EventData: The input EventData object .

        Returns:
            updated EventData item
        """
        Log.info("Starting YAMNet classification...")

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        results = []
        for event_data in input_data:
            for event in event_data.items:
                # Prepare the audio data
                if event.data.sample_rate != self.sample_rate:
                    # Log.debug(f"Resampling from {event.data.sample_rate} Hz to {self.sample_rate} Hz")

                    audio = self.resampler(event)
                else:
                    audio = event.data

                # Convert to numpy array
                audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio

                # YAMNet expects a single channel
                if audio_np.ndim > 1:
                    audio_np = np.mean(audio_np, axis=0)

                # Run YAMNet inference
                try:
                    scores, embeddings, spectrogram = self.model(audio_np)
                    class_names = self.class_names
                except Exception as e:
                    Log.error(f"Error during YAMNet inference: {e}")
                    raise

                # Compute mean scores across all frames
                mean_scores = np.mean(scores, axis=0)

                # Start of Selection
                # Apply threshold to set confidence and class on event items
                highest_score = 0   
                highest_class = ""
                for i, score in enumerate(mean_scores):
                    current_class = class_names[i]
                    if score >= self.threshold and (not self.accepted_classes or current_class in self.accepted_classes):
                        if score > highest_score:
                            highest_score = score
                            highest_class = class_names[i]

                event.set_confidence(highest_score)
                event.set_classification(highest_class)

                Log.info(f"{event.name} classified as {highest_class} with confidence {int(highest_score * 100)}%")

        return input_data #input data is transformed in place

    def resampler(self, event):
        """
        Resample audio data to target sample rate and return as tensorflow tensor.
        
        Args:
            event: Event object containing audio data
        Returns:
            tf.Tensor: Resampled audio as float32 tensor
        """
        try:
            audio_data = event.data
            # Create a copy to avoid modifying original data
            audio_array = audio_data.data.copy()
            
            # Check for empty array
            if audio_array.size == 0:
                Log.warning(f"Empty audio array detected for event {event.name}")
                return tf.zeros([0], dtype=tf.float32)

            # Ensure audio is 1D (mono)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=-1)
                
            # Resample using librosa
            resampled = librosa.resample(
                y=audio_array,
                orig_sr=audio_data.sample_rate,
                target_sr=self.sample_rate
            )
            
            # Ensure the data is float32 and in the correct range (-1 to 1)
            resampled = resampled.astype(np.float32)
            if resampled.size > 0 and np.max(np.abs(resampled)) > 1.0:
                resampled = resampled / np.max(np.abs(resampled))
                
            return tf.convert_to_tensor(resampled, dtype=tf.float32)
            
        except Exception as e:
            Log.error(f"Error during resampling: {e}")
            raise

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        Log.info(f"Output directory set to {self.output_dir}")

    def set_threshold(self, threshold):
        if not (0.0 <= threshold <= 1.0):
            Log.error("Threshold must be between 0.0 and 1.0")
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        Log.info(f"Threshold set to {self.threshold}")

    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "output_dir": self.output_dir,
            "sample_rate": self.sample_rate,
            "threshold": self.threshold,
            "input": self.input.save(),
            "output": self.output.save()
        }

    def load(self, data):
        self.set_output_dir(data.get("output_dir", self.DEFAULT_OUTPUT_DIR))
        self.set_threshold(data.get("threshold", self.DEFAULT_THRESHOLD))
        self.input.load(data.get("input")) # just need to reconnect the inputs
        self.reload()
        Log.info("Classify state loaded successfully.")