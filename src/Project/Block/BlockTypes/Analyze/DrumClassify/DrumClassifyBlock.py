from src.Project.Block.block import Block
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
import librosa
import numpy as np
import tensorflow as tf
import os
import shutil
import tempfile
import subprocess
from src.Utils.message import Log

class DrumClassifyBlock(Block):
    """
    A block that classifies drum audio samples using the pre-trained CNN model from the drum-audio-classifier repository.
    The model can classify drum types like kick, snare, hi-hat, clap, etc.
    """
    name = "DrumClassify"
    type = "DrumClassify"

    def __init__(self):
        super().__init__()
        self.name = "DrumClassify"
        self.type = "DrumClassify"

        # Add input/output types
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Initialize model path
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        
        # Class mappings from the drum-audio-classifier repo
        self.class_names = {
            0: "clap",   # Clap
            1: "hhc",     # Closed Hat
            2: "kick",   # Kick
            3: "hho", # Open Hat
            4: "snare"   # Snare
        }
        
        # Initialize model
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the TensorFlow model for drum classification"""
        try:
            Log.info(f"Loading drum classifier model from {self.model_dir}")
            
            # Check if model directory exists
            if not os.path.exists(self.model_dir) or len(os.listdir(self.model_dir)) == 0:
                Log.error(f"Model directory {self.model_dir} does not exist or is empty")
                Log.info("Attempting to download the model from GitHub...")
                self.download_model()
            
            # Load the model
            self.model = tf.saved_model.load(self.model_dir)
            
            # Log model signatures for debugging
            signature_keys = list(self.model.signatures.keys())
            Log.info(f"Model signature keys: {signature_keys}")
            
            if "serving_default" not in signature_keys:
                Log.error("Model does not have a 'serving_default' signature")
                return None
            
            # Get the concrete function for serving
            self.serving_function = self.model.signatures["serving_default"]
            
            # Check input and output specs
            input_specs = self.serving_function.structured_input_signature
            output_specs = self.serving_function.structured_outputs
            
            Log.info(f"Model input specs: {input_specs}")
            Log.info(f"Model output specs: {output_specs}")
            
            Log.info("Drum classifier model loaded successfully")
        except Exception as e:
            Log.error(f"Failed to load drum classifier model: {e}")
            self.model = None

    def download_model(self):
        """Download the drum-audio-classifier model from GitHub"""
        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                Log.info(f"Created temporary directory: {temp_dir}")
                
                # Clone the repository
                Log.info("Cloning drum-audio-classifier repository...")
                repo_url = "https://github.com/aabalke33/drum-audio-classifier.git"
                subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
                
                # Copy the model files
                model_source = os.path.join(temp_dir, "saved_model", "model_20230607_02")
                if os.path.exists(model_source):
                    # Create model directory if it doesn't exist
                    os.makedirs(self.model_dir, exist_ok=True)
                    
                    # Copy all model files
                    for item in os.listdir(model_source):
                        src_path = os.path.join(model_source, item)
                        dst_path = os.path.join(self.model_dir, item)
                        
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dst_path)
                    
                    Log.info(f"Successfully copied model files to {self.model_dir}")
                else:
                    Log.error(f"Model directory not found in repository: {model_source}")
        except Exception as e:
            Log.error(f"Failed to download model: {e}")
            # Make sure model directory exists even if download failed
            os.makedirs(self.model_dir, exist_ok=True)

    def transform_audio(self, y, sr):
        """
        Transform audio to mel-spectrogram for classification
        Based on the exact approach from the drum-audio-classifier repository
        """
        try:
            # Follow the exact preprocessing from the original repo's sample_preparer function
            # Resample to 22050Hz
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
            
            # Trim silence 
            y, _ = librosa.effects.trim(y, top_db=50)
            
            # Generate mel spectrogram
            melspect = librosa.feature.melspectrogram(y=y)
            
            # Create an empty array with the specific dimensions expected by the model
            # Exactly 128x100x3 dimensions as used in the original model
            LENGTH = 100
            sample = np.zeros((128, LENGTH, 3), dtype=np.float32)  # Explicit float32 dtype
            
            # Fill the array with the mel spectrogram values
            # Copy the same value to all 3 channels (RGB) as in original repo
            for i in range(min(128, melspect.shape[0])):
                for j in range(min(LENGTH, melspect.shape[1])):
                    value = float(melspect[i][j])  # Ensure it's a float
                    sample[i][j][0] = value  # R channel
                    sample[i][j][1] = value  # G channel
                    sample[i][j][2] = value  # B channel
            
            # Add batch dimension for model input
            sample_data = np.expand_dims(sample, axis=0).astype(np.float32)  # Ensure float32
            
            return sample_data
        except Exception as e:
            Log.error(f"Error transforming audio: {e}")
            return None

    def predict_class(self, spectrogram):
        """
        Predict drum class from spectrogram
        """
        if self.model is None:
            Log.error("Model not initialized")
            return None
        
        try:
            # Make sure data is float32 before feeding to model
            if spectrogram.dtype != np.float32:
                spectrogram = spectrogram.astype(np.float32)
            
            # Create a TensorFlow tensor with the appropriate name
            input_tensor = tf.convert_to_tensor(spectrogram)
            
            # Use the serving function to make prediction
            prediction = self.serving_function(input_tensor)
            
            # Extract the prediction array from the output dict
            prediction_key = list(prediction.keys())[0]  # Usually 'dense_3' or similar
            prediction_array = prediction[prediction_key].numpy()
            
            # Get predicted class index
            type_num = np.argmax(prediction_array, axis=1)[0]
            
            # Map to class name using our mapping
            if type_num in self.class_names:
                confidence = prediction_array[0][type_num]
                Log.info(f"Predicted class: {self.class_names[type_num]} with confidence: {confidence:.4f}")
                return self.class_names[type_num]
            else:
                return "unknown"
        except Exception as e:
            Log.error(f"Error during prediction: {e}")
            return "unknown"

    def process(self, event_data_list):
        """
        Process the input event data and classify each drum sample.
        Returns a new EventData with classified samples.
        """
        if not self.model:
            Log.error("Drum classifier model not initialized")
            return []

        Log.info(f"Processing {len(event_data_list)} event data items")
        
        # Create new event data for output
        output_event_data = EventData()
        output_event_data.name = "ClassifiedDrums"
        output_event_data.description = "Drum samples classified by type"

        for event_data in event_data_list:
            Log.info(f"Processing event data: {event_data.name} with {len(event_data.items)} items")
            for item in event_data.items:
                # Get audio data from event item
                audio_data = item.data
                if not audio_data:
                    Log.warning(f"No audio data in item {item.name}, skipping")
                    continue

                # Get audio samples and sample rate
                y = audio_data.get_data()
                sr = audio_data.get_sample_rate()
                
                if y is None or len(y) == 0:
                    Log.warning(f"Empty audio data in item {item.name}, skipping")
                    continue

                # Transform audio to mel spectrogram
                Log.info(f"Transforming audio for {item.name}")
                spectrogram = self.transform_audio(y, sr)
                if spectrogram is None:
                    Log.warning(f"Failed to transform audio for {item.name}, skipping")
                    continue

                # Classify the sample
                Log.info(f"Classifying {item.name}")
                predicted_class = self.predict_class(spectrogram)
                if not predicted_class:
                    Log.warning(f"Failed to classify {item.name}, skipping")
                    continue
                
                # Create new event item with classification
                new_item = EventItem()
                new_item.set_name(f"{predicted_class}_{item.name}")
                new_item.time = item.time
                new_item.source = "DrumClassify"
                new_item.set_data(audio_data)
                new_item.classification = predicted_class

                # Add to output event data
                output_event_data.add_item(new_item)
                Log.info(f"Classified {item.name} as {predicted_class}")

        return [output_event_data]

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata() if hasattr(self, 'data') else {}
        }

    def save(self, save_dir):
        if hasattr(self, 'data'):
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
    