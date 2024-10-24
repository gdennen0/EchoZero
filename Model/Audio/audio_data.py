from message import Log
from tools import create_audio_data
import os

class AudioData:
    def __init__(self):
        self.directory = None
        self.path = None
        self.audio = None
        self.sample_rate = None
        self.frame_rate = None
        self.type = None
        self.name = None
        self.length_ms = None
        self.processed_status = None

    def serialize(self):
        """
        Serializes the audio object.
        """
        try:
            return {
                "directory": self.directory,
                "path": self.path,
                "sample_rate": self.sample_rate,
                "frame_rate": self.frame_rate,
                "type": self.type,
                "name": self.name,
                "length_ms": self.length_ms,
                "processed_status": self.processed_status,
            }
        except (AttributeError, TypeError) as e:
            Log.error(f"Error during audio serialization: {e}")
            return {}

    def deserialize(self, data):
        """
        Deserializes the audio object from data.
        """
        try:
            self.directory = data.get("directory")
            self.path = data.get('path')
            self.sample_rate = data.get("sample_rate")
            self.frame_rate = data.get("frame_rate")
            self.type = data.get("type")
            self.name = data.get("name")
            self.length_ms = data.get("length_ms")
            self.processed_status = data.get("processed_status")

            return self
        
        except (KeyError, TypeError) as e:
            Log.error(f"Error during audio deserialization: {e}")
            return self

    def set_dir(self, dir):
        """
        Sets the directory of the audio object.
        """
        if self.directory:
            Log.error(f"Overwriting existing directory value: {self.directory}")
        self.directory = dir
        Log.info(f"Audio directory set to '{self.directory}'")

    def set_extension(self, extension):
        """
        Sets the extension of the audio object.
        """
        if hasattr(self, 'extension') and self.extension:
            Log.error(f"Overwriting existing extension value: {self.extension}")
        self.extension = extension
        Log.info(f"Extension set to '{extension}'")

    def set_path(self):
        """
        Sets the path of the audio object.
        """
        if self.path:
            Log.error(f"Overwriting existing path value: {self.path}")
        if self.extension:
            self.path = os.path.join(self.directory, f"{self.name}{self.extension}")
            Log.info(f"Set audio path to {self.path}")
        else:
            Log.error("No extension set for file, cannot set path for audio object")

    def set_audio(self, data):
        """
        Sets the audio data of the audio object.
        """
        if self.audio:
            Log.error("Overwriting existing audio data")
        self.audio = data
        Log.info("Audio data loaded")

    def set_tensor(self, tensor):
        """
        Sets the tensor data of the audio object.
        """
        if hasattr(self, 'tensor') and self.tensor:
            Log.error("Overwriting existing tensor data")
        self.tensor = tensor
        Log.info("Set tensor data")

    def set_sample_rate(self, rate):
        """
        Sets the sample rate of the audio object.
        """
        if self.sample_rate:
            Log.error(f"Overwriting existing sample rate value: {self.sample_rate}")
        self.sample_rate = rate
        Log.info(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        """
        Sets the frame rate of the audio object.
        """
        if self.frame_rate:
            Log.error(f"Overwriting existing frame rate value: {self.frame_rate}")
        self.frame_rate = rate
        Log.info(f"Frame rate set to {rate}")

    def set_type(self, type):
        """
        Sets the type of the audio object.
        """
        if self.type:
            Log.error(f"Overwriting existing type value: {self.type}")
        self.type = type
        Log.info(f"Type set to {type}")

    def set_name(self, name):
        """
        Sets the name of the audio object.
        """
        if self.name:
            Log.error(f"Overwriting existing name value: {self.name}")
        self.name = name
        Log.info(f"Name set to {name}")

    def set_length_ms(self, length):
        """
        Sets the length in milliseconds of the audio object.
        """
        if self.length_ms:
            Log.error(f"Overwriting existing length value: {self.length_ms}")
        self.length_ms = length
        Log.info(f"Length in ms set to {length}")

    def set_processed_status(self, status):
        """
        Sets the processed status of the audio object.
        """
        if self.processed_status:
            Log.error(f"Overwriting existing processed status value: {self.processed_status}")
        self.processed_status = status
        Log.info(f"Processed status set to {status}")

    def get_audio_metadata(self):
        """
        Gets the metadata of the audio object.
        """
        return self.serialize()

    def get_audio_file_path(self):
        """
        Gets the file path of the audio object.
        """
        return self.path
    

    def generate_from_path(self, path):
        """
        Generates the audio data from the given path.
        """
        SAMPLE_RATE = 44100
        try:
            self.path = self.set_path(path)
            self.set_audio(create_audio_data(path, SAMPLE_RATE)[0])
            self.set_sample_rate(SAMPLE_RATE)
            self.set_name = os.path.basename(path)
            self.set_length_ms = len(self.audio) / self.sample_rate * 1000
            self.set_processed_status("Generated")
            Log.info(f"Audio data generated from path: {self.path}")
        except Exception as e:
            Log.error(f"Error generating audio data from path: {e}")
