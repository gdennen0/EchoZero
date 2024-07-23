from message import Log
from tools import create_audio_tensor, create_audio_data
import os
from .stem import Stem
from .event_pool import EventPool
from .utils import generate_metadata

class Audio:
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
        self.event_pools = []
        self.tensor = None
        self.stems = []
        self.features = None

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
                "event_pools": [pool.serialize() for pool in self.event_pools],
                "tensor": self.tensor is not None,
                "stems": [stem.serialize() for stem in self.stems],
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

            if 'event_pools' in data:
                for pool_data in data['event_pools']:
                    event_pool_object = EventPool()
                    event_pool_object.deserialize(pool_data)
                    self.add_event_pool(event_pool_object)

            if data.get('tensor'):
                self.tensor = create_audio_tensor(self.path, self.sample_rate)

            for stem_data in data.get('stems', []):
                stem = Stem()
                stem.deserialize(stem_data)
                self.stems.append(stem)

            return self
        except (KeyError, TypeError) as e:
            Log.error(f"Error during audio deserialization: {e}")
            return self

    def set_dir(self, dir):
        """
        Sets the directory of the audio object.
        """
        self.directory = dir
        Log.info(f"Audio directory set to '{self.directory}'")

    def set_extension(self, extension):
        """
        Sets the extension of the audio object.
        """
        self.extension = extension
        Log.info(f"Extension set to '{extension}'")

    def set_path(self):
        """
        Sets the path of the audio object.
        """
        if self.extension:
            self.path = os.path.join(self.directory, f"{self.name}{self.extension}")
            Log.info(f"Set audio path to {self.path}")
        else:
            Log.error("No extension set for file, cannot set path for audio object")

    def set_audio(self, data):
        """
        Sets the audio data of the audio object.
        """
        self.audio = data
        Log.info("Audio data loaded")

    def set_tensor(self, tensor):
        """
        Sets the tensor data of the audio object.
        """
        self.tensor = tensor
        Log.info("Set tensor data")

    def set_sample_rate(self, rate):
        """
        Sets the sample rate of the audio object.
        """
        self.sample_rate = rate
        Log.info(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        """
        Sets the frame rate of the audio object.
        """
        self.frame_rate = rate
        Log.info(f"Frame rate set to {rate}")

    def set_type(self, type):
        """
        Sets the type of the audio object.
        """
        self.type = type
        Log.info(f"Type set to {type}")

    def set_name(self, name):
        """
        Sets the name of the audio object.
        """
        self.name = name
        Log.info(f"Name set to {name}")

    def set_length_ms(self, length):
        """
        Sets the length in milliseconds of the audio object.
        """
        self.length_ms = length
        Log.info(f"Length in ms set to {length}")

    def set_processed_status(self, status):
        """
        Sets the processed status of the audio object.
        """
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
    
    def get_features(self):
        return self.features
    
    def add_feature(self, feature):
        self.features.append(feature)
        Log.info(f"Added feature to {self.name}")

    def add_features(self, features):
        if isinstance(features, tuple):
            Log.error("Features is a tuple, expected a tensor")
        for feature in features:
            self.add_feature(feature)

    def add_stem(self, path):
        """
        Adds a stem to the audio object.
        """
        stem_name = os.path.basename(path).split('.')[0]
        audio_data, _ = create_audio_data(path, self.sample_rate)
        tensor, _ = create_audio_tensor(path, self.sample_rate)
        stem = Stem()
        stem.set_name(stem_name)
        stem.set_path(path)
        stem.set_audio(audio_data)
        stem.set_tensor(tensor)
        stem.set_sample_rate(self.sample_rate)
        self.stems.append(stem)
        generate_metadata(stem)

    def add_event_pool(self, event_pool_object):
        """
        Adds an event pool to the audio object.
        """
        self.event_pools.append(event_pool_object)
        Log.info(f"Added event pool object to audio object named '{self.name}'")

    