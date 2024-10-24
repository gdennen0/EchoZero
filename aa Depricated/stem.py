from message import Log
from tools import create_audio_tensor, create_audio_data
from .event_pool import EventPool
import os

class Stem:
    """
    Stem Object
    """
    def __init__(self):
        self.path = None
        self.directory = None
        self.name = None
        self.audio = None
        self.sample_rate = None
        self.frame_rate = None
        self.tensor = None
        self.event_pools = []

    def set_name(self, name):
        """
        Sets the name of the stem.
        """
        self.name = name
        Log.info(f"Set stem name to {name}")

    def set_path(self, path):
        """
        Sets the path of the stem.
        """
        self.path = path
        self.set_dir(path)
        Log.info(f"Set stem path to {self.path}")

    def set_dir(self, path):
        """
        Sets the directory of the stem from the given path.
        """
        self.directory = os.path.dirname(path)
        Log.info(f"Set stem directory to {self.directory}")

    def set_audio(self, audio):
        """
        Sets the audio data of the stem.
        """
        self.audio = audio
        Log.info("Set stem audio data")

    def set_sample_rate(self, sr):
        """
        Sets the sample rate of the stem.
        """
        self.sample_rate = sr
        Log.info(f"Set sample rate to {sr}")

    def set_frame_rate(self, rate):
        """
        Sets the frame rate of the stem.
        """
        self.frame_rate = rate
        Log.info(f"Set frame rate to {rate}")

    def set_tensor(self, tensor):
        """
        Sets the tensor data of the stem.
        """
        self.tensor = tensor
        Log.info("Set stem tensor data")

    def add_event_pool(self, event_pool_object):
        """
        Adds an event pool to the stem.
        """
        self.event_pools.append(event_pool_object)
        Log.info(f"Added event pool object to stem object named '{self.name}'")

    def serialize(self):
        """
        Serializes the stem data into a JSON format.
        """
        try:
            return {
                'stem': {
                    "name": self.name,
                    "path": self.path,
                    "sample_rate": self.sample_rate,
                    "frame_rate": self.frame_rate,
                    "audio": self.audio is not None,
                    "tensor": self.tensor is not None,
                    "event_pools": [pool.serialize() for pool in self.event_pools]
                }
            }
        except (AttributeError, TypeError) as e:
            Log.error(f"Error during stem serialization: {e}")
            return {}

    def deserialize(self, data):
        """
        Deserializes the stem data from a JSON format.
        """
        try:
            stem_data = data['stem']
            self.set_name(stem_data.get('name', ''))
            self.set_path(stem_data.get('path', ''))
            self.set_sample_rate(stem_data.get('sample_rate', 0))
            self.set_frame_rate(stem_data.get('frame_rate', 0))
            self.set_tensor(create_audio_tensor(self.path, self.sample_rate))
            self.set_audio(create_audio_data(self.path, self.sample_rate))

            if 'event_pools' in stem_data:
                for pool_data in stem_data['event_pools']:
                    event_pool_object = EventPool()
                    event_pool_object.deserialize(pool_data)
                    self.add_event_pool(event_pool_object)

            return self
        except (KeyError, TypeError) as e:
            Log.error(f"Error during stem deserialization: {e}")
            return self