from Model.audio import audio_model
from message import Log
import json

class Model:
    def __init__(self):
        self.audio = audio_model()

    def reset(self):
        self.audio.reset()

    def serialize(self):
        return self.audio.serialize()
    
    def deserialize(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
            # Deserialize audio data if present
            if 'audio' in data:
                self.audio.deserialize(data['audio'])
            # Future deserialization for other categories can be added here
            # Example:
            # if 'video' in data:
            #     self.video.deserialize(data['video'])
        