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
        self.audio.deserialize(data)