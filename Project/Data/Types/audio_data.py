from Project.Data.data import Data
from Utils.message import Log

class AudioData(Data):
    name = "AudioData"
    def __init__(self):
        super().__init__()
        self.name = "AudioData"
        self.type = "AudioData"
        self.path = None    
        self.sample_rate = None
        self.frame_rate = None
        self.length_ms = None

    def set_path(self, path):
        if self.path:
            Log.error(f"Overwriting existing path value: {self.path}")
        self.path = path
        Log.info(f"Path set to {path}")

    def set_sample_rate(self, rate):
        if self.sample_rate:
            Log.error(f"Overwriting existing sample rate value: {self.sample_rate}")
        self.sample_rate = rate
        Log.info(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        if self.frame_rate:
            Log.error(f"Overwriting existing frame rate value: {self.frame_rate}")
        self.frame_rate = rate
        Log.info(f"Frame rate set to {rate}")

    def set_length_ms(self, length):
        if self.length_ms:
            Log.error(f"Overwriting existing length value: {self.length_ms}")
        self.length_ms = length
        Log.info(f"Length in ms set to {length}")

    def save(self):
        return {"name":self.name, 
                "path":self.path, 
                "sample_rate":self.sample_rate, 
                "frame_rate":self.frame_rate, 
                "length_ms":self.length_ms
                }
