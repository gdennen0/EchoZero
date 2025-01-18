from src.Project.Data.data import Data
from src.Utils.message import Log
import os
import soundfile as sf
class AudioData(Data):
    """
    AudioData is a single audio file.
    """
    name = "AudioData" # this is redundant fix later lol
    type = "AudioData" # this is redundant fix later lol
    
    def __init__(self):
        super().__init__()

        self.name = "AudioData"
        self.type = "AudioData"
        self.path = None    
        self.sample_rate = None
        self.frame_rate = None
        self.length_ms = None
        self.source = None
        self.metadata = None


    def set_path(self, path):
        if self.path:
            Log.warning(f"{self.name} {self.type} overwriting existing path value: {self.path}")
        self.path = path
        # Log.info(f"Path set to {path}")

    def set_sample_rate(self, rate):
        # if self.sample_rate:
            # Log.warning(f"{self.name} {self.type} overwriting existing sample rate value: {self.sample_rate}")
        self.sample_rate = rate
        # Log.info(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        if self.frame_rate:
            Log.warning(f"{self.name} {self.type} overwriting existing frame rate value: {self.frame_rate}")
        self.frame_rate = rate
        # Log.info(f"Frame rate set to {rate}")

    def set_length_ms(self, length):
        if self.length_ms:
            Log.warning(f"{self.name} {self.type} overwriting existing length value: {self.length_ms}")
        self.length_ms = length
        # Log.info(f"Length in ms set to {length}")

    def get_data(self):
        return self.data
    
    def get(self):
        return self.data

    def set_source(self, source):
        self.source = source

    def get_source(self):
        return self.source
    
    def get_sample_rate(self):
        return self.sample_rate
    
    def get_sr(self):
        return self.sample_rate
    
    def get_frame_rate(self):
        return self.frame_rate
    
    def get_length_ms(self):
        return self.length_ms
    
    def get_path(self):
        return self.path
    

    def get_metadata(self):
        return {
            "name":self.name, 
            "type":self.type,
            "path":self.path, 
            "sample_rate":self.sample_rate, 
            "frame_rate":self.frame_rate, 
            "length_ms":self.length_ms,
        }

    def save(self, save_dir):
        data_to_save = self.data.squeeze()
        sf.write(os.path.join(save_dir, f"{self.name}.wav"), data_to_save, self.sample_rate)

    def load(self, data_item_metadata, data_item_dir):
        self.set_name(data_item_metadata.get("name"))
        self.set_type(data_item_metadata.get("type"))
        self.set_path(os.path.join(data_item_dir, f"{self.name}.wav"))
        self.set_sample_rate(data_item_metadata.get("sample_rate"))
        self.set_frame_rate(data_item_metadata.get("frame_rate"))
        self.set_length_ms(data_item_metadata.get("length_ms"))
        self.set_source(data_item_metadata.get("source"))

        
        data, samplerate = sf.read(self.path)
        self.set_data(data)
        self.set_sample_rate(samplerate)

