from message import Log
from tools import prompt, path_exists, yes_no_prompt

# ===================
# Audio Object Class
# ===================

class audio:
    # Audio Object
    def __init__(self):
        self.directory = None
        self.audio = None
        self.sample_rate = None
        self.frame_rate = None
        self.type = None
        self.name = None
        self.length_ms = None
        self.processed_status = None
        self.tensor = None
        self.stems = None
    
    def serialize(self):
        return {
            "directory": self.directory,
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "type": self.type,
            "name": self.name,
            "length_ms": self.length_ms,
            "processed_status": self.processed_status,
            "tensor": self.tensor is not None,
            "stems": self.stems is not None,
        }
    
    def deserialize(self, data):
        self.directory = data.get("directory")
        self.sample_rate = data.get("sample_rate")
        self.frame_rate = data.get("frame_rate")
        self.type = data.get("type")
        self.name = data.get("name")
        self.length_ms = data.get("length_ms")
        self.processed_status = data.get("processed_status")
        # Note: tensor and stems are not fully restored here, only their existence is noted
        self.tensor = None if not data.get("tensor") else "Tensor data placeholder"
        self.stems = None if not data.get("stems") else "Stems data placeholder"
        
        # Log summary of all deserialized information
        Log.info(f"Audio object deserialized with directory: {self.directory}, sample rate: {self.sample_rate}, "
                 f"frame rate: {self.frame_rate}, type: {self.type}, name: {self.name}, length (ms): {self.length_ms}, "
                 f"processed status: {self.processed_status}, tensor exists: {self.tensor is not None}, "
                 f"stems exist: {self.stems is not None}")
    
    def set_dir(self, dir):
        self.directory = dir
        Log.info(f"Audio directory set to '{self.directory}'")

    def set_audio(self, data):
        self.audio = data
        Log.info(f"Audio data loaded")

    def set_tensor(self, t):
        self.tensor = t
        Log.info(f"Set tensor data")
        
    def set_sample_rate(self, rate):
        # Sets the sample_rate variable 
        self.sample_rate = rate
        Log.info(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        # Sets the frame_rate variable 
        self.frame_rate = rate
        Log.info(f"Frame rate set to {rate}")

    def set_type(self, type):
        # Sets the type variable 
        self.type = type
        Log.info(f"Type set to {type}")

    def set_name(self, name):
        # Sets the name variable
        self.name = name
        Log.info(f"Name set to {name}")

    def set_length_ms(self, length):
        # Sets the length_ms variable 
        self.length_ms = length
        Log.info(f"Length in ms set to {length}")

    def set_processed_status(self, status):
        # Sets the processed_status variable
        self.processed_status = status
        Log.info(f"Processed status set to {status}")


# ==================
# Audio Model Class
# ==================
class audio_model:
    # Model to store instances of audio in
    def __init__(self):
        self.objects = []
        self.selected_audio = None
        Log.info("Initialized Audio Model")

    def serialize(self):
        serialized_objects = []
        for obj in self.objects:
            serialized_objects.append(obj.serialize())
        return serialized_objects
    
    def deserialize(self, data):
        self.objects = []
        self.selected_audio = None
        for obj_data in data:
            new_audio = audio()
            new_audio.deserialize(obj_data)
            self.objects.append(new_audio)
        Log.info("Audio model deserialized successfully")
    
    def reset(self):
        self.objects = []
        self.selected_audio = None
        Log.info("Reset Audio Model")

    def select(self, index):
        Log.info(f"Attempting to select audio object with index {index}")
        if index is not None:
            if index < len(self.objects):
                self.selected_audio = self.objects[index]
                Log.info(f"Selected audio {self.selected_audio.name} at index: {index}")
                return
            else:
                Log.error("Index out of range")
        else:
            Log.error(f"Passed index has a None value '{index}' to an audio object")
                    
    def list(self):
        Log.info('*' * 20 + "AUDIO OBJECTS" + '*' * 20)
        for index, object in enumerate(self.objects):
            Log.info(f"Index: {index}, Audio Name: {object.name}")
        Log.info('*' * 53)

    def create_audio_object(self, dir, data, tensor, sr, fps, type, name):
        Log.info("create_audio_object")
        # creates an audio object and updates the necessary data
        a = audio()
        a.set_dir(dir + f"/audio/{name}")
        a.set_audio(data)
        a.set_tensor(tensor)
        a.set_sample_rate(sr)
        a.set_frame_rate(fps)
        a.set_type(type)
        a.set_name(name)

        self.objects.append(a)
        Log.info(f"Added audio {a.name} to model")

    def add_stems(self, stems):
        # adds passed stems to the selected audio objects stems list
        if self.selected_audio: # check to see if audio object is selected
            for stem in stems:
                self.selected_audio.stems.append(stem)
                print(f"added stem '{stem.name} to {self.selected_audio.name}")
        else: 
            Log.error("No audio object selected, please ")

    def delete(self, a_index):
        name = self.objects[a_index].name
        del self.objects[a_index]
        Log.warning(f"Deleted audio object '{name}' at index: {a_index}")

    def rename(self, a_index, new_name):
        old_name = self.objects[a_index].name
        self.objects[a_index].name = new_name
        Log.info(f"Renamed {old_name} to {new_name}")
    
    def get_audio_file_path(self, index):
        return self.objects[index].directory + f"/{self.objects[index].name}.mp3" # FIX THIS TO HAVE DYNAMIC EXTENSIONS

    def get_stems_file_path(self, index):
        stems_path = self.objects[index].directory + "/Stems"
        return stems_path
    
    def get_tensor(self, index):
        return self.objects[index].tensor
    
    def get_sr(self, index):
        return self.objects[index].sample_rate
    
    def get_audio(self, index):
        return self.objects[index].audio