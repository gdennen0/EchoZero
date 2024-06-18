from message import Log
from tools import prompt, path_exists, yes_no_prompt

# ===================
# Audio Object Class
# ===================
class audio:
    # Audio Object
    def __init__(self):
        self.audio_file_path = None
        self.original_data = None
        self.current_data = None
        self.sample_rate = None
        self.frame_rate = None
        self.type = None
        self.name = None
        self.length_ms = None
        self.processed_status = None
        self.tensor = None
        self.stems = None
        self.path = None
    
    def set_audio_file_path(self, audio_file_path):
        self.audio_file_path = audio_file_path
        Log.info(f"Audio file path set to '{audio_file_path}'")

    def set_original_data(self, data):
        # Sets the original_data variable & updates the current_data variable? For the time being atleast
        self.original_data = data
        self.set_current_data(data)
        Log.info(f"Original data set to ... a list that im not going to print")

    def set_current_data(self, data):
        # Sets the current_data variable 
        self.current_data = data
        Log.info(f"Current data set to ... a list that im not going to print")

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

    def set_path(self, path):
        self.path = path
        Log.info(f"Set path to '{path}'")


# ==================
# Audio Model Class
# ==================
class audio_model:
    # Model to store instances of audio in
    def __init__(self):
        self.objects = []
        self.selected_audio = None
        Log.info("Initialized Audio Model")

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

    def add_audio(self, a):
        # adds passed audio object to the models objects list
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
        return self.objects[index].audio_file_path

    def get_stems_file_path(self, index):
        stems_path = self.objects[index].path + "/Stems"
        return stems_path
    
    def get_tensor(self, index):
        return self.objects[index].tensor
    
    def get_sr(self, index):
        return self.objects[index].sample_rate