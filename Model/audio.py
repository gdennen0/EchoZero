from .tools import Log
# ===================
# Audio Object Class
# ===================
class audio:
    # Audio Object
    def __init__(self):
        self.original_data = None
        self.current_data = None
        self.sample_rate = None
        self.frame_rate = None
        self.type = None
        self.name = None
        self.length_ms = None
        self.processed_status = None
        self.tensor = None

    def set_original_data(self, data):
        # Sets the original_data variable & updates the current_data variable? For the time being atleast
        self.original_data = data
        self.set_current_data(data)
        Log.debug(f"Original data set to {data}")

    def set_current_data(self, data):
        # Sets the current_data variable 
        self.current_data = data
        Log.debug(f"Current data set to {data}")

    def set_tensor(self, t):
        self.tensor = t
        Log.debug(f"Updated tensor data")
        
    def set_sample_rate(self, rate):
        # Sets the sample_rate variable 
        self.sample_rate = rate
        Log.debug(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        # Sets the frame_rate variable 
        self.frame_rate = rate
        Log.debug(f"Frame rate set to {rate}")

    def set_type(self, type):
        # Sets the type variable 
        self.type = type
        Log.debug(f"Type set to {type}")

    def set_name(self, name):
        # Sets the name variable 
        self.name = name
        Log.debug(f"Name set to {name}")

    def set_length_ms(self, length):
        # Sets the length_ms variable 
        self.length_ms = length
        Log.debug(f"Length in ms set to {length}")

    def set_processed_status(self, status):
        # Sets the processed_status variable
        self.processed_status = status
        Log.debug(f"Processed status set to {status}")


# ==================
# Audio Model Class
# ==================
class audio_model:
    # Model to store instances of audio in
    def __init__(self):
        self.objects = []
        Log.debug("Initialized Audio Model")

    def add(self, a):
        self.objects.append(a)
        Log.debug(f"Added audio {a.name}")

    def delete(self, a_index):
        name = self.objects[a_index].name
        del self.objects[a_index]
        Log.debug(f"Deleted audio object '{name}' at index: {a_index}")

    
