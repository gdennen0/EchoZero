from message import Log

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


# ==================
# Audio Model Class
# ==================
class audio_model:
    # Model to store instances of audio in
    def __init__(self):
        self.objects = []
        Log.info("Initialized Audio Model")

    def add(self, a):
        self.objects.append(a)
        Log.info(f"Added audio {a.name} to model")

    def delete(self, a_index):
        name = self.objects[a_index].name
        del self.objects[a_index]
        Log.warning(f"Deleted audio object '{name}' at index: {a_index}")

    
