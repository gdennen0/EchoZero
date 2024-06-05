from Model.tools import Log

import logging

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
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def set_original_data(self, data):
        # Sets the original_data variable 
        self.original_data = data
        logging.debug(f"Original data set to {data}")

    def set_current_data(self, data):
        # Sets the current_data variable 
        self.current_data = data
        logging.debug(f"Current data set to {data}")

    def set_sample_rate(self, rate):
        # Sets the sample_rate variable 
        self.sample_rate = rate
        logging.debug(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        # Sets the frame_rate variable 
        self.frame_rate = rate
        logging.debug(f"Frame rate set to {rate}")

    def set_type(self, type):
        # Sets the type variable 
        self.type = type
        logging.debug(f"Type set to {type}")

    def set_name(self, name):
        # Sets the name variable 
        self.name = name
        logging.debug(f"Name set to {name}")

    def set_length_ms(self, length):
        # Sets the length_ms variable 
        self.length_ms = length
        logging.debug(f"Length in ms set to {length}")

    def set_processed_status(self, status):
        # Sets the processed_status variable
        self.processed_status = status
        logging.debug(f"Processed status set to {status}")


class audio_model:
    # Model to store instances of audio in
    def __init__(self):
        self.audio = []
        Log.info("Initialized Audio Model")

    
