from src.Project.Data.data import Data
from src.Utils.message import Log
from src.Project.Data.Types.audio_data import AudioData
import os
class EventItem(Data):
    """
    EventItem is a single event in a collection of events.
    """
    name = "EventItem" # this is redundant fix later lol
    type = "EventItem"  # this is redundant fix later lol

    def __init__(self,):
        super().__init__()
        self.name = "EventItem" 
        self.type = "EventItem"
        self.description = "no description set"
        self.time = None
        self.length = None
        self.source = None
        self.confidence = None
        self.classification = None
        self.data_types = [AudioData]

    def get_metadata(self):
        metadata = {
            "name":self.name, 
            "description":self.description,
            "classification":self.classification,
            "type":self.type,
            "time":self.time,
            "length":self.length,
            "source":self.source,
            "confidence":self.confidence,
            "metadata":None
        }
        if self.data:
            metadata["metadata"] = self.data.get_metadata()
        return metadata
    
    def get_end_time(self):
        if self.length is not None:
            return self.time + self.length
        else:
            Log.error(f"Length not set for event item: {self.name}")
            return None
        
    def get_length(self):
        return self.length
    
    def set_length(self, length):
        self.length = length
        # Log.info(f"Set length for event item: {self.name} to {length}")
    
    def get_data(self):
        return self.data
    
    def get(self):
        return self.data
    
    def set_time(self, time):
        self.time = time
        # Log.info(f"Set time for item: {self.name} to {time}")

    def set_source(self, source):
        self.source = source
        # Log.info(f"Set source for item: {self.name} to {source}")

    def set_confidence(self, confidence):
        self.confidence = confidence
        # Log.info(f"Set confidence for item: {self.name} to {confidence}")

    def set_classification(self, classification):
        self.classification = classification
        # Log.info(f"Set classification for item: {self.name} to {classification}")

    def get_time(self):
        return self.time
    
    def get_name(self):
        return self.name
    
    def get_source(self):
        return self.source
    
    def get_confidence(self):
        return self.confidence
    
    def get_classification(self):
        return self.classification

    def save(self, save_dir):
        if self.data:
            self.data.save(save_dir)
    
    def load(self, event_item_metadata, event_item_dir):
        self.set_name(event_item_metadata.get("name"))
        self.set_description(event_item_metadata.get("description"))
        self.set_classification(event_item_metadata.get("classification"))
        self.set_time(event_item_metadata.get("time"))
        self.set_source(event_item_metadata.get("source"))
        self.set_confidence(event_item_metadata.get("confidence"))

        event_item_data_dir = os.path.join(event_item_dir, 'data')
        data_metadata = event_item_metadata.get("metadata") # get the data metadata
        if data_metadata:
            if self.get_data_type(data_metadata.get("type")):
                data = self.get_data_type(data_metadata.get("type"))()
                data.load(data_metadata, event_item_data_dir)
                self.data = data
            else:
                Log.error(f"Data type {data_metadata.get('type')} not found")
        else:
            Log.error(f"Data metadata not found")

    def get_data_type(self, type):
        for data_type in self.data_types:
            if data_type.type == type:
                return data_type
        return None
    
    def get_type(self):
        return self.type

    def set_classification(self, classification):
        self.classification = classification

    def set_confidence(self, confidence):
        self.confidence = confidence
