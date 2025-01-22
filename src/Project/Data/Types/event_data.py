from src.Project.Data.data import Data
from src.Utils.message import Log
import os
from src.Project.Data.Types.event_item import EventItem

class EventData(Data):
    """
    Event Data is a collection of EventItems.
    """
    type = "EventData" # this is redundant fix later lol
    name = "EventData" # this is redundant fix later lol

    def __init__(self):
        super().__init__()
        self.name = "EventData" 
        self.type = "EventData"
        self.description = "A collection of events"
        self.items = []
        self.source = None


    def add_item(self, item):
        self.items.append(item)
        # Log.info(f"Added item: {item.name} to event data: {self.name}")

    def set_source(self, source):
        self.source = source
        # Log.info(f"Set source for event data: {self.name} to {source}")

    def get_source(self):
        return self.source
    
    def get(self):
        return self.items
    
    def get_all(self):
        return self.items
    
    def get_name(self):
        return self.name
    
    def get_metadata(self):
        return {
            "name":self.name, 
            "type":self.type,
            "description":self.description, 
            "items":[item.get_metadata() for item in self.items]
        }

    def save(self, save_dir):
        for item in self.items:
            item_dir = os.path.join(save_dir, item.name)
            if not os.path.exists(item_dir):
                os.makedirs(item_dir)
            data_dir = os.path.join(item_dir, "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            item.save(data_dir)
    
    def load(self, data_item_metadata, event_data_dir):
        self.set_name(data_item_metadata.get("name"))
        self.set_description(data_item_metadata.get("description"))
        self.set_source(data_item_metadata.get("source"))

        for item in data_item_metadata.get("items"):
            data_item_dir = os.path.join(event_data_dir, item.get("name"))
            event_item = EventItem()
            event_item.load(item, data_item_dir)
            self.add_item(event_item)