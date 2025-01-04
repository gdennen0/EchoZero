from src.Utils.message import Log
from abc import ABC, abstractmethod
from src.Project.Data.data import Data
from src.Utils.tools import prompt
import os
"Standardized data Controller Module for all data in primary application processes"
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Data.Types.event_data import EventData

class DataController(ABC):
    """Responsible for managing data for another object. Will only accept DataTypes that are added to the data_types list"""
    def __init__(self, parent):
        self.name = "data_controller"
        self.parent = parent
        self.data = []  # list of data items
        self.data_types = [AudioData, EventData]


    def get_by_type(self, type):
        for item in self.data:
            if item.type == type:
                return item
        return None
    
    def get(self, name):
        for item in self.data:
            if item.name == name:
                return item
        return None
    
    def get_all(self):
        return self.data
    
    def set_name(self, name):
        self.name = name
        Log.info(f"Set data controller name to {name}")
    
    def get_name(self):
        return self.name
    
    def add(self, data):
        """adds data to the controller if its type matches a type in data_types list"""
        if isinstance(data, list):
            Log.error("Lists of data cannot be added to a data controller")
        else:
            self.data.append(data)
            if data.name:
                Log.info(f"Added data {data.name}")
            else:
                Log.info(f"Added data {data}")

    def clear(self):
        """clears the data list"""
        self.data = []

    def set_parent(self, parent_object):
        """sets the parent of the data controller"""
        self.parent = parent_object
        Log.info(f'Set data modules parent to {parent_object.name}')

    def get_metadata(self):
        metadata = {
            "name":self.name,
            "metadata":[item.get_metadata() for item in self.data]
        }  
        return metadata
    
    def save(self, save_dir):
        for item in self.data:
            item_dir = os.path.join(save_dir, item.name)
            if not os.path.exists(item_dir):
                os.makedirs(item_dir)
            item.save(item_dir)
    
    def load(self, data_controller_metadata, block_dir): 
        self.set_name(data_controller_metadata.get("name"))

        data_items = data_controller_metadata.get("metadata")
        for data_item in data_items: 
            data_item_name = data_item.get("name")
            data_item_type = data_item.get("type")
            data_item_dir = os.path.join(block_dir, data_item_name)

            Log.info(f"Data item name: {data_item_name}")
            Log.info(f"Data item type: {data_item_type}")
            Log.info(f"Data item dir: {data_item_dir}")

            for data_type in self.data_types:
                if data_item_type == data_type.type:
                    new_data_item = data_type()

                    new_data_item.load(data_item, data_item_dir)

                    self.add(new_data_item)
                    break