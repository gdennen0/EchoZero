from message import Log
from abc import ABC, abstractmethod


"""
Generic structure to hold hata
"""


class Data(ABC):
    def __init__(self):
        Log.info(f"Creating Instance of the DataType Object")
        self.name = None
        self.description = None
        self.data = None

    def set_data(self, data):
        self.data = data
        Log.info(f"data set")

    def set_name(self, name):
        self.name = name
        Log.info(f"Set DataType name attribite to: '{name}'")

    def set_description(self, description):
        self.description = description
        Log.info(f"Set description: {description}")

    # Convert the object to a dictionary
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "data": self.data_to_dict() 
        }
    
    def load(self, data_dict):
        self.name = data_dict["name"]
        self.description = data_dict["description"]
        self.data_from_dict(data_dict["data"])

    @abstractmethod
    def data_to_dict(self):
        # Convert the data attribute to a dictionary
        # This method must be implemented by the subclass
        pass

    def data_from_dict(self, data_dict):
        # Convert the dictionary to the data attribute
        # This method must be implemented by the subclass
        pass