from src.Utils.message import Log
from abc import ABC, abstractmethod
from src.Project.Data.data import Data
from src.Utils.tools import prompt
"Standardized data Controller Module for all data in primary application processes"

class DataController(ABC):
    """Responsible for managing data for another object. Will only accept DataTypes that are added to the data_types list"""
    def __init__(self, parent):
        self.name = "data_controller"
        self.parent = parent
        self.data = []  # list of data items
    
    def get(self, name=None):
        if name:
            for item in self.data:
                if item.name == name:
                    return item
        else:
            name = prompt("Enter the name of the data item to get: ")
            for item in self.data:
                if item.name == name:
                    return item
        return None
    
    def get_all(self):
        return self.data
    
    def add(self, data):
        """adds data to the controller if its type matches a type in data_types list"""
        if isinstance(data, list):
            Log.error("Lists of data cannot be added to a data controller")
        else:
            self.data.append(data)
            Log.info(f"Added data {data} to data controller")

    def clear(self):
        """clears the data list"""
        self.data = []

    def set_parent(self, parent_object):
        """sets the parent of the data controller"""
        self.parent = parent_object
        Log.info(f'Set data modules parent to {parent_object.name}')

    # def set_data_type(self, data_type):
    #     self.data_type = data_type
    #     Log.info(f"Set data modules data type to {self.data_type}")

    # def add_data_type(self, data_type):
    #     if data_type not in self.data_types:
    #         if isinstance(data_type, Data):
    #             self.data_types.append(data_type)
    #             Log.info(f"Added data type {data_type.name} to data modules data types")
    #         else:
    #             Log.error(f"Data type {data_type.name} is not a valid data type")
    #     else:
    #         Log.error(f"Data type {data_type.name} already exists within data modules data types")


    def save(self):
        """saves the data controller to a dictionary"""
        data = {}
        for item in self.data:
            data[item.name] = item.save()
        return {
            "name":self.name,
            "data":data
        }  
    
    def load(self, data): 
        self.name = data["name"]
        self.data = data["data"]
        for item in self.data:
            item.load(data[item.name])
        