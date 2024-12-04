from message import Log
import time
from Save.save_controller import SaveController

"Standardized data Controller Module for all data in primary application processes"

class DataController():
    def __init__(self):
        self.name = "data_controller"
        self.data_types = None 
        self.data_type = None
        self.data = None
        self.parent = None
        self.last_pull = None
        self.save = SaveController(self)
        self.save.add_attribute("data")
    
    def get(self):
        if self.data:
            return self.data
        else:
            Log.error(f"No data to get from {self.parent.name} data module")

    def append(self, data):
        self.data.append(data)

    def set_data(self, data):
        self.data = data
        if self.parent: Log.info(f"updated data module in {self.parent}") 
        else: Log.error(f"updated data module without parent set")

    def set_parent(self, parent_object):
        self.parent = parent_object
        Log.info(f"Set data modules parent to {self.parent.name}")

    def refresh_last_pull(self):
        self.last_pull = time.time()

    def set_data_type(self, data_type):
        self.data_type = data_type
        Log.info(f"Set data modules data type to {self.data_type}")

    def add_data_type(self, data_type):
        self.data_types.append(data_type)
        Log.info(f"Added data type {data_type} to data modules data types")
