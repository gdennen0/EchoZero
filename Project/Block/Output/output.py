from Utils.message import Log
from Utils.tools import prompt
from Project.Command.command_controller import CommandController
from Project.Data.data_controller import DataController

class Output:
    """
    Output is responsible for holding the last data from its parent block and delivering it to the input port of the connected block
    """
    def __init__(self, parent_block):
        self.name = None
        self.type = None # should be set in subclass
        self.parent_block = parent_block

        self.data_types = []
        self.data_type = None
        self.data = DataController(self) # initialize a data controller
        self.data.set_parent(self)

        self.command = CommandController() # initialize a command controller
        self.command.add("set_name", self.set_name)
        
    def set_parent_block(self, parent_block):
        self.parent_block = parent_block
        Log.info(f"Input {self.name} parent block set to: {parent_block.name}")


    def set_name(self, name=None):
        if name:
            self.name = name
            Log.info(f"Port name set to: {name}")
        else:
            self.name = prompt(f"Please enter the name of the port: ")
            Log.info(f"Port name set to: {self.name}")

    def add_data_type(self, data_type):
        """adds a DataType object to the output allowed_data_types list"""
        if data_type not in self.allowed_data_types:
            self.data_types.append(data_type)
            Log.info(f"Added data type {data_type} to output {self.name}")
        else:
            Log.error(f"Data type {data_type} already exists in output {self.name}")

    def set_data_type(self, data_type):
        if data_type in self.data_types:
            self.data_type = data_type
            Log.info(f"Block {self.parent_block.name} output data type updated to: {data_type.name}")
        else:
            Log.error(f"Invalid data type: {data_type.name}. Valid data types: {self.allowed_data_types}")


    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type.name
        }

    def load(self, data):
        self.name = data.get("name")
        self.type = data.get("type")
        self.data_type = data.get("data_type")
