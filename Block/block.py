from Command.command_controller import CommandController
from message import Log
from tools import gtimer
from Port.port_controller import PortController
from Data.data_controller import DataController
from abc import ABC, abstractmethod
from Save.save_controller import SaveController

class Block(ABC):
    def __init__(self):
        Log.info(f"Creating Instance of the Block Object")
        self.name = None
        self.type = None
        self.data = None
        self.parent = None  # Reference to project

        self.port = PortController(self) # Port controller for input and output ports with parent reference to self
        self.data = DataController()

        self.save = SaveController(self)
        self.save.add_attribute("name")
        self.save.add_attribute("type")
        self.save.add_attribute("data")
        self.save.add_sub_module("port")

        self.command = CommandController()
        self.command.add("reload", self.reload)

 ## CORE BLOCK PROCESS METHODS

    def reload(self):
        timer = gtimer()
        timer.start()
        Log.info(f"Reloading block {self.name}")
        input_data = self.port.pull_all()
        results = self.process(input_data)
        if results:
            self.set_data(results)
            Log.info(f"Reloaded block {self.name} with results: {results}")
        Log.info(f"Reloaded block {self.name} in {timer.end():.2f} seconds")

    @abstractmethod
    def process(self, input_data):
        # This method should be implemented by the subclass to carry out bespoke functionality 
        pass

## CORE BLOCK SET METHODS

    def set_name(self, name):
        self.name = name
        Log.info(f"Updated Blocks name to: '{name}'")   

    def set_type(self, type):
        self.type = type
        Log.info(f"Set type: {type}")

    def set_data(self, data):
        self.data = data
        if isinstance(self.data, list):
            Log.info(f"Block '{self.name}' 'data' set to a list with {len(self.data)} items")
            for item in self.data:
                if hasattr(item, 'name'):
                    Log.info(f"- {item.name}")
                else:
                    Log.info(f"- {item}")
        elif hasattr(self.data, 'name'):
            Log.info(f"Block '{self.name}' 'data' value set to: {self.data.name}")
        else:
            Log.info(f"Block '{self.name}' 'data' value set to: {self.data}")

    def set_parent(self, parent):
        self.parent = parent
        Log.info(f"Set Block {self.name} parent object to: {parent.name}")
