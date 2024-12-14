from Utils.message import Log
from Utils.tools import prompt, prompt_selection, prompt_selection_with_type_and_parent_block
from Command.command_controller import CommandController
from Data.data_controller import DataController
from Save.save_controller import SaveController
from Attribute.attribute_controller import AttributeController
from abc import abstractmethod

class Port:
    def __init__(self):
        self.attribute = AttributeController()
        self.attribute.add("name", None)
        self.attribute.add("type", None)
        self.attribute.add("parent", None)
        self.attribute.add("parent_block", None)
        self.attribute.add("data_types", [])
        self.attribute.add("data_type", None)
        
        self.connected_output = []

        self.data = DataController() # initialize a data controller
        self.data.set_parent(self)

        self.command = CommandController() # initialize a command controller
        self.command.add("set_name", self.set_name)
        self.command.add("list_connections", self.list_connections)
        self.command.add("create_connection", self.connect)
        
    def set_parent_block(self, block):
        self.attribute.set("parent_block", block)
        Log.info(f"Port {self.attribute.get('name')} parent block set to: {block.attribute.get('name')}")

    def set_name(self, name=None):
        if name:
            self.attribute.set("name", name)
            Log.info(f"Port name set to: {name}")
        else:
            self.attribute.set("name", prompt(f"Please enter the name of the port: "))
            Log.info(f"Port name set to: {self.attribute.get('name')}")

    def set_type(self, type=None):
        types = ["input", "output"]
        if type:
            if type in types:   
                self.attribute.set("type", type)
                Log.info(f"Port type set to: {type}")
            else:
                Log.error(f"Invalid port cannot set type to invalid port type'{type}'")
        else:
            self.attribute.set("type", prompt_selection(f"Please enter the type of the port: ", types))
            Log.info(f"Port type set to: {self.attribute.get('type')}")

    def set_data_type(self, data_type):
        self.attribute.set("data_type", data_type)
        Log.info(f"Port data type set to: {data_type}")

    def pull(self):
        if self.attribute.get("type") == "input":
            self.data.clear() #clear the ports data before adding new data
            for connected_port in self.connections:
                data_items = connected_port.data.get()
                if data_items:
                    for data_item in data_items:
                        Log.info(f"Pulling data item {data_item.attribute.get('name')} of type {data_item.attribute.get('type')}")
                        if data_item.attribute.get("type") == self.attribute.get("data_type"): # Ensure the pulled data is of the correct type
                            self.data.add(data_item)
                    Log.info(f"Pulled data from {connected_port.attribute.get('parent_block').attribute.get('name')}.{connected_port.attribute.get('type')}.{connected_port.attribute.get('name')} to {self.attribute.get('parent_block').attribute.get('name')}.{self.attribute.get('type')}.{self.attribute.get('name')}")
                else:
                    Log.error(f"No data found in {connected_port.attribute.get('parent_block').attribute.get('name')}.{connected_port.attribute.get('type')}.{connected_port.attribute.get('name')}")
        else:
            Log.error("You can only run pull with an input port")

    def push(self):
        if self.type == "output":
            for data_item in self.attribute.get("parent_block").data.get():
                Log.info(f"Checking type {data_item.attribute.get('type')} if it matches {self.attribute.get('data_type')}")
                if data_item.attribute.get("type") == self.attribute.get("data_type"):
                    self.data.add(data_item)
                    Log.info(f"Pushed data item {data_item.attribute.get('name')} of type {data_item.attribute.get('type')} to {self.attribute.get('parent_block').attribute.get('name')}.{self.attribute.get('type')}.{self.attribute.get('name')}")
                else:
                    Log.error(f"Pushed data item {data_item.attribute.get('name')} of type {data_item.attribute.get('type')} to {self.attribute.get('parent_block').attribute.get('name')}.{self.attribute.get('type')}.{self.attribute.get('name')} but it is not of the correct data type {self.attribute.get('data_type')        }")
        else:
            Log.error("You can only run push with an output port")

    def connect(self, port=None):
        if port.attribute.get("type") == "output":
            if port not in self.connected_output:
                self.connected_output.append(port)
                Log.info(f"Connected '{self.attribute.get('parent_block').attribute.get('name')}' {self.attribute.get('type')} port '{self.attribute.get('name')}' <--> '{port.attribute.get('parent_block').attribute.get('name')}' {port.attribute.get('type')} port '{port.attribute.get('name')}'")

        else:
            Log.error(f"Error connecting Block {self.attribute.get('parent_block').attribute.get('name')} port {self.attribute.get('name')}. Connection must be an output port")
            
    def list_connections(self):
        for connected_port in self.connections:
            Log.info(f"Block '{connected_port.attribute.get('parent_block').attribute.get('name')}' Port '{connected_port.attribute.get('name')}' <--> Block '{self.attribute.get('parent_block').attribute.get('name')}' Port '{self.attribute.get('name')}'")

    def items(self):
        return self.connections
    
    


