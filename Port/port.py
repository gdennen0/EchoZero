from Port.Connection.connection import Connection
from message import Log
from tools import prompt, prompt_selection, prompt_selection_with_type_and_parent_block
from Command.command_controller import CommandController
from Data.data_controller import DataController
from Save.save_controller import SaveController
## SOMETHING TO DO WITH THE TYPES NOT BEING SET PROPERLY?
## FUCK TYPES 

class Port():
    def __init__(self):
        self.data = DataController() # initialize a data controller
        self.command = CommandController() # initialize a command controller

        self.type = None  # Input or output

        self.data_types = []
        self.name = None
        self.parent = None
        self.parent_block = None
        self.connections = []

        self.command.add("set_name", self.set_name)
        self.command.add("list_connections", self.list_connections)
        self.command.add("create_connection", self.connect)

        self.save = SaveController(self)
        self.save.add_attribute("name")
        self.save.add_attribute("parent_block.name")
        self.save.add_attribute("type")
        self.save.add_sub_module("connections")

    def set_address(self):
        if self.parent_block and self.name and self.type:
            self.address = self.parent_block.name + "." + self.type + "." + self.name
        else:
            Log.error("Port parent block not set")
        
    def set_parent_block(self, block):
        self.parent_block = block
        Log.info(f"Port {self.name} parent block set to: {block.name}")

    def set_name(self, name=None):
        if name:
            self.name = name
            Log.info(f"Port name set to: {name}")
        else:
            self.name = prompt(f"Please enter the name of the port: ")
            Log.info(f"Port name set to: {self.name}")

    def set_type(self, type=None):
        types = ["input", "output"]
        if type:
            if type in types:   
                self.type = type
                Log.info(f"Port type set to: {type}")
            else:
                Log.error(f"Invalid port cannot set type to invalid port type'{type}'")
        else:
            self.type = prompt_selection(f"Please enter the type of the port: ", types)
            Log.info(f"Port type set to: {self.type}")

    def pull(self):
        if self.type == "input":
            if len(self.connections) == 1:
                for connection in self.connections:
                    data = connection.output_port.data.get()
                    Log.info(f"Pulled data from {connection.output_port.parent_block.name} port {connection.output_port.name}")
                    return data
            else:
                Log.error("There are too many connections for this input port")
        else:
            Log.error("You can only run pull with an input port")

    def connect(self, port=None, port_path=None):
        """
        args:
            port: Port object to connect to
            port_path: a string path of an external port within the parent container "BlockName, PortType, PortName"
        """
        if port:
            if self.type == "input" and port.type == self.type:
                connection = Connection()
                connection.set_input_port(self)
                connection.set_output_port(port)
                connection.set_parent(self)
                self.connections.append(connection)
                Log.info(f"Block '{port.parent_block.name}' Input Port '{port.name}' connected to Block '{self.parent_block.name}' Output Port '{self.name}'")
            elif self.type == "output" and port.type == self.type:
                connection = Connection()
                connection.set_input_port(port)
                connection.set_output_port(self)
                connection.set_parent(self)
                self.connections.append(connection)
                Log.info(f"Block '{port.parent_block.name}' Output Port '{port.name}' connected to Block'{self.parent_block.name}' Input Port '{self.name}'")
            else:
                Log.error(f"Error adding connection to Block {self.parent_block.name} port {self.name} please pass connection item")

        if port_path:
            # match the port_name argument ("BlockName","PortType","PortName") to the port object
            try:
                block_name, port_type, port_name = port_path.split(' ')
            except ValueError:
                Log.error(f"port_name '{port_path}' is not in the format 'BlockName PortType PortName'")
                return
            # try to match input to an item within parent container   
            for block in self.parent_block.parent_container.blocks:
                extern_block_name = block.name
                if extern_block_name == block_name:
                    for port in block.ports:
                        if port.type == port_type:  
                            if port.name == port_name:
                                self.connect(port)
                                break
            
            Log.error(f"Block {self.parent_block.name} port {self.name} connect method input 'port_path' arg parts invalid {block_name} {port_type} {port_name}")
        else:
            external_blocks = []
            for block in self.parent_block.parent_container.blocks:
                for port in block.ports:
                    if self.type == "input":
                        if port.type == "output":
                            external_blocks.append(port)
                    elif self.type == "output":
                        if port.type == "input":
                            external_blocks.append(port)
            
            connecting_port = prompt_selection_with_type_and_parent_block(f"Select the port to connect to {self.parent_block.name} port {self.name}", external_blocks)
            self.connect(connecting_port)

    def add_connection(self, connection):
        if connection not in self.connections:
            self.connections.append(connection)
        else:
            Log.error(f"Connection '{connection.name}' already exists within block '{self.parent_block.name}' port '{self.name}' connections")
            
    def list_connections(self):
        for connection in self.connections:
            Log.info(f"Block '{connection.input.parent_block.name} Port '{connection.input.name}' <--> Block '{connection.output.parent_block.name}' Port '{connection.output.name}'")

    def items(self):
        return self.connections