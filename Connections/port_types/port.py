from ..connection import Connection
from message import Log
from tools import prompt, prompt_selection, prompt_selection_with_type_and_parent_block
from command_module import CommandModule

class Port(CommandModule):
    def __init__(self):
        super().__init__()
        Log.info(f"Creating Instance of the Port Object")
        self.type = None
        self.port_types = ["input", "output"]
        self.name = None
        self.data = None
        self.data_type = None
        self.connections = []
        self.parent_block = None


        self.add_command("set_name", self.set_name)
        self.add_command("set_type", self.set_type)
        self.add_command("link_attribute", self.link_attribute)
        self.add_command("list_connections", self.list_connections)
        self.add_command("create_connection", self.create_connection)

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

    def set_type(self, port_type = None):
        if port_type:
            if port_type in self.port_types:
                self.type = port_type
                Log.info(f"Port type set to: {port_type}")
            else:
                port_type = prompt_selection(f"Please select a valid port type: ", self.port_types)
                self.type = port_type
                Log.info(f"Port type set to: {port_type}")
        else:
            port_type = prompt_selection(f"Please select a valid port type: ", self.port_types)
            self.type = port_type
            Log.info(f"Port type set to: {port_type}")


    def link_attribute(self, name=None): # use this to link to an attribute of a block (only works for attributes of the parent block)
        try:
            if name:
                self.data = getattr(self.parent_block, name)
            else:
                name = prompt(f"Please enter the name of the attribute to link to: ")
                self.data = getattr(self.parent_block, name)

        except AttributeError:
            Log.error(f"Attribute '{name}' does not exist in parent block '{self.parent_block.name}'.")
        except Exception as e:
            Log.error(f"An error occurred while linking attribute: {e}")

    def set_data_type(self, data_type_name=None):
        if data_type_name:
            self.data_type = data_type_name
            Log.info(f"Data type set to: {data_type_name}")
        else:
            data_type_name = prompt(f"Please enter the name of the data type: ")
            self.data_type = data_type_name
            Log.info(f"Data type set to: {data_type_name}")

    def create_connection(self):
        external_ports = []
        for block_name, block in self.parent_block.parent_container.blocks.items():
            for input_port in block.input_ports:
                if input_port not in self.parent_block.input_ports:
                    external_ports.append(input_port)
            for output_port in block.output_ports:
                if output_port not in self.parent_block.output_ports:
                    external_ports.append(output_port)
        if not external_ports:
            Log.error(f"No external ports found to connect to")
            return
        connecting_port, _ = prompt_selection_with_type_and_parent_block(f"Please select the port to connect to: ", external_ports)
        connection = Connection(self, connecting_port)
        self.add_connection(connection)

    def add_connection(self, connection):
        if connection:
            self.connections.append(connection)
            Log.info(f"'{self.name}' Port connected to '{connection.port2.name}' Port")
        else:
            Log.error(f"Error adding connection to Block {self.parent_block.name} port {self.name}")

    def disconnect(self, connection):
        self.connections.remove(connection) 
        Log.info(f"'{self.name}' Port disconnected from '{connection.port2.name}' Port")

    def list_connections(self):
        for connection in self.connections:
            Log.info(f"Connection: {connection.port1.parent_block.name}:{connection.port1.type}>{connection.port1.name} <-> {connection.port2.parent_block.name}:{connection.port2.type}>{connection.port2.name}")


