from command_module import CommandModule
from message import Log
from abc import ABC, abstractmethod
from Block.part import Part  # Assuming Part is defined in Block/part.py
from tools import prompt_selection, prompt_selection_with_type, prompt_selection_with_type_and_parent_block
from Connections.port_types.port import Port
from Connections.connection import Connection

class Block(CommandModule, ABC):
    def __init__(self):
        super().__init__()
        Log.info(f"Creating Instance of the Block Object")
        self.name = None
        self.type = None
        self.parts = []
        self.part_types = []
        self.input_types = []
        self.output_types = []
        self.data = None
        self.parent_container = None  # Reference to parent container

        self.port_types = []
        self.input_ports = []
        self.output_ports = []
    
        self.add_command("add_part", self.add_part)
        self.add_command("remove_part", self.remove_part)
        self.add_command("list_parts", self.list_parts)
        self.add_command("list_ports", self.list_ports)
        self.add_command("create_connection", self.create_connection)
        self.add_command("list_connections", self.list_connections)

    def set_parent_container(self, container):
        self.parent_container = container
        Log.info(f"Set container: {container.name}")

    def start(self, input_data):
        Log.info("Start method is depriciated, passing")
        pass

        if not self.input_ports:
            Log.error("No input ports available.")
            return None
        Log.info(f"Starting block {self.name}")
        if self._validate_input(input_data):
            result = self._process_parts(input_data)
            if self._validate_output(result):
                return result
            else:
                Log.error(f"Result {result} not in output types {self.output_types}")
                return None
        else:
            Log.error(f"Input data {input_data} is not a valid input type.")
            return None
        
    # def reload(self):
    #     Log.info(f"Reloading block {self.name}")
    #     check if input_port exists
    #     grab the data from input_port.output_ports
    #     iterate through the ports 
    #     check if the port type is in the input_types list
    #     pass

    def create_connection(self, port1=None , port2=None):
        # Start of Selection
        if not port1:
            combined_ports = self.input_ports + self.output_ports
            port1, _ = prompt_selection_with_type("Please select a local port to connect: ", combined_ports)

        if not port2:
            if port1.type == "input":
                external_ports = []
                for block_name, block in self.parent_container.blocks.items():
                    for output_port in block.output_ports:
                        if output_port not in self.output_ports:
                            external_ports.append(output_port)
            elif port1.type == "output":
                external_ports = []
                for block_name, block in self.parent_container.blocks.items():
                    for input_port in block.input_ports:
                        if input_port not in self.input_ports:
                            external_ports.append(input_port)
            if external_ports:
                port2, _ = prompt_selection_with_type_and_parent_block(f"Please select an external port to connect to Block {self.name}", external_ports)
            else:
                Log.error("No external ports available.")
                return
            
        if port1 and port2:
            if port1.type == port2.type:
                Log.error("Cannot connect ports of the same type")
                self.connect()
            else:
                connection = Connection(port1, port2)
                port1.add_connection(connection)
                port2.add_connection(connection)
                Log.info(f"Connected {port1.name} to {port2.name}")
        else:
            if not self_port:
                self_port, _ = prompt_selection(f"Please select a port to connect to Block {self.name}", self.ports)
            if not other_port:
                other_port, _ = prompt_selection(f"Please select a port to connect to Block {self.name}", self.ports)
            self.connect(self_port, other_port)
            Log.info(f"Connected Block '{self.name}' port '{self_port.name}' to Block '{other_port.parent_block.name}' port '{other_port.name}'")


    def list_connections(self):
        Log.info(f"Block {self.name} connections:")
        for input_port in self.input_ports:
            for connection in input_port.connections:
                Log.info(f"- Block {self.name}:{input_port.name} connected to Block {connection.output_port.parent_block.name}:{connection.output_port.name}")
        for output_port in self.output_ports:
            for connection in output_port.connections:
                Log.info(f"- Block {self.name}:{output_port.name} connected to Block {connection.input_port.parent_block.name}:{connection.input_port.name}")

    def add_port_type(self, port_type_class):
        self.port_types.append(port_type_class)
        Log.info(f"Added port type: {port_type_class.name} to Block {self.name}")

    def list_ports(self):
        inputs = [port.name for port in self.input_ports]
        outputs = [port.name for port in self.output_ports]
        
        max_length = max(len(inputs), len(outputs))
        inputs += [''] * (max_length - len(inputs))
        outputs += [''] * (max_length - len(outputs))
        
        Log.info(f"{'Block ' + self.name:^41}")
        Log.info(f"{'[INPUTS]':<20} {'[OUTPUTS]':<20}")

        port_number = 0 
        for input_port, output_port in zip(inputs, outputs):
            Log.info(f"{port_number:<2} {input_port:<16} {port_number:<2} {output_port:<16}")
            port_number += 1

    
    def add_input_port(self, port_name=None):
        if not port_name:
            port_name, _ = prompt_selection(f"Please select a port type to add to Block '{self.name}': ", self.port_types)
        for port_type in self.port_types:
            if port_type.name == port_name:
                port_object = port_type()
                port_object.set_type("input")
                port_object.set_parent_block(self)
                self.input_ports.append(port_object)
                Log.info(f"Added input port: {port_object.name} to Block {self.name}")
                return
    
    def add_output_port(self, port_name=None):
        if not port_name:
            port_name, _ = prompt_selection(f"Please select a port type to add to Block {self.name} ", self.port_types)
        for port_type in self.port_types:
            if port_type.name == port_name:
                port_object = port_type()
                port_object.set_type("output")
                port_object.set_parent_block(self)
                self.output_ports.append(port_object)
                Log.info(f"Added output port: {port_object.name} to Block {self.name}")
                return
            
    def link_port_attribute(self, port_type, port_name, attribute_name):
        Log.info(f"Linking block {self.name} attribute {attribute_name} to {port_type} port {port_name} ")
        if port_type == "input":
            for port in self.input_ports:
                if port.name == port_name:
                    port.link_attribute(name=attribute_name)
                    return
                else:
                    Log.error(f"Port {port_name} not found in input ports list.")
                    return
                
        elif port_type == "output":
            for port in self.output_ports:
                if port.name == port_name:
                    port.link_attribute(name=attribute_name)
                    return
                else:
                    Log.error(f"Port {port_name} not found in output ports list.")
                    return
        else:
            Log.error(f"Invalid port type: {port_type}")
            return
        
    def _validate_input(self, input_data):
        if input_data in self.input_types:
            return True
        else:
            Log.error(f"Input data {input_data} is not a valid input type.")
            return False

    def _process_parts(self, input_data):
        result = input_data
        for part in self.parts:
            Log.info(f"Processing with part {part.name} in Block {self.name}")
            result = part.start(result)
            if result is None:
                Log.error(f"Part {part.name} failed to process the data.")
                return None
        return result

    def _validate_output(self, result):
        return result in self.output_types

    def set_name(self, name):
        self.name = name
        Log.info(f"Updated Blocks name to: '{name}'")   

    def set_type(self, type):
        self.type = type
        Log.info(f"Set type: {type}")

    def add_part(self, part_name=None):
        if part_name: # if part name is specified, add the part type with that name string
            part_type = str(part_name)
            for part_type in self.part_types: # iterate through part types
                Log.info(f"Checking part type name '{part_type.name}' against '{part_name}'")
                if part_type.name == part_name: 
                    self.parts.append(part_type)
                    Log.info(f"Added part: {part_name} to Block {self.name}")
                    return      
            Log.error(f"Part type {part_name} not found in part types list.")
        else:
            part_type, _ = prompt_selection("Please select a part type to add: ", self.part_types)
            self.parts.append(part_type)
            Log.info(f"Added part: {part_type.name} to Block {self.name}")

    def remove_part(self, part_name=None):
        if part_name:
            for part in self.parts:
                if part.name == part_name:
                    self.parts.remove(part)
                    Log.info(f"Removed part: {part_name} from Block {self.name}")
                    return
            Log.error(f"Part {part_name} not found in parts list.")
        else:
            part_type, _ = prompt_selection("Please select a part to remove: ", self.parts)
            self.parts.remove(part_type)
            Log.info(f"Removed part: {part_type.name} from Block {self.name}")
            part_type, _ = prompt_selection("Please select a part to remove: ", self.parts)
            self.parts.remove(part_type)
            Log.info(f"Removed part: {part_type.name} from Block {self.name}")

    def list_parts(self):
        Log.info(f"Block {self.name} current parts:")
        for part in self.parts:
            Log.info(f"- {part.name} ({part.type})")
        return self.parts

    def clear_parts(self):
        self.parts = []
        Log.info("Cleared all parts")

    def add_part_type(self, part_type):
        self.part_types.append(part_type)
        Log.info(f"Added part type: '{part_type}'")

    def remove_part_type(self, part_type):
        self.part_types.remove(part_type)
        Log.info(f"Removed part type: {part_type}")

    def list_part_types(self):
        Log.info(f"Block '{self.name}' part types")
        for part_type in self.part_types:
            Log.info(f"- {part_type}")
        return self.part_types

    def add_input_type(self, input_type):
        self.input_types.append(input_type)
        Log.info(f"Added input type: '{input_type}'")

    def remove_input_type(self, input_type):
        if input_type in self.input_types:
            self.input_types.remove(input_type)
            Log.info(f"Removed input type: '{input_type}'")
        else:
            Log.error(f"Input type '{input_type}' not found in input types list.")

    def add_output_type(self, output_type):
        self.output_types.append(output_type)
        Log.info(f"Added output type: {output_type.name}")

    def list_output_types(self):
        Log.info(f"Block '{self.name}' output types")
        for output_type in self.output_types:
            Log.info(f"- {output_type}")
        return self.output_types

    def list_input_types(self):
        Log.info(f"Block '{self.name}' input types")
        for input_type in self.input_types:
            Log.info(f"- {input_type}")
        return self.input_types
    
    def set_data(self, data):
        self.data = data
        Log.info(f"Block '{self.name}' 'data' value set to: {data.name}")