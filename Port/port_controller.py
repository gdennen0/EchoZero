from tools import prompt_selection, prompt
from message import Log
from Command.command_controller import CommandController
from Port.Connection.connection import Connection
from Port.port import Port
from Save.save_controller import SaveController

class PortController():
    def __init__(self, parent_block):
        self.set_parent_block(parent_block)
        self.name = "port_controller"
        self.port_types = []
        self.ports = []
        self.command = CommandController()

        self.command.add("list", self.list)
        self.command.add("create_connection", self.create_connection)
        self.command.add("list_ports", self.list_ports)
        self.command.add("add", self.add_port_type)

        self.save = SaveController(self)
        self.save.add_sub_module("ports")

    def set_parent_block(self, parent_block):
        self.parent_block = parent_block
        Log.info(f"Set port controller parent block to {self.parent_block.name}")

    def add_port_type(self, type):
        self.port_types.append(type)
        Log.info(f"Added port type {type.name}")

    def add(self, name, type=None):
        if not type:
            type = prompt_selection(f"Enter the type of the port: ", ["input", "output"])
        for port_type in self.port_types:
            Log.info(f"Checking {name} vs Port type: {port_type.name}")
            if port_type.name == name:
                # Check for existing port with the same name
                if any(port.name == name for port in self.ports):
                    counter = 1
                    new_name = f"{name}{counter}"
                    while new_name in [port.name for port in self.ports]:
                        counter += 1
                        new_name = f"{name}{counter}"
                    name = new_name
                
                new_port = port_type()
                new_port.set_name(name=name)
                new_port.set_type(type=type)
                new_port.set_parent_block(self.parent_block)
                self.ports.append(new_port)
                Log.info(f"Added new port: {name} of type: {type}")
                return
            
        Log.error(f"(add) Port type {name} with type {type} not found in registered port types")

    def add_input(self, name):
        self.add(name, type="input")

    def add_output(self, name):
        self.add(name, type="output")

    def pull_all(self):
        pulled_data = []
        for port in self.ports:
            pulled_data.append(port.data.get())
        return pulled_data

    def list(self):
        counter = 1
        Log.info("Listing ports")
        for port in self.ports:
            Log.info(f"Port {counter}: {port.name} ({port.type})")
            counter += 1 
        Log.info("End of port list")

    def create_connection(self):
        selection_type = prompt_selection("Select local port type:", ["input", "output"])
        
        if selection_type == "input":
            local_ports = []
            for port in self.get_ports():
                if port.type == "input":
                    local_ports.append(port)
            if local_ports:
                local_port = self.prompt_ports(local_ports, "input")
                Log.info(f"selected local port: {local_port.type}.{local_port.name}")
            else:
                Log.error("There are no local input ports!")
                return

            external_ports = []
            for block_name, block in local_port.parent_block.parent_container.blocks:
                if block_name != self.parent_block.name:
                    for external_port in block.port.get_ports():
                        if external_port.type == "output":
                            external_ports.append(external_port)

            if external_ports:
                external_port = prompt_selection("Select external port:", external_ports )
            else:
                Log.error("There are no output external ports")
                return

        elif selection_type == "output":
            local_ports = []
            for port in self.get_ports():
                if port.type == "output":
                    local_ports.append(port)
            if local_ports:
                local_port = self.prompt_ports(local_ports, "output")
                Log.info(f"selected local port: {local_port.type}.{local_port.name}")
            else:
                Log.error("There are no local output ports!")
                return

            external_ports = []
            for block in local_port.parent_block.parent.blocks:
                if block.name != self.parent_block.name:
                    for external_port in block.port.get_ports():
                        if external_port.type == "input":
                            external_ports.append(external_port)

            if external_ports:
                external_port = self.prompt_ports(external_ports, "input")
            else:
                Log.error("There are no external ports")
                return
        else:
            Log.error("Invalid port type, must be input or output")
            return
        
        if local_port and external_port:
            connection = Connection()
            if local_port.type == "input":  
                connection.set_input_port(local_port)
                connection.set_output_port(external_port)
                Log.info(f"Connected local input port {local_port.name} to external output port {external_port.name}")
            elif local_port.type == "output":
                connection.set_output_port(local_port)
                connection.set_input_port(external_port)
                Log.info(f"Connected local output port {local_port.name} to external input port {external_port.name}")
                local_port.add_connection(connection)
                external_port.add_connection(connection)
    

    def items(self):
        return self.ports
    
    def list_ports(self): 
        for port in self.ports:
            Log.info(f"Port: {port.name}, type: {port.type}")

    def get_ports(self):
        return self.ports
    
    def prompt_ports(self, port_list, type):
        Log.info(f"Select {type} port ")
        for counter, port in enumerate(port_list):
            Log.info(f"{counter}: {port.parent_block.name} - {port.name} ({port.type})")
        
        while True:
            selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
            if not selection: 
                Log.info("Selection exited by user.")
                return None, None
            if selection.isdigit():
                index = int(selection)
                if 0 <= index < len(port_list):
                    return port_list[index]
            elif selection in port_list:
                return port_list[selection]
            Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

