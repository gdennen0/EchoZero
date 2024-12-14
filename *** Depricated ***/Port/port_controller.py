from Utils.tools import prompt_selection, prompt, generate_unique_name
from Utils.message import Log
from Command.command_controller import CommandController
from Port.port import Port
from Attribute.attribute_controller import AttributeController

class PortController():
    def __init__(self, parent_block):
        # Attribute Controller
        self.attribute = AttributeController()
        self.attribute.add("parent_block", None)
        self.attribute.add("name", None)
        self.attribute.add("port_types", [])
        self.attribute.add("ports", [])

        # Set attributes
        self.attribute.set("name","port_controller")
        self.attribute.set("parent_block", parent_block)

        # Command Controller
        self.command = CommandController()
        self.command.add("list", self.list)
        self.command.add("create_connection", self.create_connection)
        self.command.add("list_connections", self.list_connections)
        self.command.add("list_ports", self.list_ports)
        self.command.add("add", self.add_port_type)

    def set_parent_block(self, parent_block):
        self.attribute.set("parent_block", parent_block)
        Log.info(f"Set port controller parent block to {self.attribute.get('parent_block').attribute.get('name')}")

    def add_port_type(self, type):
        self.attribute.append("port_types", type)
        Log.info(f"Added port type {type.name}")

    def add(self, name, type=None):
        if not type:
            type = prompt_selection(f"Enter the type of the port: ", ["input", "output"])
        for port_type in self.attribute.get("port_types"):
            Log.info(f"Checking {name} vs Port type: {port_type.name}")
            if port_type.name == name:
                # Check for existing port with the same name
                if any(port.attribute.get("name") == name for port in self.attribute.get("ports")):
                    counter = 1
                    new_name = f"{name}{counter}"
                    while new_name in [port.attribute.get("name") for port in self.attribute.get("ports")]:
                        counter += 1
                        new_name = f"{name}{counter}"
                    name = new_name
                
                new_port = port_type()
                new_port.attribute.set("name", name)
                new_port.attribute.set("type", type)
                new_port.attribute.set("parent_block", self.attribute.get("parent_block"))
                self.attribute.append("ports", new_port)
                Log.info(f"Added new port: {name} of type: {type}")
                return
            
        Log.error(f"(add) Port type {name} with type {type} not found in registered port types")

    def add_input(self, name):
        input_name = generate_unique_name(name, self.attribute.get("ports"))
        self.add(input_name, type="input")

    def add_output(self, name):
        output_name = generate_unique_name(name, self.attribute.get("ports"))
        self.add(output_name, type="output")

    def list_connections(self):
        for port in self.attribute.get("ports"):
            Log.info(f"Port: {port.attribute.get('name')}, type: {port.attribute.get('type')}, connections: {[connection.attribute.get('parent_block').attribute.get('name') + '.' + connection.attribute.get('type') + '.' + connection.attribute.get('name') for connection in port.connections] if port.connections else 'No connections'}")

    def pull_all(self):
        pulled_data = []
        for port in self.attribute.get("ports"):
            if port.attribute.get("type") == "input":
                port_data = port.pull()
                pulled_data.append(port_data)
        return pulled_data
    
    def push_all(self):
        for port in self.attribute.get("ports"):
            Log.info(f"Pushing Port: {port.attribute.get('name')}, type: {port.attribute.get('type')}")
            if port.attribute.get("type") == "output":
                port.push()

    def list(self):
        counter = 1
        Log.info("Listing ports")
        for port in self.attribute.get("ports"):
            Log.info(f"Port {counter}: {port.attribute.get('name')} ({port.attribute.get('type')})")
            counter += 1 
        Log.info("End of port list")

    def get_port(self, name, type):
        for port in self.attribute.get("ports"):
            if port.attribute.get("name") == name and port.attribute.get("type") == type:
                return port
        return None 

    def create_connection(self):
        local_ports = []
        for port in self.get_ports():
            if port.attribute.get("type") == "input":
                local_ports.append(port)
        if local_ports:
            Log.info(f"Select which {self.attribute.get('parent_block').attribute.get('name')} input port you want to connect to an external port")
            local_port = self.prompt_ports(local_ports, "input")
            Log.info(f"selected block {self.attribute.get('parent_block').attribute.get('name')} input port: {local_port.attribute.get('name')}")
        else:
            Log.error("There are no local input ports to create a connection from")
            return

        external_ports = []
        for block in local_port.attribute.get("parent_block").attribute.get("parent").attribute.get("blocks"):
            if block.attribute.get("name") != local_port.attribute.get("parent_block").attribute.get("name"):
                for external_port in block.port.get_ports():
                    if external_port.attribute.get("type") == "output":
                        external_ports.append(external_port)

        if external_ports:
            external_port = prompt_selection(f"Select output port to connect {local_port.attribute.get('name')} to: ", external_ports )
        else:
            Log.error("There are no output external ports")
            return
        
        if local_port and external_port:
            local_port.connect(port=external_port)

    def list_ports(self): 
        for port in self.attribute.get("ports"):
            Log.info(f"Port: {port.attribute.get('name')}, type: {port.attribute.get('type')}")

    def items(self):
        return self.attribute.get("ports")

    def get_ports(self):
        return self.attribute.get("ports")
    
    def prompt_ports(self, port_list, type):
        Log.info(f"Select {type} port ")
        for counter, port in enumerate(port_list):
            Log.info(f"{counter}: {port.attribute.get('parent_block').attribute.get('name')} - {port.attribute.get('name')} ({port.attribute.get('type')})")
        
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

    def set_name(self, name):
        self.attribute.set("name", name)
        Log.info(f"Set port controller name to {name}")