from message import Log
from tools import prompt

class Connection:
    def __init__(self, input_port, output_port):
        self.input_port = input_port
        self.output_port = output_port
        self.name = "Connection"
        self.description = f"{input_port.name} -> {output_port.name}"

    def set_name(self, name=None):
        if name:
            self.name = name
        else:
            self.name = prompt(f"Please enter a name for the connection: ")
            Log.info(f"Connection name set to: {self.name}")
        
    def set_description(self, description=None):
        if description:
            self.description = description
        else:
            self.description = prompt(f"Please enter a description for the connection: ")
            Log.info(f"Connection description set to: {self.description}")

    def set_input_port(self, input_port):
        self.input_port = input_port
        self.description = f"{input_port.name} -> {self.output_port.name}"
        Log.info(f"'{self.name}' Connection input port set to: {input_port.name}")

    def set_output_port(self, output_port):
        self.output_port = output_port
        self.description = f"{self.input_port.name} -> {output_port.name}"
        Log.info(f"'{self.name}' Connection output port set to: {output_port.name}")

    

