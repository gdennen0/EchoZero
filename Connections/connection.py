from message import Log
from tools import prompt

class Connection:
    def __init__(self, port1, port2):
        self.port1 = port1
        self.port2 = port2
        self.name = "Connection"
        self.description = f"{port1.name} -> {port2.name}"

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

    def set_port1(self, port1):
        self.port1 = port1
        self.description = f"{port1.name} -> {self.port2.name}"
        Log.info(f"'{self.name}' Connection input port set to: {port1.name}")

    def set_output_port(self, port2):
        self.port2 = port2
        self.description = f"{self.port1.name} -> {port2.name}"
        Log.info(f"'{self.name}' Connection output port set to: {port2.name}")

    

