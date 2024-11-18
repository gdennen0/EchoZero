from message import Log
from tools import prompt

class Connection:
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.name = "Connection"
        self.description = f"{input.name} -> {output.name}"

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

    def set_input(self, input):
        self.input = input
        self.description = f"{input.name} -> {self.output.name}"
        Log.info(f"'{self.name}' Connection input port set to: {input.name}")

    def set_output(self, output ):
        self.output = output
        self.description = f"{self.input.name} -> {output.name}"
        Log.info(f"'{self.name}' Connection output port set to: {output.name}")

    

