from command_module import CommandModule
from message import Log
from abc import ABC

class Part(CommandModule, ABC): # Inherit CommandModule and ABC (Abstract Base Class)
    def __init__(self):
        super().__init__()
        self.name = None
        self.type = None
        self.input_types = []
        self.output_types = []

    def set_name(self, name):
        self.name = name
        Log.info(f"Set name: {name}")

    def set_type(self, type):
        self.type = type
        Log.info(f"Set type: {type}")

    def list_input_types(self):
        Log.list("Input Types", self.input_types)

    def list_output_types(self):
        Log.list("Output Types", self.output_types)

    def add_input_type(self, input_type):
        self.input_types.append(input_type)
        Log.info(f"Added input type: {input_type}")

    def add_output_type(self, output_type):
        self.output_types.append(output_type)
        Log.info(f"Added output type: {output_type}")
