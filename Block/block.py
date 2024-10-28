# Base Block class
from command_module import CommandModule
from message import Log

class Block(CommandModule):
    def __init__(self):
        super().__init__()
        self.name = None
        self.type = None
        self.parts = []
        self.part_types = []
        self.input_types = []
        self.add_command("add_part", self.AddPart)
        self.add_command("remove_part", self.RemovePart)
        self.add_command("list_parts", self.ListParts)

    def AddPart(self, part):
        self.parts.append(part)
        Log.info(f"Added part: {part}")

    def RemovePart(self, part):
        self.parts.remove(part)
        Log.info(f"Removed part: {part}")

    def ListParts(self):
        Log.info("Listing parts")
        return self.parts

    def ClearParts(self):
        self.parts = []
        Log.info("Cleared all parts")

    def add_part_type(self, part_type):
        self.part_types.append(part_type)
        Log.info(f"Added part type: {part_type}")

    def remove_part_type(self, part_type):
        self.part_types.remove(part_type)
        Log.info(f"Removed part type: {part_type}")

    def list_part_types(self):
        Log.info("Listing part types")
        return self.part_types
    
    def add_input_type(self, input_type):
        self.input_types.append(input_type)
        Log.info(f"Added input type: {input_type}")

    def remove_input_type(self, input_type):
        self.input_types.remove(input_type)
        Log.info(f"Removed input type {input_type}")

    def add_output_type(self, output_type)
        





