# Base Block class
from command_module import CommandModule

class Block(CommandModule):
    def __init__(self):
        super().__init__()
        self.name = None
        self.type = None
        self.parts = []
        self.part_types = []
        self.add_command("add_part", self.AddPart)
        self.add_command("remove_part", self.RemovePart)
        self.add_command("list_parts", self.ListParts)

    def AddPart(self, part):
        self.parts.append(part)

    def RemovePart(self, part):
        self.parts.remove(part)

    def ListParts(self):
        return self.parts

    def ClearParts(self):
        self.parts = []        

    def add_part_type(self, part_type):
        self.part_types.append(part_type)

    def remove_part_type(self, part_type):
        self.part_types.remove(part_type)

    def list_part_types(self):
        return self.part_types





