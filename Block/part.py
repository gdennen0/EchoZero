from command_module import CommandModule

class Part(CommandModule):
    def __init__(self):
        self.name = None
        self.type = None
        self.block_type = None