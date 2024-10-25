from command_module import CommandModule

class Part(CommandModule):
    def __init__(self):
        super().__init__()
        self.name = None
        self.type = None
        self.block_type = None