class Request:
    def __init__(self):
        self.block = None
        self.command = None
        self.args = []

    def add_block(self, block):
        self.block = block

    def add_command(self, command):
        self.command = command

    def add_args(self, args):
        self.args = args
