class CommandModule:
    def __init__(self, name=None, command=None):
        self.name = None
        self.commands = []
        self.sub_modules = []

    def set_name(self, name):
        self.name = name

    def add_command(self, command):
        self.commands.append(command)

    def add_sub_module(self, sub_module):
        self.sub_modules.append(sub_module)
