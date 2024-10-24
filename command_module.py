from command_item import CommandItem
from message import Log
class CommandModule:
    """
        Parent class for command module Command modules are pretty generic 
        but outline general program functionality for example Audio and Digest classes

        name = the name of the module (ex. "Audio")
        commands = a list of command items
        sub_modules = a list of sub modules 
    
    """
    def __init__(self):
        self.commands = []
        self.sub_modules = []
        self.name = None


    def add_command(self, name, command):
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        self.commands.append(cmd_item)

    def add_sub_module(self, sub_module):
        self.sub_modules.append(sub_module)
        Log.info(f"Adding sub module {sub_module.name}")

    def set_name(self, name):
        self.name = name

    def list_commands(self):
        for command in self.commands:
            Log.info(f"Command: {command.name}")
