from command_item import CommandItem
from message import Log
class CommandModule:
    """
    
    """
    def __init__(self):
        self.name = ""
        self.commands = []
        self.sub_modules = {}
        self.containers = {}

    def add_command(self, name, command):
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        self.commands.append(cmd_item)

    def set_name(self, name):
        self.name = name

    def list_commands(self):
        for command in self.commands:
            Log.info(f"Command: {command.name}")

    def add_container(self, container):
        self.containers.append(container)