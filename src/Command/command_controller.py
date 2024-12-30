from src.Command.command_item import CommandItem
from src.Utils.message import Log

class CommandController:
    """
    
    """
    def __init__(self):
        self.name = None
        self.commands = []
        self.sub_modules = {}
        self.containers = {}

    def add(self, name, command):
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        self.commands.append(cmd_item)

    def remove(self, name):
        for command in self.commands:
            if command.name == name:
                self.commands.remove(command)
                break

    def set_name(self, name):
        self.name = name

    def list_commands(self):
        for command in self.get_commands():
            Log.info(f"Command: {command.name}")

    def add_container(self, container):
        self.containers.append(container)

    def get_commands(self):
        "method to shield the interaction to commands list for modularity"
        return self.commands
