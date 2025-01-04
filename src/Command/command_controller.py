from src.Command.command_item import CommandItem
from src.Utils.message import Log

class CommandController:
    """
    Maintains a list of CommandItem objects and allows execution by name.
    Useful for decoupling entity logic from direct UI calls.
    """

    def __init__(self):
        self.controller_name = None
        self.commands = []

    def set_name(self, name):
        self.controller_name = name

    def add(self, name, function_pointer):
        """
        Register a command by giving it a name and the function to call.
        """
        cmd_item = CommandItem(name, function_pointer)
        self.commands.append(cmd_item)

    def remove(self, name):
        """
        Remove a command from this controller by name.
        """
        for command_item in self.commands:
            if command_item.name == name:
                self.commands.remove(command_item)
                break

    def execute_command_by_name(self, command_name, *args, **kwargs):
        """
        Locate a command by name and call it with supplied args/kwargs
        """
        for cmd_item in self.commands:
            if cmd_item.name == command_name:
                return cmd_item.execute(*args, **kwargs)
        Log.warn(f"Command '{command_name}' not found in this controller.")

    def list_commands(self):
        """
        Log out the commands we have.
        """
        for command_item in self.commands:
            Log.info(f"Command: {command_item.name}")

    def get_commands(self):
        """
        Return the raw list of commands (for introspection).
        """
        return self.commands