from .Audio.audio import Audio
from .Project.project import Project
from .Digest.digest import Digest
from message import Log


"""
Each module is responsible for its own commands, 
and the Command class aggregates these commands, 
reducing the need for separate command modules.


"""
class Command:
    def __init__(self, project, settings):
        self.model = project.model
        self.settings = settings
        self.project_dir = project.dir
        self.project = project
        self.digest = Digest(self.model)
        self.command_registry = CommandRegistry()

        self.audio = Audio(self.model, self.settings, self.project_dir)
        self.project = Project(self.project)

        self.register_commands()

    def register_commands(self):
        # Register commands from AudioCommands
        for cmd_name, cmd_func in self.audio.get_commands().items():
            self.command_registry.register(cmd_name, cmd_func)
            Log.info(f"Registered Audio command: {cmd_name}")
        
        # Register commands from ProjectCommands
        for cmd_name, cmd_func in self.project.get_commands().items():
            self.command_registry.register(cmd_name, cmd_func)
            Log.info(f"Registered Project command: {cmd_name}")
        
        # Register commands from Digest
        for cmd_name, cmd_func in self.digest.get_commands().items():
            self.command_registry.register(cmd_name, cmd_func)
            Log.info(f"Registered Digest command: {cmd_name}")

    def execute_command(self, command_name, *args, **kwargs):
        self.command_registry.execute(command_name, *args, **kwargs)


class CommandRegistry:
    def __init__(self):
        self.commands = {}

    def register(self, name, func):
        self.commands[name] = func

    def execute(self, name, *args, **kwargs):
        if name in self.commands:
            return self.commands[name](*args, **kwargs)
        else:
            Log.error(f"Command '{name}' not recognized.")