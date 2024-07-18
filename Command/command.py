from .Audio.audio import Audio
from .Project.project import Project
from .Digest.digest import Digest
from message import Log
from .command_module import CommandModule
from .command_item import CommandItem


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
        self.commands = []
        self.modules = []

        self.register_command("list_commands", self.list_commands)

        self.add_module(Audio(self.model, self.settings, self.project_dir))
        self.add_module(Project(self.project))
        self.add_module(Digest(self.model))
        

    def add_module(self, module_item):
        module_name = module_item.name
        cmd_module_item = CommandModule()
        cmd_module_item.set_name(module_name)
        Log.info(f"Registering module: {module_name}")
        for cmd_item in module_item.commands:
            cmd_module_item.add_command(cmd_item)
            Log.info(f"     Regisering Command: '{cmd_item.name}'")
            if module_item.sub_modules:
                for sub_module_item in module_item.sub_modules:
                    sub_module_name = sub_module_item.name
                    sub_cmd_module_item = CommandModule()
                    sub_cmd_module_item.set_name(sub_module_name)
                    Log.info(f"     Registering sub module: {sub_module_name}")
                    for sub_cmd_item in sub_module_item.commands:
                        sub_cmd_module_item.add_command(sub_cmd_item)
                        Log.info(f"          Regisering Command: '{sub_cmd_item.name}'")
                    cmd_module_item.add_sub_module(sub_cmd_module_item)
                    
            self.modules.append(cmd_module_item)
                    
        
            
            

    def list_commands(self):
        self.module_name = "list_commands"
        Log.info(f"Listing Commands")
        for key, func in self.commands.items():
            Log.info(f"Command {key}")

    def register_command(self, name, command):
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        self.commands.append(cmd_item)

    def execute_command(self, name, *args, **kwargs):
        if name in self.commands:
            return self.commands[name](*args, **kwargs)
        else:
            Log.error(f"Command '{name}' not recognized.")

