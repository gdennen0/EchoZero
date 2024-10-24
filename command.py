from message import Log
from command_module import CommandModule
from command_item import CommandItem
from project import Project

"""
Each module is responsible for its own commands, 
and the Command class aggregates these commands, 

"""
class Command:
    def __init__(self, project):
        self.project_dir = project.dir
        self.project = project
        self.command_modules = []

        self.add_command_module(project)
        # self.add_command_module(Audio(self.model, self.settings))
        # self.add_command_module(Digest(self.model, self.settings))
        # self.add_command_module(Ingest(self.model, self.project, self.settings))

    def add_command_module(self, module_item):
        module_name = module_item.name
        cmd_module_item = CommandModule()
        cmd_module_item.set_name(module_name)
        Log.info(f"Registering module: {module_name}")
        for cmd_item in module_item.commands:
            cmd_module_item.add_command(cmd_item.name, cmd_item.command)
            Log.info(f"     Regisering Command: '{cmd_item.name}'")
        if module_item.sub_modules:
            for sub_module_item in module_item.sub_modules:
                sub_module_name = sub_module_item.name
                sub_cmd_module_item = CommandModule()
                sub_cmd_module_item.set_name(sub_module_name)
                Log.info(f"     Registering sub module: {sub_module_name}")
                for sub_cmd_item in sub_module_item.commands:
                    sub_cmd_module_item.add_command(sub_cmd_item.name, sub_cmd_item.command)
                    Log.info(f"          Regisering Command: '{sub_cmd_item.name}'")

                cmd_module_item.add_sub_module(sub_cmd_module_item)
                    
        self.command_modules.append(cmd_module_item)
