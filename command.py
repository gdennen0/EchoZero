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
        Log.info(f"Registering module: {module_name}")
        
        # Register top-level commands
        for cmd in module_item.commands:
            Log.info(f"     Registering Command: '{cmd.name}'")
        
        # Register containers and their blocks
        for container_name, container in module_item.containers.items():
            Log.info(f"     Registering container: {container_name}")
            
            # Register container-level commands
            for cmd in container.commands:
                Log.info(f"          Registering Container Command: '{cmd.name}'")
            
            # Register blocks within the container
            if hasattr(container, 'blocks'):
                for block_name, block in container.blocks.items():
                    Log.info(f"          Registering block: {block_name}")
                    
                    # Register block-level commands
                    for cmd in block.commands:
                        Log.info(f"               Registering Block Command: '{cmd.name}'")
        
        self.command_modules.append(module_item)
