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

    def add_command_module(self, module_item):
        self.command_modules.append(module_item)
        module_item.list_commands()

    

