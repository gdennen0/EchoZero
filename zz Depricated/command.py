"""
Each module is responsible for its own commands, 
and the Command class aggregates these commands, 

"""
class Command:
    def __init__(self, project):
        self.project = project
        self.command_controllers = []

        self.add_command_controller(project)

    def add_command_controller(self, module_item):
        self.command_controllers.append(module_item)
        module_item.list_commands()

    

