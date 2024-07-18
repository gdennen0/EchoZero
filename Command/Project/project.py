from Command.command_item import CommandItem

class Project:
    def __init__(self, project):
        self.project = project
        self.commands = []
        self.name = "Project"
        self.sub_modules = []

        self.add_command("save", self.save)
        self.add_command("save_as", self.save_as)

    def add_command(self, name, command):
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        self.commands.append(cmd_item)

    def add_sub_module(self, sub_module):
        self.sub_modules.append(sub_module)

    def save(self):
        self.project.save()

    def save_as(self):
        self.project.save_as()