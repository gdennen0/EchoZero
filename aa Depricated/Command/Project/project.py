from Command.command_module import CommandModule

class Project(CommandModule):
    def __init__(self, project):
        super().__init__(project=project) 
        self.project = project
        self.set_name("Project")
        self.add_command("save", self.save)
        self.add_command("save_as", self.save_as)

    def save(self):
        self.project.save()

    def save_as(self):
        self.project.save_as()

    def add_container(self, container):
        self.project.add_container(container)
    
    def remove_container(self, container_name):
        self.project.remove_container(container_name)
