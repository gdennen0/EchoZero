from .parser import Parser
class ProjectInterface:
    def __init__(self, project):
        self.project = project
        self.name = "ProjectInterface"
        self.type = "Project"
        self.parser = Parser(self.project)

    def enque_command(self, command, args):
        self.command_queue.append((command, args))

    def parse(self, input_string):
        block, input, output, command, args = self.parser.parse(input_string)
        if block:


    def prompt_user(self):
        for interface in self.project.interfaces:   
            interface.prompt_user()

