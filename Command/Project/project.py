class Project:
    def __init__(self, project):
        self.project = project

    def get_commands(self):
        return {
            "save": self.save,
            "save_as": self.save_as,
        }

    def save(self):
        self.project.save()

    def save_as(self):
        self.project.save_as()