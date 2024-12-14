from Command.command_controller import CommandController
from Utils.message import Log

class Part():
    def __init__(self):
        Log.info(f"Creating Instance of the Part Object")
        self.name = None
        self.type = None
        self.command = CommandController()

    def set_name(self, name):
        self.name = name
        Log.info(f"Set name: {name}")

    def set_type(self, type):
        self.type = type
        Log.info(f"Set type: {type}")
