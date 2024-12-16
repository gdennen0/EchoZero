class CommandItem:
    """
        Generic class structure for a command action
        name = command name (whatever you want to input into cli)
        command = pointer to the command 

    """

    def __init__(self, name=None, command=None, controller=None):
        self.name = name
        self.command = command
        self.controller = controller

    def set_name(self, name):
        self.name = name

    def set_command(self, command):
        self.command = command

    def set_controller(self, controller):
        self.controller = controller

    def add_to_queue(self, *args, **kwargs):
        self.controller.enqueue_command(self, *args, **kwargs)

    def execute(self, *args, **kwargs):
        self.command(*args, **kwargs)

