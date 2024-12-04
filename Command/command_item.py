class CommandItem:
    """
        Generic class structure for a command action
        name = command name (whatever you want to input into cli)
        command = pointer to the command 

    """

    def __init__(self, name=None, command=None):
        self.name = name
        self.command = command

    def set_name(self, name):
        self.name = name

    def set_command(self, command):
        self.command = command
