class CommandItem:
    """
    A simple wrapper for a command action:
      - name: The name of the command
      - command_function: The callable function/method
    """

    def __init__(self, name=None, command_function=None):
        self.name = name
        self.command_function = command_function

    def set_name(self, name):
        self.name = name

    def set_command(self, command_function):
        self.command_function = command_function

    def execute(self, *args, **kwargs):
        """
        Execute the underlying function pointer with any arguments.
        """
        if not self.command_function:
            raise ValueError(f"Command '{self.name}' has no function attached.")
        return self.command_function(*args, **kwargs)