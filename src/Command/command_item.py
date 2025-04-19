from src.Utils.message import Log
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
        
        # Log the command name
        Log.info(f"Executing command: {self.name}")
        
        # Log all positional arguments
        if args:
            for i, arg in enumerate(args):
                Log.info(f"Arg {i}: {arg}")
        else:
            Log.info("No positional arguments")
            
        # Log all keyword arguments
        if kwargs:
            for key, value in kwargs.items():
                Log.info(f"Kwarg: {key}={value}")
        else:
            Log.info("No keyword arguments")

        if not self.command_function:
            raise ValueError(f"Command '{self.name}' has no function attached.")
        try:
            self.command_function(*args, **kwargs)
            return True
        except Exception as e:
            Log.error(f"Error executing command '{self.name}': {e}")
            return False