class CommandItem:
    """
    A simple wrapper for a command action:
      - name: The name of the command
      - command_function: The callable function/method
      - description: A description of what the command does
    """

    def __init__(self, name=None, command_function=None, description=None):
        self.name = name
        self.command_function = command_function
        self.description = description or ""

    def set_name(self, name):
        self.name = name

    def set_command(self, command_function):
        self.command_function = command_function

    def set_description(self, description):
        self.description = description

    def execute(self, args=None):
        """
        Execute the underlying function pointer with provided arguments.
        
        Args:
            args (list): Arguments to pass to the command function
            
        Returns:
            The result of the command function execution
        """
        if not self.command_function:
            raise ValueError(f"Command '{self.name}' has no function attached.")
            
        if args is None:
            args = []
            
        return self.command_function(args)