from src.Command.command_item import CommandItem
from src.Utils.message import Log

class CommandController:
    """
    Maintains a list of CommandItem objects and allows execution by name.
    Now with enhanced command registration.
    """

    def __init__(self, db_controller=None, parent=None):
        """
        Initialize a command controller.
        
        Args:
            db_controller: Optional DatabaseController instance
            parent: Parent object that owns this controller
        """
        self.db = db_controller
        self.parent = parent
        self.controller_name = None
        self.commands = []
        
        # Register commands after initialization
        if self.parent:
            self._register_command_methods()
    
    def _register_command_methods(self):
        """
        Automatically register methods that start with cmd_ as commands.
        """
        if not self.parent:
            return
            
        Log.info(f"Registering commands for {self.parent.__class__.__name__}")
        
        # Find all methods that start with cmd_
        for attr_name in dir(self.parent):
            if attr_name.startswith('cmd_'):
                method = getattr(self.parent, attr_name)
                if callable(method):
                    # Convert cmd_some_name to some-name
                    command_name = attr_name[4:].replace('_', '-')
                    self.add(command_name, method)
                    # Log.info(f"Auto-registered command: {command_name} from {attr_name}")

    def set_name(self, name):
        self.controller_name = name

    def add(self, name, function_pointer, description=None):
        """
        Register a command by giving it a name and the function to call.
        
        Args:
            name (str): Name of the command
            function_pointer: Function to execute when command is called
            description (str, optional): Description of what the command does
        """
        cmd_item = CommandItem(name, function_pointer, description)
        self.commands.append(cmd_item)
        Log.info(f"Registered command: {name}")
        
        # Ensure cmd_item has proper references
        if not cmd_item.command_function:
            Log.error(f"Command '{name}' has no function attached")

    def remove(self, name):
        """
        Remove a command from this controller by name.
        """
        for command_item in self.commands:
            if command_item.name == name:
                self.commands.remove(command_item)
                Log.info(f"Removed command: {name}")
                break

    def execute_command_by_name(self, command_name, args=None, **kwargs):
        """
        Locate a command by name and call it with supplied args/kwargs
        
        Args:
            command_name (str): Name of the command to execute
            args (list): List of arguments to pass to the command
            **kwargs: Additional keyword arguments
            
        Returns:
            bool: True if command executed successfully
        """
        if args is None:
            args = []
            
        Log.info(f"Attempting to execute command: {command_name}")
        for cmd_item in self.commands:
            if cmd_item.name.lower() == command_name.lower():
                Log.info(f"Executing command: {cmd_item.name}")
                try:
                    result = cmd_item.execute(args)
                    
                    # If this controller belongs to a database model, persist changes
                    if self.parent and hasattr(self.parent, 'persist') and callable(self.parent.persist):
                        self.parent.persist()
                        
                    return result if result is not None else True
                except Exception as e:
                    Log.error(f"Error executing command '{command_name}'", exception=e)
                    return False
        
        Log.warning(f"Command '{command_name}' not found in this controller with {len(self.commands)} commands")
        if self.commands:
            Log.info(f"Available commands: {', '.join([cmd.name for cmd in self.commands])}")
        return False

    def list_commands(self):
        """
        List all commands with their descriptions.
        """
        Log.info(f"{len(self.commands)} commands in {self.controller_name or 'unnamed controller'}:")
        for command_item in self.commands:
            desc = f" - {command_item.description}" if command_item.description else ""
            Log.info(f"  {command_item.name}{desc}")

    def get_commands(self):
        """
        Return the raw list of commands (for introspection).
        """
        return self.commands