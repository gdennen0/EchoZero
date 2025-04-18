from src.Utils.message import Log

class CommandParser:
    """
    Command parser that works with the ApplicationController.
    """

    def __init__(self, app_controller):
        """
        Initialize command parser.
        
        Args:
            app_controller: ApplicationController instance
        """
        self.app = app_controller
    
    def parse_and_execute(self, command_string):
        """
        Parse and execute a command string.
        
        Args:
            command_string (str): The command to parse and execute
            
        Returns:
            bool: True if command executed successfully
        """
        # Split the command string by spaces
        parts = command_string.lower().split()
        if not parts:
            return False
        
        Log.info(f"Parsing command: {command_string}")
        
        # Check for application-level commands
        if parts[0] in ["app", "application"]:
            # Remove "app"/"application" prefix and execute on app controller
            remaining_parts = parts[1:]
            if not remaining_parts:
                Log.error("No application command specified")
                return False
                
            command_name = remaining_parts[0]
            args = remaining_parts[1:]
            
            Log.info(f"Executing application command: {command_name}")
            return self.app.command.execute_command_by_name(command_name, args)
        
        # Check for project-level commands
        if parts[0] in ["project", "p"]:
            # Get active project
            active_project = self.app.get_active_project()
            if not active_project:
                Log.error("No active project")
                return False
                
            # Remove "project" prefix and execute on active project
            remaining_parts = parts[1:]
            if not remaining_parts:
                Log.error("No project command specified")
                return False
                
            command_name = remaining_parts[0]
            args = remaining_parts[1:]
            
            Log.info(f"Executing project command: {command_name}")
            return active_project.command.execute_command_by_name(command_name, args)
        
        # Get active project
        active_project = self.app.get_active_project()
        if not active_project:
            # If no active project, try application commands
            command_name = parts[0]
            args = parts[1:]
            
            Log.info(f"No active project, trying application command: {command_name}")
            return self.app.command.execute_command_by_name(command_name, args)
        
        # Try to get matching block
        block = self._get_matching_block(active_project, parts)
        
        if not block:
            # If no block matched, try as a project command
            command_name = parts[0]
            args = parts[1:]
            
            Log.info(f"No block matched, trying project command: {command_name}")
            return active_project.command.execute_command_by_name(command_name, args)
        
        # Block found, remove its name from parts
        remaining_parts = parts[1:]
        
        if not remaining_parts:
            # Just the block name was provided, show info
            Log.info(f"No command specified for block: {block.name}")
            return block.command.execute_command_by_name("info", [])
        
        # Try to get input/output controllers
        input_controller = self._get_matching_input(block, remaining_parts)
        output_controller = self._get_matching_output(block, remaining_parts)
        
        if input_controller:
            # Input controller matched, adjust remaining parts
            remaining_parts = remaining_parts[1:]  # Remove "input"
            
            # If we have a specific port, remove that too
            if remaining_parts and hasattr(input_controller, 'items'):
                port_names = [port.name.lower() for port in input_controller.items()]
                if remaining_parts[0] in port_names:
                    port_name = remaining_parts[0]
                    remaining_parts = remaining_parts[1:]
                    
                    # Find the port object
                    port = next((p for p in input_controller.items() if p.name.lower() == port_name), None)
                    
                    if port and hasattr(port, 'command'):
                        # Execute command on port
                        if not remaining_parts:
                            Log.error(f"No command specified for port: {port.name}")
                            return False
                            
                        command_name = remaining_parts[0]
                        args = remaining_parts[1:]
                        
                        Log.info(f"Executing port command: {command_name}")
                        return port.command.execute_command_by_name(command_name, args)
            
            # Execute command on input controller
            if not remaining_parts:
                Log.error(f"No command specified for input controller")
                return False
                
            command_name = remaining_parts[0]
            args = remaining_parts[1:]
            
            Log.info(f"Executing input controller command: {command_name}")
            return input_controller.command.execute_command_by_name(command_name, args)
            
        elif output_controller:
            # Output controller matched, adjust remaining parts
            remaining_parts = remaining_parts[1:]  # Remove "output"
            
            # Similar logic as input controller
            # ...
            
        # No input/output matched, execute command on block
        command_name = remaining_parts[0]
        args = remaining_parts[1:]
        
        Log.info(f"Executing block command: {command_name}")
        return block.command.execute_command_by_name(command_name, args)
    
    def _get_matching_block(self, project, parts):
        """
        Find a block that matches the first part of the command.
        
        Args:
            project: Project instance
            parts (list): Command parts
            
        Returns:
            Block: Matching block or None
        """
        if not parts:
            return None
            
        block_name = parts[0]
        
        for block in project.get_blocks():
            if block is not None and block.name.lower() == block_name:
                return block
                
        return None
            
    def _get_matching_input(self, block, parts):
        """
        Check if the next part refers to the input controller.
        
        Args:
            block: Block object
            parts (list): Remaining command parts
            
        Returns:
            InputController: Input controller or None
        """
        if not parts or parts[0] != "input":
            return None
            
        return block.input if hasattr(block, 'input') else None

    def _get_matching_output(self, block, parts):
        """
        Check if the next part refers to the output controller.
        
        Args:
            block: Block object
            parts (list): Remaining command parts
            
        Returns:
            OutputController: Output controller or None
        """
        if not parts or parts[0] != "output":
            return None
            
        return block.output if hasattr(block, 'output') else None
