from src.Project.Block.block import Block
from src.Utils.message import Log
import json
import os
import time
import uuid
from src.Utils.tools import prompt, prompt_yes_no, prompt_selection, prompt_selection_by_name, prompt_file_path

class Command:
    """
    Represents a single command in the sequence
    """
    def __init__(self, command_type, parent_name, command_name, args=None):
        """
        Initialize a new Command
        
        Args:
            command_type (str): Type of command - "block" or "project"
            parent_name (str): Name of the parent block (if command_type is "block"), otherwise None
            command_name (str): The command to execute
            args (str/list, optional): Arguments for the command
        """
        self.command_type = command_type  # "block" or "project"
        self.parent_name = parent_name    # Block name or None for project commands
        self.command_name = command_name  # The actual command
        self.args = args                  # Command arguments
        
    def to_dict(self):
        """Convert command to dictionary format"""
        result = {
            "type": self.command_type,
            "command": self.command_name,
        }
        
        if self.parent_name:
            result["parent"] = self.parent_name
            
        if self.args:
            # Handle CommandItem objects or other non-serializable types
            if isinstance(self.args, list):
                serialized_args = []
                for arg in self.args:
                    if hasattr(arg, '__class__') and arg.__class__.__name__ == 'CommandItem':
                        # For CommandItem, just store the name
                        serialized_args.append(arg.name if hasattr(arg, 'name') else str(arg))
                    else:
                        serialized_args.append(arg)
                result["args"] = serialized_args
            elif isinstance(self.args, dict):
                # Convert dictionary to a structure that can be serialized
                serialized_dict = {}
                for k, v in self.args.items():
                    if hasattr(v, '__class__') and v.__class__.__name__ == 'CommandItem':
                        serialized_dict[k] = v.name if hasattr(v, 'name') else str(v)
                    else:
                        serialized_dict[k] = v
                result["args"] = serialized_dict
            elif hasattr(self.args, '__class__') and self.args.__class__.__name__ == 'CommandItem':
                # For CommandItem, just store the name
                result["args"] = self.args.name if hasattr(self.args, 'name') else str(self.args)
            else:
                result["args"] = self.args
            
        return result
                
    def to_string(self):
        """Convert command to human-readable string format"""
        if self.command_type == "block":
            prefix = f"Block '{self.parent_name}': "
        else:
            prefix = "Project: "
            
        if self.args:
            if isinstance(self.args, list):
                args_str = ' '.join(str(arg) for arg in self.args)
            elif isinstance(self.args, dict):
                # Format dictionary arguments as "key=value" pairs
                args_str = ' '.join(f"{k}={v}" for k, v in self.args.items())
            else:
                args_str = str(self.args)
            return f"{prefix}{self.command_name} {args_str}"
        else:
            return f"{prefix}{self.command_name}"
            
    def get_executable_command(self):
        """Return the command string that can be executed by send_command"""
        if self.args:
            if isinstance(self.args, list):
                args_str = ' '.join(str(arg) for arg in self.args)
            elif isinstance(self.args, dict):
                # Format dictionary arguments as "key=value" pairs for execution
                args_str = ' '.join(f"{k}={v}" for k, v in self.args.items())
            else:
                args_str = str(self.args)
            return f"{self.command_name} {args_str}"
        else:
            return self.command_name
            
    @classmethod
    def from_dict(cls, data):
        """Create a Command from dictionary data"""
        if isinstance(data, str):
            # Simple string format (legacy)
            return cls("project", None, data)
            
        if not isinstance(data, dict):
            return None
            
        command_id = data.get("id", str(uuid.uuid4())[:8])
        command_type = data.get("type", "project")
        parent_name = data.get("parent")
        
        # Handle legacy format
        if "block" in data:
            command_type = "block"
            parent_name = data["block"]
            
        command_name = data.get("command")
        if not command_name:
            return None
            
        args = data.get("args")
        
        cmd = cls(command_type, parent_name, command_name, args)
        cmd.id = command_id
        return cmd

class CommandSequencer(Block):
    """
    CommandSequencer Block
    
    This block manages sequences of commands and executes them in order.
    It supports project-level commands and commands directed at specific blocks.
    """
    name = "CommandSequencer"
    type = "CommandSequencer"
    
    def __init__(self):
        super().__init__()
        self.name = "CommandSequencer"
        self.type = "CommandSequencer"
        self.commands_file = None
        self.commands = []  # List of Command objects
        self.execution_delay = 0.1  # Default delay between commands in seconds
        self.stop_on_failure = True  # Default to stop execution on failure
        self.current_position = 0  # Position in the command sequence for iteration
        
        # Add block-specific commands
        self.command.add("load_commands", self.load_commands_file)
        self.command.add("reload_commands", self.reload_commands)
        self.command.add("save_commands", self.save_commands)
        self.command.add("add_command", self.add_command)
        self.command.add("edit_command", self.edit_command_args)
        self.command.add("edit_command_args", self.edit_command_args)
        self.command.add("delete_command", self.delete_command)
        self.command.add("execute", self.execute_commands)
        self.command.add("execute_once", self.execute_once)
        self.command.add("show_commands", self.show_commands)
        self.command.add("set_stop_on_failure", self.set_stop_on_failure)
        self.command.add("reset_position", self.reset_position)
        
    def process(self, input_data):
        """
        Process method required by Block base class
        
        Since this block doesn't transform data in the traditional sense,
        we simply pass through any input data.
        """
        return input_data
    
    def load_commands_file(self, file_path=None):
        """
        Load commands from a JSON file
        
        Args:
            file_path (str, optional): Path to JSON file. If not provided, user will be prompted.
        
        Returns:
            bool: True if file loaded successfully, False otherwise
        """
        if not file_path:
            file_path = prompt_file_path("Enter path to commands JSON file: ", file_ext="json")
            if not file_path:
                Log.error("No file path provided")
                return False
        
        try:
            if not os.path.exists(file_path):
                Log.error(f"File not found: {file_path}")
                return False
                
            with open(file_path, 'r') as file:
                commands_data = json.load(file)
                
            self.commands_file = file_path
            
            # Clear existing commands
            self.commands = []
            
            # Parse commands from JSON
            if "sequential_commands" in commands_data:
                for cmd_data in commands_data["sequential_commands"]:
                    command_obj = Command.from_dict(cmd_data)
                    if command_obj:
                        self.commands.append(command_obj)
            elif "project_commands" in commands_data or "block_commands" in commands_data:
                # Legacy format
                if "project_commands" in commands_data:
                    for cmd_str in commands_data["project_commands"]:
                        command_obj = Command("project", None, cmd_str)
                        self.commands.append(command_obj)
                
                if "block_commands" in commands_data:
                    for block_name, cmd_list in commands_data["block_commands"].items():
                        for cmd_str in cmd_list:
                            # Check if the command has arguments
                            parts = cmd_str.split(" ", 1)
                            if len(parts) > 1:
                                command_obj = Command("block", block_name, parts[0], parts[1])
                            else:
                                command_obj = Command("block", block_name, parts[0])
                            self.commands.append(command_obj)
            
            Log.info(f"Loaded {len(self.commands)} commands from {file_path}")
            self.show_commands()
            self.reset_position()
            return True
            
        except json.JSONDecodeError:
            Log.error(f"Invalid JSON format in file: {file_path}")
            return False
        except Exception as e:
            Log.error(f"Error loading commands file: {str(e)}")
            return False
    
    def reload_commands(self):
        """
        Reload commands from the current file
        
        Returns:
            bool: True if file reloaded successfully, False otherwise
        """
        if not self.commands_file:
            Log.error("No commands file has been loaded yet. Use load_commands first.")
            return False
            
        return self.load_commands_file(self.commands_file)
    
    def save_commands(self, file_path=None):
        """
        Save the current commands to a JSON file
        
        Args:
            file_path (str, optional): Path to save the file. If not provided, 
                                      the original file will be used or user will be prompted.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not file_path:
            if self.commands_file:
                file_path = self.commands_file
            else:
                file_path = prompt_file_path("Enter path to save commands file: ", file_ext="json")
                if not file_path:
                    Log.error("No file path provided")
                    return False
        
        try:
            # Ensure file_path has a directory component
            file_path = os.path.abspath(file_path)
            directory = os.path.dirname(file_path)
            
            # Create directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Prepare data for saving
            command_data = {
                "sequential_commands": [cmd.to_dict() for cmd in self.commands]
            }
            
            # Save to file
            with open(file_path, 'w') as file:
                json.dump(command_data, file, indent=4)
            
            self.commands_file = file_path
            Log.info(f"Saved {len(self.commands)} commands to {file_path}")
            return True
            
        except Exception as e:
            Log.error(f"Error saving commands file: {str(e)}")
            return False
    
    def add_command(self, command_type=None, parent_name=None, command_name=None, args=None):
        """
        Add a new command to the sequence
        
        Args:
            command_type (str, optional): Type of command ("block" or "project")
            parent_name (str, optional): Name of the parent block for block commands
            command_name (str, optional): Name of the command to run
            args (str/list, optional): Arguments for the command
        
        Returns:
            bool: True if command added successfully, False otherwise
        """
        if not command_type:
            # Interactive mode
            command_type = prompt_selection("Command type: ", ["block", "project"])
            if command_type == "block":
                block = prompt_selection("Select block: ", self.parent.get_blocks())
                if not block:
                    Log.error("Block is required for block commands")
                    return False
                parent_name = block.get_name()
                command, command_name = prompt_selection_by_name("Select command: ", block.get_commands())
                has_args = prompt_yes_no("Does this command have arguments? (y/n): ")
                if has_args:
                    args = prompt("Enter command arguments: ")
                else:
                    args = None
                    
                # Create and add the command
                command_obj = Command(command_type, parent_name, command_name, args)
                self.commands.append(command_obj)
                Log.info(f"Added command: {command_obj.to_string()}")
                return True
            elif command_type == "project":
                parent_name = None
                command, command_name = prompt_selection_by_name("Select command: ", self.parent.get_commands())
                has_args = prompt_yes_no("Does this command have arguments? (y/n): ")
                if has_args:
                    args = prompt("Enter command arguments: ")
                else:
                    args = None
                    
                # Create and add the command
                command_obj = Command(command_type, None, command_name, args)
                self.commands.append(command_obj)
                Log.info(f"Added command: {command_obj.to_string()}")
                return True
            else:
                Log.error(f"Invalid command type: {command_type}")
                return False
        else:
            # Validate command type
            if command_type not in ["block", "project"]:
                Log.error(f"Invalid command type: {command_type}")
                return False
                
            # Validate parent name for block commands
            if command_type == "block":
                if not parent_name:
                    Log.error("Block name is required for block commands")
                    return False
                    
                # Find the block
                parent_name_lower = parent_name.lower()
                block_found = False
                correct_block_name = parent_name  # Use provided name by default
                
                for block in self.parent.get_blocks():
                    if block.get_name().lower() == parent_name_lower:
                        block_found = True
                        correct_block_name = block.get_name()  # Get correct case of block name
                        break
                
                if not block_found:
                    Log.error(f"Block '{parent_name}' not found")
                    return False
                    
                if not command_name:
                    Log.error("Command name is required")
                    return False
                    
                # Create the command
                command_obj = Command(command_type, correct_block_name, command_name, args)
                self.commands.append(command_obj)
                Log.info(f"Added command: {command_obj.to_string()}")
                return True
            elif command_type == "project":
                if not command_name:
                    Log.error("Command name is required")
                    return False
                    
                command_obj = Command(command_type, None, command_name, args)
                self.commands.append(command_obj)
                Log.info(f"Added command: {command_obj.to_string()}")
                return True
                
        # We should never reach here, but just in case
        return False
    
    def edit_command_args(self, command_type=None, parent_name=None, command_name=None, **kwargs):
        """
        Edit an existing command in the sequence
        
        Args:
            command_type (str, optional): Type of command ("block" or "project")
            parent_name (str, optional): Name of the parent block to identify command
            command_name (str, optional): Name of the command to edit
            **kwargs: Additional keyword arguments to be used as command arguments
        
        Returns:
            bool: True if command edited successfully, False otherwise
        """
        Log.info(f"kwargs: {kwargs}")

        # Convert kwargs to args string in format "kwarg1=value, kwarg2=value"
        args = ", ".join([f"{key}={value}" for key, value in kwargs.items()])
        
        if command_type:
            Log.info(f"Editing command: {command_type}")
            if parent_name:
                Log.info(f"parent_name: {parent_name}")
            if command_name:
                Log.info(f"command_name: {command_name}")
            if args:
                Log.info(f"args: {args}")
            for cmd in self.commands:
                Log.info(f"checking command: {cmd.command_type.lower()} to {command_type.lower()}")
                if cmd.command_type.lower() == command_type.lower():
                    Log.info(f"matched command type {cmd.command_type}")
                    if parent_name: 
                        if cmd.parent_name and cmd.parent_name.lower() == parent_name.lower():
                            Log.info(f"matched parent_name {cmd.parent_name}")
                            if cmd.command_name.lower() == command_name.lower():
                                Log.info(f"matched command_name {cmd.command_name}")
                                cmd.args = args
                                Log.info(f"Updated command: {cmd.to_string()}")
                                return True
                    else: 
                        if cmd.command_name.lower() == command_name.lower():
                            cmd.args = args
                            Log.info(f"Updated command: {cmd.to_string()}")
                            return True
        else:
            selected_command = self.prompt_command_selection("Select command: ")
            if selected_command:
                new_args = prompt("Enter new arguments: ")
                selected_command.args = new_args
                Log.info(f"Updated command: {selected_command.to_string()}")
                return True
        return False
    
    def execute_once(self):
        """
        Execute the next command in the sequence
        
        Returns:
            bool: True if command executed successfully, False otherwise
        """
        if not self.commands:
            Log.error("No commands to execute. Please add commands first.")
            return False
        
        if self.current_position >= len(self.commands):
            Log.info("End of command sequence reached. Resetting position.")
            self.reset_position()
            return True
        
        cmd = self.commands[self.current_position]
        result = self._execute_single_command(cmd, self.current_position + 1, len(self.commands))
        
        self.current_position += 1
        return result
    
    def reset_position(self):
        """
        Reset the current position in the command sequence
        
        Returns:
            bool: Always returns True
        """
        self.current_position = 0
        Log.info("Command position reset to beginning")
        return True
    
    def prompt_command_selection(self, prompt_text):
        Log.info(prompt_text)
        for i, obj in enumerate(self.commands):
            Log.info(f"{i}: {obj.command_type}, {obj.parent_name}, {obj.command_name}")
        while True:
            selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
            if not selection: 
                Log.info("Selection exited by user.")
                return None, None
            if selection.isdigit():
                index = int(selection)
                if 0 <= index < len(self.commands):
                    return self.commands[index]
            elif selection in self.commands:
                return self.commands[selection]
            Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")
    
    def show_commands(self):
        """
        Display all commands in the sequence
        
        Returns:
            bool: True if commands are displayed
        """
        if not self.commands:
            Log.info("No commands in sequence")
            return True
        
        Log.info(f"Command sequence ({len(self.commands)} commands):")
        
        for i, cmd in enumerate(self.commands):
            marker = "→ " if i == self.current_position else "  "
            Log.info(f"cmd type: {cmd.command_type}")
            if cmd.parent_name:
                Log.info(f"cmd parent_name: {cmd.parent_name}")
            Log.info(f"cmd command_name: {cmd.command_name}")
            if cmd.args:
                Log.info(f"cmd args: {cmd.args}")
        return True
    
    def get_commands(self):
        return self.commands
    
    def set_stop_on_failure(self, should_stop=None):
        """
        Set whether command execution should stop on failure
        
        Args:
            should_stop (bool, optional): Whether to stop on failure. If not provided, user will be prompted.
            
        Returns:
            bool: True if setting was updated successfully
        """
        if should_stop is None:
            response = prompt_yes_no("Stop execution when a command fails? (y/n): ")
            should_stop = response
        
        self.stop_on_failure = bool(should_stop)
        Log.info(f"Stop on failure set to: {self.stop_on_failure}")
        return True
    
    def get_metadata(self):
        """
        Get metadata for saving the block
        
        Returns:
            dict: Block metadata
        """
        # Convert commands to serializable format
        commands_list = [cmd.to_dict() for cmd in self.commands]
        
        metadata = {
            "name": self.name,
            "type": self.type,
            "commands_file": self.commands_file,
            "commands": commands_list,
            "execution_delay": self.execution_delay,
            "stop_on_failure": self.stop_on_failure,
            "current_position": self.current_position,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
        return metadata
    
    def save(self, save_dir):
        """
        Save the block data
        
        Args:
            save_dir (str): Directory to save data to
        """
        self.data.save(save_dir)
    
    def load(self, block_dir):
        """
        Load the block from saved data
        
        Args:
            block_dir (str): Directory to load data from
        """
        block_metadata = self.get_metadata_from_dir(block_dir)
        
        # Load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        
        # Load command sequencer specific attributes
        self.commands_file = block_metadata.get("commands_file")
        
        commands_data = block_metadata.get("commands", [])
        self.commands = []
        for cmd_data in commands_data:
            command_obj = Command.from_dict(cmd_data)
            if command_obj:
                self.commands.append(command_obj)
        
        delay = block_metadata.get("execution_delay")
        if delay is not None:
            self.execution_delay = float(delay)
            
        stop_on_failure = block_metadata.get("stop_on_failure")
        if stop_on_failure is not None:
            self.stop_on_failure = bool(stop_on_failure)
            
        current_position = block_metadata.get("current_position")
        if current_position is not None:
            self.current_position = int(current_position)
        
        # Load sub-components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))
        
        # Push loaded data to output
        self.output.push_all(self.data.get_all())

    def delete_command(self, command_index=None):
        """
        Delete a command from the sequence
        
        Args:
            command_index (int, optional): Index of the command to delete. If not provided, 
                                           user will be prompted to select a command.
        
        Returns:
            bool: True if command deleted successfully, False otherwise
        """
        if not self.commands:
            Log.error("No commands to delete. Please add commands first.")
            return False
            
        if command_index is None:
            # Show available commands and prompt for selection
            Log.info("Select command to delete:")
            for i, cmd in enumerate(self.commands):
                Log.info(f"{i}: {cmd.to_string()}")
                
            while True:
                selection = prompt("Enter command index to delete (or 'c' to cancel): ")
                if selection.lower() == 'c':
                    Log.info("Command deletion cancelled")
                    return False
                    
                if selection.isdigit():
                    command_index = int(selection)
                    if 0 <= command_index < len(self.commands):
                        break
                        
                Log.error("Invalid selection. Please enter a valid index or 'c' to cancel.")
        elif not isinstance(command_index, int) or command_index < 0 or command_index >= len(self.commands):
            Log.error(f"Invalid command index: {command_index}. Valid range: 0-{len(self.commands)-1}")
            return False
            
        # Delete the command
        deleted_command = self.commands.pop(command_index)
        Log.info(f"Deleted command: {deleted_command.to_string()}")
        
        # Adjust current position if needed
        if self.current_position >= len(self.commands):
            self.current_position = max(0, len(self.commands) - 1)
        elif self.current_position > command_index:
            self.current_position -= 1
            
        return True
    
    def execute_commands(self):
        """
        Execute all commands in the sequence
        
        Returns:
            bool: True if all commands executed successfully, False otherwise
        """
        if not self.commands:
            Log.error("No commands to execute. Please add commands first.")
            return False
        
        self.reset_position()
        Log.info(f"Beginning command sequence execution with {self.execution_delay}s delay between commands")
        Log.info(f"Commands will execute sequentially and will stop if any command fails")
        
        results = []
        command_count = 0
        total_commands = len(self.commands)
        
        for cmd in self.commands:
            command_count += 1
            result = self._execute_single_command(cmd, command_count, total_commands)
            results.append(result)
            
            if not result and self.stop_on_failure:
                Log.error(f"Command execution halted due to failure. {command_count} of {total_commands} commands executed.")
                break
        
        success = all(results)
        Log.info(f"**************************************************")
        Log.info(f"Command execution {'completed successfully' if success else 'completed with errors'}")
        Log.info(f"{sum(1 for r in results if r)} of {len(results)} commands succeeded")
        Log.info(f"**************************************************")
        
        return success
        
    def _execute_single_command(self, cmd, command_number, total_commands):
        """
        Execute a single command in the sequence
        
        Args:
            cmd (Command): Command object to execute
            command_number (int): Current command number (for logging)
            total_commands (int): Total number of commands (for logging)
            
        Returns:
            bool: True if command executed successfully, False otherwise
        """
        try:
            Log.info(f"Executing command {command_number}/{total_commands}: {cmd.to_string()}")
            
            if cmd.command_type == "block":
                # Find the block
                target_block = None
                if self.parent.get_block(cmd.parent_name):
                    target_block_name = self.parent.get_block(cmd.parent_name).get_name()
                else:
                    Log.error(f"Block '{cmd.parent_name}' not found")
                    return False
                if cmd.args is None:
                    command_string = f"{target_block_name} {cmd.command_name}"
                else:
                    command_string = f"{target_block_name} {cmd.command_name} {cmd.args}"
                Log.info(f"Sending command: {command_string}")
                result = self.send_command(command_string)
                
                if result is None:
                    Log.warning(f"Command returned None: {cmd.to_string()}")
                    # Treat None as success
                    return True
                
                return bool(result)
            
            elif cmd.command_type == "project":
                # Execute the command on the project
                command_string = f"{cmd.command_name} {cmd.args}"
                Log.info(f"Sending command: {command_string}")
                result = self.send_command(command_string)
                
                if result is None:
                    Log.warning(f"Command returned None: {cmd.to_string()}")
                    # Treat None as success
                    return True
                
                return bool(result)
            
            else:
                Log.error(f"Unknown command type: {cmd.command_type}")
                return False
                
        except Exception as e:
            Log.error(f"Error executing command: {cmd.to_string()}")
            Log.error(f"Exception: {str(e)}")
            return False 
            Log.error(f"Exception: {str(e)}")
            return False 