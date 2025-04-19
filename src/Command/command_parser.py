from src.Utils.message import Log

class CommandParser:
    """
    Interprets an input string (like from CLI or another UI) and tries to
    match it to blocks, ports, and finally commands to execute.
    
    Format: [block_name] [command_name] arg1,arg2,kwarg1=value1,...
    Arguments are comma-separated to allow spaces within individual arguments.
    """

    def __init__(self, project):
        self.project = project

    def parse_and_execute(self, input_string):
        Log.info(f"Parsing and executing command: {input_string}")
        # Initialize variables
        block = None
        port = None
        command_item = None
        args = []
        kwargs = {}

        # We need to identify the structure part (block/command) and the argument part
        # The process is:
        # 1. Try to match a block from the first word
        # 2. If there's a block, try to match input/output ports from the second word
        # 3. Try to match a command from the next word
        # 4. Everything after that, separated by commas, are arguments

        parts = input_string.split()
        if not parts:
            return False
            
        # Try to match a block from the first word
        block = self._get_matching_block([parts[0].lower()])
        selected_module = self.project
        next_word_index = 0  # Where to look for the command
        
        if block:
            selected_module = block
            next_word_index = 1
            
            # If there are more parts, try to match input/output ports
            if len(parts) > 1:
                input_match = self._get_matching_input(block, [parts[1].lower()])
                output_match = self._get_matching_output(block, [parts[1].lower()])
                if input_match or output_match:
                    selected_module = input_match if input_match else output_match
                    next_word_index = 2
        
        # Try to match a command
        if len(parts) > next_word_index:
            command_item, _ = self._get_command(selected_module, [parts[next_word_index].lower()])
            next_word_index += 1 if command_item else 0
        
        # Now we know what the command structure is
        # Everything after next_word_index in the original string should be arguments
        arg_string = ""
        if command_item and len(parts) > next_word_index:
            # Get original string to maintain original case and spacing
            command_prefix = " ".join(parts[:next_word_index])
            # Find where this prefix ends in the original string
            prefix_end = input_string.find(command_prefix) + len(command_prefix)
            
            # The rest of the string contains the arguments
            arg_string = input_string[prefix_end:].strip()
        
        # Process arguments - they are comma-separated
        if arg_string:
            # Split by commas, but preserve them if they're inside quotes
            # This is a simple approach - for complex parsing with nested quotes,
            # a more sophisticated parser would be needed
            arg_parts = []
            current_arg = ""
            i = 0
            while i < len(arg_string):
                if arg_string[i] == ',':
                    arg_parts.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += arg_string[i]
                i += 1
            
            if current_arg:  # Add the last argument if it exists
                arg_parts.append(current_arg.strip())
            
            # Process each argument
            for arg in arg_parts:
                if not arg:  # Skip empty args
                    continue
                    
                if '=' in arg:
                    # This is a kwarg in the format key=value
                    key, value = arg.split('=', 1)
                    kwargs[key.strip()] = value.strip()
                else:
                    args.append(arg.strip())

        Log.parser(', '.join(filter(None, [
            f"Matched block: {block.name}" if block else None,
            f"Command item: {command_item.name}" if command_item else None,
        ])))
        for arg in args:
            Log.parser(f"Arg: {arg}")
        for key, value in kwargs.items():
            Log.parser(f"Kwarg: {key}={value}")
        
        if command_item:
            result = command_item.execute(*args, **kwargs) # Execute Command with args and kwargs
            return True  # Command executed successfully
        else:
            # If we found a block but not a command, show available commands
            if (block or port) and hasattr(selected_module, "command"):
                Log.parser("Command not found. Possible commands:")
                selected_module.command.list_commands()
            return False  # No command executed
        
    def _get_matching_block(self, parts):
        if not parts:
            return None
        for block in self.project.get_blocks():
            if block is not None:   
                if block.name.lower() == parts[0]:
                    return block
        return None
            
    def _get_matching_input(self, block, parts):
        if not parts:
            return None
        
        if parts[0] == "input":
            if len(parts) > 1:  # If we have a specific input port name
                Log.parser(f"Attempting to match {parts[1]} to an input port")
                for input_port in block.input.items():
                    if input_port.name.lower() == parts[1]:
                        return input_port  
            return block.input

        return None

    def _get_matching_output(self, block, parts):
        if not parts:
            return None
        if parts[0] == "output":
            if len(parts) > 1:  # If we have a specific output port name
                for output_port in block.output.items():
                    if output_port.name.lower() == parts[1]:
                        return output_port
            return block.output
    
        return None
    
    def _check_if_command(self, command_item):
        if command_item:
            return True
        return False

    def _get_command(self, selected_module, remaining_parts):
        """
        Attempt to find a command in the selected module's command controller
        that matches the next token in remaining_parts.
        """
        if not remaining_parts:
            return None, None
        if selected_module and hasattr(selected_module, "command"):
            for cmd_item in selected_module.command.get_commands():
                if cmd_item.name.lower() == remaining_parts[0]:
                    # Everything after the command name is considered an argument
                    return cmd_item, remaining_parts[1:]
        return None, None
