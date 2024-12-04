from message import Log

class CommandParser:
    def __init__(self, project):
        self.project = project
        self.commands = None

    def parse_and_execute(self, input_string):
        # Initialize variables
        block = None
        port = None
        command_item = None
        args = []

        # Split the input string into parts
        all_input_parts = input_string.lower().split()
        Log.parser(f"All input parts: {all_input_parts}")
        remaining_parts = all_input_parts.copy()

        selected_module = self.project
        block = self._get_matching_block(remaining_parts)
        if block:
            Log.parser(f"Matched block: {block.name}")
            remaining_parts.pop(0)  # Remove block from the list
            selected_module = block
            # Parse part or port
            port = self._get_matching_port(block, remaining_parts)
            if port:
                Log.parser(f"Matched port: {port.name}")
                remaining_parts.pop(0)  # Remove port type from the list (input_port or output_port)
                selected_module = port
            else:
                Log.parser("No port found.")
        else:
            Log.parser("No block found.")
        command_item, args = self._get_command(selected_module, remaining_parts)
        if command_item:
            Log.parser(f"Matched command: {command_item.name}")
            command_item.command(*args)
        else:
            Log.parser("No command found.")
            if hasattr(selected_module, "command"):
                Log.parser("Please select from the following commands instead:")
                selected_module.command.list_commands()

    def _get_matching_block(self, parts):
        if not parts:
            return None
        for block in self.project.blocks:
            if block.name.lower() == parts[0]:
                return block
            else:
                return None
            
    def _get_matching_port(self, block, parts):
        # Check if 'ports' attribute exists
        if '.' in parts[0] and len(parts[0].split('.')) == 2: # if the port has a sub command
            if parts[0] == "port.input" or parts[0] == "port.output":
                port = None
                Log.parser(f"Attempting to match {parts[1]} to input an input port")
                for port_item in block.port.items():
                    if port_item.name.lower() == parts[1]: # if a port.name atrib matches the input part
                        if port_item.type == "input" and parts[0] == "port.input":
                            port = port_item
                            return port
                        elif port_item.type == "output" and parts[0] == "port.output":
                            port = port_item
                            return port
        elif parts[0] == "port":
            Log.parser(f"Matched command part: '{parts[0]}' to '{block.name}s' port controller")
            port_controller = block.port
            return port_controller
        else:
            return None
    
    def _check_if_command(self, command_item):
        if command_item:
            return True
        return False

    def _get_command(self, selected_module, remaining_parts):
        if not remaining_parts:
            return None, None
        if selected_module:
            Log.parser(f"attempting to match command '{remaining_parts[0]}' within module {selected_module.name} ")
            for command in selected_module.command.get_commands():
                # Log.parser(f"Command: {command.name}")
                if command.name.lower() == remaining_parts[0]:
                    return command, remaining_parts[1:]
        return None, None
