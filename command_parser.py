from message import Log

class CommandParser:
    def __init__(self, command):
        self.command_modules = command.command_modules
        self.commands = None
        Log.parser("Parser: Registering commands in parser")

    def parse_and_execute(self, input_string):
        # Initialize variables
        main_module = None
        container = None
        block = None
        part = None
        port = None
        command_item = None
        args = []

        # Split the input string into parts
        all_input_parts = input_string.lower().split()
        remaining_parts = all_input_parts.copy()

        # Parse main module
        main_module = self._get_matching_module(remaining_parts)
        if main_module:
            Log.parser(f"Matched main module: {main_module.name}")
            remaining_parts.pop(0)  # Remove main module from the list

            # Parse container
            container = self._get_matching_container(main_module, remaining_parts)
            if container:
                Log.parser(f"Matched container: {container.name}")
                remaining_parts.pop(0)  # Remove container from the list
                selected_module = container
                # Parse block
                block = self._get_matching_block(container, remaining_parts)
                if block:
                    Log.parser(f"Matched block: {block.name}")
                    remaining_parts.pop(0)  # Remove block from the list
                    selected_module = block
                    # Parse part or port
                    part = self._get_matching_part(block, remaining_parts)
                    if part:
                        Log.parser(f"Matched part: {part.name}")
                        remaining_parts.pop(0)  # Remove part from the list
                        selected_module = part
                    port = self._get_matching_port(block, remaining_parts)
                    if port:
                        Log.parser(f"Matched port: {port.name}")
                        remaining_parts.pop(0)  # Remove port type from the list (input_port or output_port)
                        remaining_parts.pop(0)  # Remove port from the list
                        selected_module = port
        
        command_item, args = self._get_command(selected_module, remaining_parts)
        if command_item:
            Log.parser(f"Matched command: {command_item.name}")
            Log.parser(
                f"Parse result - Main module: {main_module.name}, Container: {container.name if container else 'None'}, "
                f"Block: {block.name if block else 'None'}, Part: {part.name if part else 'None'}, Port: {port.name if port else 'None'}, Command: {command_item.name}, Args: {args}"

            )
            command_item.command(*args)
        else:
            Log.parser("No command found")

    def _get_matching_module(self, parts):
        if not parts:
            return None
        return next((mod for mod in self.command_modules if mod.name.lower() == parts[0]), None)

    def _get_matching_container(self, module, parts):
        if not parts:
            return None
        return next((cont for cont in module.containers.values() if cont.name.lower() == parts[0]), None)

    def _get_matching_block(self, container, parts):
        if not parts:
            return None
        return next((blk for blk in container.blocks.values() if blk.name.lower() == parts[0]), None)

    def _get_matching_part(self, block, parts):
        if not parts:
            return None
        part = next((prt for prt in block.parts if prt.name.lower() == parts[0]), None)
        return part

    def _get_matching_port(self, block, parts):
        # Check if 'ports' attribute exists
        if parts[0] == "input_port":
            port = next((prt for prt in block.input_ports if prt.name.lower() == parts[1]), None)
        elif parts[0] == "output_port":
            port = next((prt for prt in block.output_ports if prt.name.lower() == parts[1]), None)
        else:
            port = None
        return port
    
    def _check_if_command(self, command_item):
        if command_item:
            return True
        return False

    def _get_command(self, selected_module, remaining_parts):
        if not remaining_parts:
            return None, None
        Log.parser(f"attempting to match command '{remaining_parts[0]}' within module {selected_module.name} ")
        for command in selected_module.commands:
            Log.parser(f"Command: {command.name}")
            if command.name.lower() == remaining_parts[0]:
                return command, remaining_parts[1:]
        return None, None
