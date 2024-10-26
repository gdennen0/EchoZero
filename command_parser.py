from message import Log

class CommandParser:
    def __init__(self, command):
        self.command_modules = command.command_modules
        self.commands = None
        Log.parser(f"Parser: Registering commands in parser")

    def parse_and_execute(self, input_string):
        parts = input_string.lower().split()
        Log.parser(f"Parts: {parts}")
        if not parts:
            Log.error("No command provided")
            return

        main_module = None
        sub_module = None
        command_item = None
        block = None
        args = []

        # Find main module
        main_module = next((mod for mod in self.command_modules if mod.name.lower() == parts[0]), None)
        if main_module:
            parts = parts[1:]
            Log.parser(f"Matched main module: {main_module.name}")
        else:
            # New check for container
            Log.error(f"Main module '{parts[0]}' not found")
            return
        
        # Find sub module, container, or command
        if parts:
            sub_module = next((mod for mod in main_module.sub_modules if mod.name.lower() == parts[0]), None)
            if sub_module:
                parts = parts[1:]
                Log.parser(f"Matched sub module: {sub_module.name}")
            else:
                # Check for container
                container = next((cont for cont in main_module.containers.values() if cont.name.lower() == parts[0]), None)
                if container:
                    parts = parts[1:]
                    Log.parser(f"Matched container: {container.name}")
                    # Check for block within container
                    if parts:
                        block = next((blk for blk in container.blocks.values()  if blk.name.lower() == parts[0]), None)
                        if block:
                            parts = parts[1:]
                            Log.parser(f"Matched block: {block.name}")
                else:
                    command_item = next((cmd for cmd in main_module.commands if cmd.name.lower() == parts[0]), None)
                    if command_item:
                        parts = parts[1:]
                        # Log.parser(f"Matched command in main module: {command_item.name}")

        # Find command in sub module, container, or block if not found yet
        if not command_item and parts:
            if sub_module:
                command_item = next((cmd for cmd in sub_module.commands if cmd.name.lower() == parts[0]), None)
            elif container:
                if block:
                    command_item = next((cmd for cmd in block.commands if cmd.name.lower() == parts[0]), None)
                else:
                    command_item = next((cmd for cmd in container.commands if cmd.name.lower() == parts[0]), None)
            
            if command_item:
                parts = parts[1:]
                # Log.parser(f"Matched command in {'sub module' if sub_module else 'container' if container else 'block'}: {command_item.name}")

        # Remaining parts are args
        args = parts

        Log.parser(f"Parse result - Main module: {main_module.name}, Sub module: {sub_module.name if sub_module else 'None'}, "
                f"Container: {container.name if container else 'None'}, Block: {block.name if block else 'None'}, "
                f"Command: {command_item.name if command_item else 'None'}, Args: {args}")

        if command_item:
            command_item.command(*args)
        else:
            Log.parser(f"No command found")

        # except Exception as e:
            # Log.error(f"Error parsing command '{input_string}': {str(e)}")