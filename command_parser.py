from message import Log

class CommandParser:
    def __init__(self, command):
        self.command_modules = command.command_modules
        self.commands = None
        Log.info(f"Parser: Registering commands in parser")

    def parse_and_execute(self, input_string):
        try:
            parts = input_string.lower().split()
            Log.info(f"Parse Parts: {parts}")
            if not parts:
                Log.error("No command provided")
                return

            main_module = None
            sub_module = None
            command_item = None
            args = []

            for i, part in enumerate(parts):
                Log.info(f"Parsing part: {part}")
                for module in self.command_modules:
                    if part == module.name.lower(): # Match a module name
                        if not main_module: # if there isnt a main module
                            main_module = next((mod for mod in self.command_modules if mod.name.lower() == part), None) #matches the module 
                            Log.info(f"Latched base command module part {main_module.name.lower()}")
                    elif main_module and not sub_module:  
                        sub_module = next((mod for mod in main_module.sub_modules if mod.name.lower() == part), None)
                        command_item = next((cmd for cmd in main_module.commands if cmd.name.lower() == part), None)

                    elif (main_module and part in [cmd.name.lower() for cmd in main_module.commands]) or (sub_module and part in [cmd.name for cmd in sub_module.commands]):
                        if main_module:
                            if sub_module:
                                command_item = next((cmd for cmd in sub_module.commands if cmd.name.lower() == part), None)
                            else:
                                command_item = next((cmd for cmd in main_module.commands if cmd.name.lower() == part), None)
                        else:
                            command_item = next((cmd for cmd in self.commands if cmd.name.lower() == part), None)
                        Log.info(f"PARSER: Matched command: {part}")

                    add logic to collect args


            Log.info(f"Parsed results: main_module: {main_module.name if main_module else 'None'} sub_module: {sub_module.name if sub_module else 'None'} command: {command_item.name if command_item else 'None'}")

            if command_item:
                command_item.command()
            else:
                Log.info(f"No command found")

        except Exception as e:
            Log.error(f"Error parsing command '{input_string}': {str(e)}")