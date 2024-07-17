import argparse
from message import Log

class CommandParser:
    def __init__(self, command_obj):
        self.command_obj = command_obj
        self.parser = argparse.ArgumentParser(description="Command Line Interface")
        self.subparsers = self.parser.add_subparsers(dest="command")

        self._register_commands()

    def _register_commands(self):
        for cmd_name, cmd_func in self.command_obj.command_registry.commands.items():
            subparser = self.subparsers.add_parser(cmd_name)
            subparser.set_defaults(func=cmd_func)

    def parse_and_execute(self, input_string):
        try:
            args = self.parser.parse_args(input_string.split())
            if hasattr(args, 'func'):
                args.func()
            else:
                Log.error(f"No function associated with command: {args.command}")
        except Exception as e:
            Log.error(f"Error parsing command: {str(e)}")