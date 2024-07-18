import argparse
from message import Log

class CommandParser:
    def __init__(self, modules):
        self.parser = argparse.ArgumentParser(description="Command Line Interface")
        self.subparsers = self.parser.add_subparsers(dest="command")

        Log.info(f"Parser: Registering commands in parser")
        for module in modules:
            Log.info(f"Parser: Registering module: {module.name}")
            subparser = self.subparsers.add_parser(module.name)
            subparser.set_defaults(func=self._create_module_handler(module))

    def _create_module_handler(self, module):
        def handler(args):
            subparser = argparse.ArgumentParser(prog=f"{module.name}")
            subparsers = subparser.add_subparsers(dest="subcommand")
            for cmd_item in module.commands:
                Log.info(f"Parser: Registering command: {cmd_item.name} in module: {module.name}")
                cmd_subparser = subparsers.add_parser(cmd_item.name)
                cmd_subparser.set_defaults(func=cmd_item.command)
            for sub_module in module.sub_modules:
                Log.info(f"Parser: Registering sub-module: {sub_module.name} in module: {module.name}")
                sub_module_parser = subparsers.add_parser(sub_module.name)
                sub_module_parser.set_defaults(func=self._create_module_handler(sub_module))
            sub_args = subparser.parse_args(args._get_args())
            if hasattr(sub_args, 'func'):
                sub_args.func()
            else:
                Log.error(f"No function associated with subcommand: {sub_args.subcommand}")
        return handler

    def parse_and_execute(self, input_string):
        try:
            args = self.parser.parse_args(input_string.split())
            Log.info(f"Parse args: {args}")
            if hasattr(args, 'func'):
                args.func(args)
            else:
                Log.error(f"No function associated with command: {args.command}")
        except SystemExit as e:
            Log.error(f"SystemExit occurred when parsing command '{input_string}': {str(e)}")
        except Exception as e:
            Log.error(f"Error parsing command '{input_string}': {str(e)}")