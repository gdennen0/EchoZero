# src/Input/cli_input.py
from Project.Command.command_parser import CommandParser
from Utils.message import Log
import threading

class CLIInterface:
    def __init__(self, command_parser):
        self.command_parser = command_parser
        self.running = False
        self.thread = threading.Thread(target=self.listen, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()
        Log.info("CLI Input Handler started.")

    def listen(self):
        while self.running:
            try:
                user_input = input("Enter command: ")
                if user_input.lower() in ['exit', 'quit']:
                    self.running = False
                    Log.info("CLI Input Handler stopping.")
                    break
                self.command_parser.parse_and_execute(user_input)
            except EOFError:
                break

    def stop(self):
        self.running = False
        self.thread.join()
        Log.info("CLI Input Handler stopped.")