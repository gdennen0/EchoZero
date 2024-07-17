import sys
from tools import prompt
from message import Log
from Command.command_parser import CommandParser

def main_listen_loop(command):
    Log.info("Listening for commands. Type 'exit' to quit.")
    parser = CommandParser(command)
    while True:
        user_input = prompt("Enter command: ")
        if user_input.lower() == 'exit':
            Log.special("Exiting application.")
            break
        parser.parse_and_execute(user_input)