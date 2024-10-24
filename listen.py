import sys
from tools import prompt
from message import Log
from command_parser import CommandParser

def main_listen_loop(command):
    Log.info("Listening for commands")
    parser = CommandParser(command)
    while True:
        user_input = prompt("Enter command: ")
        parser.parse_and_execute(user_input)