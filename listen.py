import sys
from tools import prompt
from message import Log
from Command.command_parser import CommandParser

def main_listen_loop(parser):
    Log.info("Listening for commands")
    while True:
        user_input = prompt("Enter command: ")
        parser.parse_and_execute(user_input)