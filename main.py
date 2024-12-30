from src.Utils.message import Log
from src.Project.project import Project
from src.Command.command_parser import CommandParser
from src.Utils.tools import prompt

def main():
    project = Project() # main data structure 
    parser = CommandParser(project) # command parser module
    Log.info("Listening for commands")
    while True:
        user_input = prompt("Enter command: ")
        parser.parse_and_execute(user_input)

if __name__ == "__main__":
    main()


