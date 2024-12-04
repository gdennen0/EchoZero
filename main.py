from message import Log
from listen import main_listen_loop
import json
from project import Project
from Command.command_parser import CommandParser
def main():
    project = Project()  #initializes project with settings.json data
    parser = CommandParser(project)
    main_listen_loop(parser)

if __name__ == "__main__":
    main()

