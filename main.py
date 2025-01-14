from src.Utils.message import Log
from src.Project.project import Project
from src.Command.command_parser import CommandParser
from src.Utils.tools import prompt
from src.API.APIController import APIController
def main():
    project = Project() # main data structure 
    api_controller = APIController()
    api_controller.add_interface()

if __name__ == "__main__":
    main()


