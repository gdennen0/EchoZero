from src.Utils.message import Log
from src.Project.project import Project
from src.Command.command_parser import CommandParser
from src.Utils.tools import prompt
from PyQt5 import QtWidgets

# def run_qt_app(app):
#     """Function to run the Qt application in a separate thread."""
#     app.exec_()

def main():
    app = QtWidgets.QApplication([])
    project = Project() # main data structure 
    parser = CommandParser(project) # command parser module
    project.set_parser(parser)  # Store parser reference in the project
    
    Log.info("Listening for commands")
    while True:
        user_input = prompt("Enter command: ")
        parser.parse_and_execute(user_input)

    # Call exec_() directly in the main thread
    app.exec_()

if __name__ == "__main__":
    main()
