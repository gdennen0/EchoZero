from src.Utils.message import Log
from src.Project.project import Project
from src.Command.command_parser import CommandParser
from src.Utils.tools import prompt
from PyQt5 import QtWidgets
import threading
import time

def run_qt_app(app):
    """Function to run the Qt application in a separate thread."""
    app.exec_()

def main():
    app = QtWidgets.QApplication([])
    project = Project() # main data structure 
    parser = CommandParser(project) # command parser module

    qt_thread = threading.Thread(target=run_qt_app, args=(app,))
    qt_thread.daemon = True  # Ensure the thread exits when the main program exits
    qt_thread.start()

    Log.info("Listening for commands")
    while True:
        user_input = prompt("Enter command: ")
        parser.parse_and_execute(user_input)



if __name__ == "__main__":
    main()
