from message import Log
from listen import main_listen_loop
import json
from project import Project
from Command.command import Command
from Model.main_model import Model

def load_settings():
    with open("settings.json", "r") as file:
        settings = json.load(file)
        Log.info(f"Settings loaded: {settings}")
        return settings

def main():
    project = Project(load_settings())
    command = Command(project, load_settings())
    main_listen_loop(command)

if __name__ == "__main__":
    main()

