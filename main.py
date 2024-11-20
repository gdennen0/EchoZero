from message import Log
from listen import main_listen_loop
import json
from project import Project
from Command.command import Command
from Container.ContainerTypes.generic_container import GenericContainer

def fetch_saved_settings():
    with open("settings.json", "r") as file:
        settings = json.load(file)
        if not isinstance(settings, dict):
            Log.error("Settings file does not contain a valid dictionary")
            return {}
        # Log.info(f"Settings loaded: {settings}")
        return settings

def main():
    project = Project(fetch_saved_settings())  #initializes project with settings.json data
    command = Command(project)

    main_listen_loop(command)

if __name__ == "__main__":
    main()

