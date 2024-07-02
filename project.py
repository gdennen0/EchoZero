import json
from Model.main_model import Model
from message import Log
from tools import prompt
import os

PROJECT_DIR_BYPASS = "/Users/gdennen/Desktop/Testing"

class Project:
    def __init__(self, settings):
        self.settings = settings
        self.path = None
        self.name = None
        self.model = Model()
        self.loaded = False

        self.initialize()
        Log.info(f"Initialized Project")

    def initialize(self):
        while True:
            response = prompt("Do you want to load an existing project or create a new one? (load/new): ").lower()
            if response == 'load':
                self.load(self.path)
                break
            elif response == 'new':
                self.new()
                break
            else:
                Log.error("Invalid input. Please enter 'load' or 'new'.")

    def generate_folders(self):
        # Ensure both 'Data' and 'Audio' folders exist
        audio_path = os.path.join(self.path, 'Audio')

        if not os.path.exists(audio_path):
            os.makedirs(audio_path)
            Log.info("Created 'Audio' folder inside the project dir.")

    def new(self, path=PROJECT_DIR_BYPASS, name=None):
        if path is not None:
            self.path = path
            Log.info(f"Set project dir to '{path}'")
        elif self.path is None:
            while True:
                self.path = prompt("Please specify a project path: ")
                if os.path.exists(self.path):
                    break
                else:
                    Log.error("The specified path does not exist. Please try again.")
        if self.name is None:
            self.name = prompt("Please specify a project name: ")

        self.model.reset()
        self.generate_folders()
        Log.special(f"New project creation complete!")

    def load(self, path):
        self.model.reset()
        self.model.deserialize(path)
        Log.info(f"Loaded New Project")

    def save(self):
        with open('project_data.json', 'w') as file:
            project_data = {
                'project': {'path': self.path, 'name': self.name},
                'model': self.model.serialize()
            }
            json.dump(project_data, file)
        Log.info("Project and model data saved successfully.")


class ProjectObject:
    def __init__(self):
        self.path = None
        self.name = None