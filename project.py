import json
from Model.main_model import Model
from message import Log
from tools import prompt, check_project_path, prompt_selection, yes_no_prompt
import os
import datetime

# PROJECT_DIR_BYPASS = "/Users/gdennen/Desktop/Testing"
PROJECT_DIR_BYPASS = None
RECENT_PROJECTS_FILE = "recent_projects.json"

class Project:
    def __init__(self, settings):
        self.settings = settings
        self.dir = None
        self.name = None
        self.model = Model()
        self.loaded = False
        self.version = settings["APPLICATION_VERSION"]

        self.initialize()
        Log.info(f"Initialized Project")

    def initialize(self):
        options = {'load': self.load, 'new': self.new}
        while True:
            response = prompt("Do you want to load an existing project or create a new one? (load/new): ").lower()
            if response in options:
                options[response]()
                break
            Log.error("Invalid input. Please enter 'load' or 'new'.")

    def generate_folders(self):
        # Ensure both 'Data' and 'Audio' folders exist
        audio_path = os.path.join(self.dir, 'Audio')

        if not os.path.exists(audio_path):
            os.makedirs(audio_path)
            Log.info("Created 'Audio' folder inside the project dir.")

    def new(self, dir=PROJECT_DIR_BYPASS, name=None):
        if dir is not None:
            self.dir = dir
            Log.info(f"Set project dir to '{dir}'")
        elif self.dir is None:
            while True:
                self.dir = prompt("Please specify a project dir: ")
                if os.dir.exists(self.dir):
                    break
                else:
                    Log.error("The specified dir does not exist. Please try again.")
        if self.name is None:
            self.name = prompt("Please specify a project name: ")

        self.model.reset()
        self.generate_folders()
        Log.special(f"New project creation complete!")

    def load(self):
        try:
            recent_projects = self._load_recent_projects()
            if len(recent_projects) > 0:
                for i, object in enumerate(recent_projects):
                    Log.info(f"{i}: {object}")

                while True:
                    selection = prompt(f"Please enter the key or index for your selection or enter path of project: ")
                    if selection.isdigit():
                        index = int(selection)
                        path = recent_projects[index]
                    else:
                        path = selection

                    if not check_project_path(path):
                        Log.error("Invalid path. Please try again.")
                        continue
                    else:
                        break  # Exit the loop if a valid path is provided
            else:
                while True:
                    path = prompt(f"Please enter the path of project: ")
                    if not check_project_path(path):
                        Log.error("Invalid path. Please try again.")
                        continue
                    else:
                        break  # Exit the loop if a valid path is provided

                    
            with open(path, 'r') as file:
                project_data = json.load(file)
                project_info = project_data.get('project', {})
                self.dir = project_info.get('directory', self.dir)
                self.version = project_info.get('version', self.version)
                self.name = project_info.get('name', self.name)
                
            self.model.reset()
            self.model.deserialize(path)
            self._save_recent_project(path)
            Log.info(f"Loaded project from {path}")
        except FileNotFoundError:
            Log.error(f"File not found: {path}")
        except json.JSONDecodeError:
            Log.error(f"Error decoding JSON from file: {path}")
        except Exception as e:
            Log.error(f"An unexpected error occurred: {str(e)}")

    def save(self):
        if not self.dir or not self.name:
            Log.error("Project directory or name is not set. Cannot save the project.")
            return

        file_name = f"{self.name}.json"
        file_path = os.path.join(self.dir, file_name)
        if os.path.exists(file_path):
            overwrite = yes_no_prompt(f"A save file named '{file_name}' already exists. Do you want to overwrite it? (yes/no): ")
            if overwrite != 'yes':
                Log.warning("Save operation cancelled.")
                return
        self._save_to_file(file_path)
        Log.info(f"Project and model data saved successfully in '{file_name}'.")

    def save_as(self, name=None):
        if not name:
            name = prompt("What would you like to save the project as?: ")
        if not self.dir:
            Log.error("Project directory is not set. Cannot save the project.")
            return

        self.name = name
        file_name = f"{self.name}.json"
        file_path = os.path.join(self.dir, file_name)
        if os.path.exists(file_path):
            overwrite = prompt(f"A save file named '{file_name}' already exists. Do you want to overwrite it? (yes/no): ")
            if overwrite != 'yes':
                Log.warning("Save operation cancelled.")
                return
        self._save_to_file(file_path)
        Log.info(f"Project and model data saved successfully as '{file_name}'.")

    def _save_to_file(self, file_path):
        with open(file_path, 'w') as file:
            project_data = {
                'project': {
                    'directory': self.dir,
                    'version': self.version,
                    'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                    'time': datetime.datetime.now().strftime("%H:%M:%S"),
                    'name': self.name,
                },
                'audio': self.model.serialize()
            }
            json.dump(project_data, file, indent=4, separators=(',', ': '))  # Added separators for better distinction of dict items
        self._save_recent_project(file_path)
        
    def _save_recent_project(self, path):
        recent_projects = self._load_recent_projects()
        if path not in recent_projects:
            recent_projects.append(path)
        with open(RECENT_PROJECTS_FILE, 'w') as file:
            json.dump(recent_projects, file, indent=4)

    def _load_recent_projects(self):
        valid_paths = []
        try:
            if os.path.exists(RECENT_PROJECTS_FILE):
                with open(RECENT_PROJECTS_FILE, 'r') as file:
                    recent_projects = json.load(file)
                    for path in recent_projects:
                        if os.path.exists(path):
                            valid_paths.append(path)
        except (IOError, json.JSONDecodeError) as e:
            Log.error(f"RECENT_PROJECTS_FILE is formatted incorrectly, bypassing... {e}")
            valid_paths = []
            if yes_no_prompt("The recent projects file is invalid. Do you want to regenerate/reset the file? (yes/no): "):
                with open(RECENT_PROJECTS_FILE, 'w') as file:
                    json.dump([], file, indent=4)
                Log.info("RECENT_PROJECTS_FILE has been reset.")
        return valid_paths
