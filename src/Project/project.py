import json
from src.Utils.message import Log
from src.Utils.tools import prompt, check_project_path, prompt_selection, generate_unique_name
import os
import sys
from src.Command.command_controller import CommandController
import importlib.util
import json
from collections import defaultdict, deque
import zipfile
import shutil

PROJECT_VERSION = "0.0.1"
PROJECT_DIR_BYPASS = None
RECENT_PROJECTS_FILE = os.path.join(os.getcwd(), "data", "recent_projects.json")
class Project():
    """
    Project is the main container for all blocks and data in the project.

    Project is the main container for all blocks and data in the project.

    Attributes:
        name (str): The name of the project.
        save_directory (str): The directory where the project will be saved.
        application_directory (str): The directory where the application is running.
        loaded (bool): A flag indicating whether the project is loaded.
        block_types (list): A list of available block types.
        blocks (list): A list of blocks in the project.
        command (CommandController): The command controller for the project.

    Methods:
        __init__(): Initializes a new instance of the Project class.
        initialize_project(): Prompts the user to load an existing project or create a new one.
        load_block_types(): Loads the available block types.
        load(): Loads a project from a file.
        new(): Creates a new project.
        recent(): Loads a recent project.
        add_block(block_type, block_name): Adds a new block to the project.
        list_commands(): Lists all available commands.
        list_block_types(): Lists all available block types.
        list_blocks(): Lists all blocks in the project.
        save(): Saves the project to a file.
        save_as(): Prompts the user to save the project with a new name and directory.
        build_project_data(): Builds the project data for saving.
        add_recent_project(project_path): Adds a project to the list of recent projects.
    """

    def __init__(self):
        super().__init__()
        self.name = None
        self.save_directory = None
        self.application_directory = os.getcwd()
        self.loaded = False
        self.block_types = []
        self.blocks = []

        self.command = CommandController()

        self.load_block_types()

        self.command.add("load", self.load)
        self.command.add("new", self.new)
        self.command.add("save", self.save)
        self.command.add("save_as", self.save_as)
        self.command.add("saveas", self.save_as)
        self.command.add("list_commands", self.list_commands)
        self.command.add("listcommands", self.list_commands)
        self.command.add("list_block_types", self.list_block_types)
        self.command.add("listblocktypes", self.list_block_types)
        self.command.add("list_blocks", self.list_blocks)
        self.command.add("listblocks", self.list_blocks)
        self.command.add("add_block", self.add_block)
        self.command.add("addblock", self.add_block)
        self.command.add("add", self.add_block)

        self.initialize_project()

    def initialize_project(self):
        options = {
            'l': self.load, 
            'n': self.new, 
            'r': self.recent, 
            'load': self.load, 
            'new': self.new, 
            'recent': self.recent
        }
        while True:
            response = prompt("Do you want to load an existing project or create a new one? (l/load, n/new, r/recent): ")
            if response: 
                response = response.lower()
            if response in options:
                if options[response]():
                    break
            else:
                Log.error("Invalid input. Please enter 'load', 'new', or 'recent' (or 'l', 'n', 'r').")

    def recent(self):
        with open(RECENT_PROJECTS_FILE, 'r') as file:
            recent_projects = json.load(file)
            if recent_projects:
                project_path = prompt_selection("Please select a project: ", recent_projects)
                self.load_project(project_path)
                return True
            else:
                Log.info("No recent projects found.")
                return False

    def new(self):
        self.name = "Untitled"
        Log.special(f"New project creation complete!")
        return True


    def load(self, load_path=None):
        if not load_path:
            load_path = prompt("Please enter the path of the project to load: ")
        if not check_project_path(load_path):
            Log.error("Invalid path. Please try again.")
            return False
        self.load_project(load_path)
        self.add_recent_project(load_path)
        Log.info(f"Project '{self.name}' loaded successfully from {load_path}")
        return True


    def load_project(self, file_path):
        projectdata_dir = os.path.join(self.application_directory, "tmp", "projectdata")
        if os.path.exists(projectdata_dir):
            for root, dirs, files in os.walk(projectdata_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(projectdata_dir)

        with zipfile.ZipFile(file_path, 'r') as project_file:
            project_file.extractall(projectdata_dir)

            metadata_path = os.path.join(projectdata_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as metadata_file:
                    project_data = json.load(metadata_file)
                    Log.info(f"-"*50)
                    Log.info(f"Project Name: {project_data.get('name')}")
                    Log.info(f"Save Directory: {project_data.get('save_directory')}")
                    Log.info(f"-"*50)

                    self.set_name(project_data.get("name"))
                    self.set_save_directory(project_data.get("save_directory"))

                    blocks_list = project_data.get("blocks", [])
                    for block in blocks_list:
                        block_type = block.get("type")
                        block_name = block.get("name")
                        self.add_block(type=block_type, name=block_name)
                    Log.info("Completed adding blocks to project")
                    Log.info(f"-"*50)

            else:
                Log.error(f"Metadata file not found in {projectdata_dir}")
                
        blocks_dir = os.path.join(projectdata_dir, 'blocks')
        if os.path.exists(blocks_dir):
            for block_folder in os.listdir(blocks_dir):
                block_folder_path = os.path.join(blocks_dir, block_folder)
                if os.path.isdir(block_folder_path):
                    metadata_path = os.path.join(block_folder_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as metadata_file:
                            Log.info(f"Begin Loading block: {block_folder}")
                            block_metadata = json.load(metadata_file)
                            for block in self.blocks:
                                if block.name == block_metadata.get("name"):
                                    block.load(block_folder_path)
                                    Log.info(f"Loaded block: {block.name}")
                    else:
                        Log.error(f"Metadata file not found in {block_folder_path}")
        else:
            Log.error(f"Blocks directory not found in {projectdata_dir}")  
            
    def load_block(self, name, info):
        for block in self.blocks:
            if block.name == name:
                block.load(info)
                Log.info(f"Block '{block.name}' loaded successfully.")
                return

    def list_commands(self):
        indent_unit = '    '  # Define the indentation unit (e.g., 4 spaces or any other characters)
        Log.info(f"Project: {self.name}")
        # Register Project-level commands
        for cmd in self.command.get_commands():
            Log.info(f"{indent_unit}Command: '{cmd.name}'")
        for block in self.blocks:
            Log.info(f"{indent_unit}Block: {block.name}")
            # Register block-level commands
            for cmd in block.command.get_commands():
                Log.info(f"{indent_unit * 2}Command: '{cmd.name}'")
            for input in block.input.items():
                Log.info(f"{indent_unit * 2}Input: {input.name}")
                for cmd in input.command.get_commands():
                    Log.info(f"{indent_unit * 3}Command: '{cmd.name}'")
            for output in block.output.items():
                Log.info(f"{indent_unit * 2}Output: {output.name}")
                for cmd in output.command.get_commands():
                    Log.info(f"{indent_unit * 3}Command: '{cmd.name}'")

    def save(self):
        if not self.save_directory:
            Log.error("No save directory set. Please set one before saving.")
            return
        
        project_root = os.path.join(self.save_directory, self.name)
        if not os.path.exists(project_root):
            os.makedirs(project_root, exist_ok=True)

        project_json = {
            "name": self.name,
            "version": PROJECT_VERSION,
            "save_directory": self.save_directory,
            "loaded": self.loaded,
            "blocks": [{"name": block.name, "type": block.type} for block in self.blocks]
        }
        
        project_json_path = os.path.join(project_root, "metadata.json")
        with open(project_json_path, 'w') as file:
            json.dump(project_json, file, indent=4)
        
        blocks_dir = os.path.join(project_root, "blocks")
        os.makedirs(blocks_dir, exist_ok=True)


        for block in self.blocks:
            block_name = block.name
            block_dir = os.path.join(blocks_dir, block_name)
            os.makedirs(block_dir, exist_ok=True)
            metadata = block.get_metadata()
            block.save(block_dir)

            metadata_path = os.path.join(block_dir, "metadata.json")
            with open(metadata_path, 'w') as file:
                json.dump(metadata, file, indent=4)


        # Define the path for the zip file with .ez extension
        zip_path = os.path.join(self.save_directory, f"{self.name}.ez")

        # Create a zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, _, files in os.walk(project_root):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, start=project_root)
                    zip_ref.write(full_path, arcname=rel_path)

        # Optional: Remove the unzipped folder after creating the .ez file
        shutil.rmtree(project_root)
        self.add_recent_project(zip_path)

        Log.info(f"Project saved successfully to {zip_path}")



    def save_as(self):
        self.name = prompt("Please enter the name of the project: ")
        self.save_directory = prompt("Please enter the directory to save the project to: ")

        self.save()

        # Start of Selection
    def add_recent_project(self, project_path):
        with open(RECENT_PROJECTS_FILE, 'r') as file:
            recent_projects = json.load(file)
        if not isinstance(recent_projects, list):
            recent_projects = []
        if project_path in recent_projects:
            recent_projects.remove(project_path)
        recent_projects.insert(0, project_path)
        with open(RECENT_PROJECTS_FILE, 'w') as file:
            json.dump(recent_projects, file, indent=4)

    def get_main_dir(self):
        try:
            # Get the absolute path of the main.py file
            main_file_path = os.path.abspath(sys.argv[0])
            # Get the directory of the main.py file
            main_dir = os.path.dirname(main_file_path)
            return main_dir
        except Exception as e:
            Log.error(f"CRITICAL ERROR: Failed to get main directory: {e}")
            return None

    def load_block_types(self):
        base_path = os.path.join(self.application_directory, "src/Project/Block/BlockTypes")
        for dir in os.listdir(base_path):
            dir_path = os.path.join(base_path, dir)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".py") and file_name != "__init__.py":
                        try:
                            block_name = file_name[:-3]
                            file_path = os.path.join(dir_path, file_name)

                            spec = importlib.util.spec_from_file_location(block_name, file_path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            cls = getattr(module, block_name)
                            self.add_block_type(cls)

                        except FileNotFoundError as e:
                            Log.error(f"File not found: {file_path} - {e}")
                        except ImportError as e:
                            Log.error(f"Failed to import module '{block_name}' from '{file_path}': {e}")
                        except AttributeError as e:
                            Log.error(f"Module '{block_name}' does not have class '{block_name}': {e}")
                        except Exception as e:
                            Log.error(f"Unexpected error loading block '{block_name}' from '{file_path}': {e}")

    def add_block_type(self, block_type):
        self.block_types.append(block_type)
        Log.info(f"Initialized block type: {block_type.name}")

    def add_block(self, type=None, name=None):
        if not type:
            block_type = prompt_selection("Please select the block type: ", self.block_types)
            if block_type is None:
                Log.info("Block creation exited")
                return
            block = block_type()
            block.set_parent(self)
            if not name:
                new_name = generate_unique_name(block_type.type, self.blocks)
                block.set_name(new_name)
            else:
                block.set_name(name)
            self.blocks.append(block)
            Log.info(f"Added block: {new_name}")
        else:
            for block_type in self.block_types:
                if block_type.type == type:
                    block = block_type()
                    block.set_parent(self)
                    if not name:
                        new_name = generate_unique_name(block_type.type, self.blocks)
                        block.set_name(new_name)
                    else:
                        block.set_name(name)
                    self.blocks.append(block)
                    Log.info(f"Added block: {name}")

    
    def get_block(self, block_name):
        for block in self.blocks:
            if block.name == block_name:
                return block
            
        return None
        
    def remove_block(self, block_name):
        if block_name in self.blocks:
            del self.blocks[block_name]
        else:
            raise ValueError(f"Block with name '{block_name}' not found in container")

    def list_block_types(self):
        try:        
            for block_type in self.block_types:
                Log.info(f"{block_type.name}")
        except Exception as e:
            Log.error(f"Failed to list block types: {e}")
            
    def list_blocks(self):
        counter = 1
        Log.info(f"Listing current blocks in project {self.name}")
        for block in self.blocks:
            Log.info(f"{counter}: {block.name}")

            counter += 1

    def get_blocks(self):
        return self.blocks
    
    def set_name(self, name):
        self.name = name
        Log.info(f"Set project name to {name}")

    def get_name(self):
        return self.name
    
    def set_save_directory(self, save_directory):
        self.save_directory = save_directory
        Log.info(f"Set project save directory to {save_directory}")

    def get_save_directory(self):
        return self.save_directory