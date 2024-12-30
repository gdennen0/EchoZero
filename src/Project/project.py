import json
from src.Utils.message import Log
from src.Utils.tools import prompt, check_project_path, prompt_selection, generate_unique_name
import os
import sys
from src.Command.command_controller import CommandController
import importlib.util
import json
from collections import defaultdict, deque

PROJECT_DIR_BYPASS = None
RECENT_PROJECTS_FILE = os.path.join(os.getcwd(), "data", "recent_projects.json")
class Project():
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
        self.command.add("project_data", self.project_data)
        self.command.add("projectdata", self.project_data)

        self.initialize_project()
        # self.list_commands()

        # Start of Selection
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
            response = prompt("Do you want to load an existing project or create a new one? (l/load, n/new, r/recent): ").lower()
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
        Log.info(f"Starting to load project from '{file_path}'.")
        
        Log.info(f"Opening project file '{file_path}'.")
        with open(file_path, 'r') as file:
            project_data = json.load(file)
        Log.info("Project data loaded successfully.")

        dependencies = defaultdict(list)
        in_degree = defaultdict(int)

        self.name = project_data.get("name")
        self.save_directory = project_data.get("save_directory")

        Log.info("Building dependency graph.")
        for block_name, block_data in project_data['blocks'].items():
            # Add block based on its type
            Log.info(f"---> block_name: {block_name}")
            for block_type in self.block_types:
                if block_type.name == block_data.get('type'):
                    self.add_block(block_type_name=block_type.name)
                    break
            else:
                Log.warning(f"Unknown block type '{block_data.get('type')}' for block '{block_name}'.")
                continue  # Skip processing dependencies for unknown block types

            # Validate 'input' before accessing 'inputs'
            input_data = block_data.get('input')
            if not input_data or 'inputs' not in input_data:
                Log.warning(f"No input data for block '{block_name}'. Skipping dependency processing.")
                continue

            for input_item in input_data['inputs']:
                connected_output = input_item.get('connected_output')
                if connected_output:
                    dependent_block = connected_output.split('.')[0]  # Extract block name
                    dependencies[dependent_block].append(block_name)
                    in_degree[block_name] += 1
                    Log.debug(f"Block '{block_name}' depends on '{dependent_block}'.")
                else:
                    Log.info(f"No connected_output found for block '{block_name}'.")

        # Proceed with topological sort as before
        Log.info("Initializing queue with blocks that have no dependencies.")
        queue = deque([block for block in project_data['blocks'] if in_degree[block] == 0])
        Log.info(f"Initial load queue: {list(queue)}")
        loading_order = []

        while queue:
            current = queue.popleft()
            loading_order.append(current)
            Log.info(f"Processing block '{current}'.")
            for dependent in dependencies[current]:
                in_degree[dependent] -= 1
                Log.debug(f"Decremented in_degree of '{dependent}' to {in_degree[dependent]}.")
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    Log.info(f"Added block '{dependent}' to queue as its in_degree is now 0.")

        if len(loading_order) != len(project_data['blocks']):
            Log.error("Cyclic dependency detected!")
            raise Exception("Cyclic dependency detected!")

        Log.info(f"Loading blocks in the following order: {loading_order}")
        for block_name in loading_order:
            block_data = project_data['blocks'][block_name]
            self.load_block(block_name, block_data)
            

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
        if self.save_directory:
            save_path = os.path.join(self.save_directory, f"{self.name}.json")
            project_data = self.build_project_data()
            self.add_recent_project(save_path)
            with open(save_path, 'w') as file:
                json.dump(project_data, file, indent=4)
            Log.info(f"Project saved successfully to {save_path}")
        else:
            Log.error("No save directory set. Please set one before saving.")

    def save_as(self):
        self.name = prompt("Please enter the name of the project: ")
        self.save_directory = prompt("Please enter the directory to save the project to: ")

        self.save()

    def build_project_data(self):
        project_data = {
            "name" : self.name,
            "save_directory" : self.save_directory,
            "loaded" : self.loaded,
            "blocks": {block.name : block.save() for block in self.blocks}
        }
        return project_data

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

    def project_data(self):
        project_data = self.build_project_data()
        # formatted_data = project_data
        formatted_data = json.dumps(project_data, indent=4)
        Log.info(formatted_data)

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

    def add_block(self, block_type_name=None):
        if not block_type_name:
            block_type = prompt_selection("Please select the block type: ", self.block_types)
            if block_type is None:
                Log.info("Block creation exited")
                return
            block = block_type()
            block.set_parent(self)
            new_name = generate_unique_name(block_type.name, self.blocks)
            block.name =  new_name  

            self.blocks.append(block)
            Log.info(f"Added block: {new_name}")
        else:
            for block_type in self.block_types:
                if block_type.name == block_type_name:
                    block = block_type()
                    block.set_parent(self)
                    self.blocks.append(block)
                    Log.info(f"Added block: {block_type_name}")

    
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