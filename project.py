import json
from message import Log
from tools import prompt, check_project_path, prompt_selection, yes_no_prompt
import os
import datetime
import sys
from Command.command_controller import CommandController
import importlib.util

# PROJECT_DIR_BYPASS = "/Users/gdennen/Desktop/Testing"
PROJECT_DIR_BYPASS = None
RECENT_PROJECTS_FILE = "recent_projects.json"

class Project():
    def __init__(self):
        super().__init__()
        self.application_directory = self.get_main_dir()
        self.save_directory = None
        self.name = None
        self.loaded = False
        self.block_types = []
        self.blocks = []
        self.command = CommandController()

        self.load_block_types()

        self.command.add("load", self.load)
        self.command.add("new", self.new)
        self.command.add("save", self.save)
        self.command.add("save_as", self.save_as)
        self.command.add("list_commands", self.list_commands)
        self.command.add("list_block_types", self.list_block_types)
        self.command.add("list_blocks", self.list_blocks)
        self.command.add("add_block", self.add_block)

        self.initialize_project()
        self.list_commands()

    def initialize_project(self):
        options = {'load': self.load, 'new': self.new}
        while True:
            response = prompt("Do you want to load an existing project or create a new one? (load/new): ").lower()
            if response in options:
                options[response]()
                break
            Log.error("Invalid input. Please enter 'load' or 'new'.")

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

                if hasattr(block, 'parts'):
                    Log.info(f"Block {block.name} has {len(block.parts)} parts")
                    for part in block.parts:
                        Log.info(f"{indent_unit * 2}Part: {part.name}")
                        for cmd in part.command.get_commands():
                            Log.info(f"{indent_unit * 3}Command: '{cmd.name}'")
                else:
                    Log.info(f"Block {self.name} has no attributes named 'parts'")

                if hasattr(block, 'port'):
                    Log.info(f"Block {block.name} has {len(block.port.items())} ports")
                    for port in block.port.items():
                        Log.info(f"{indent_unit * 2} Port: {port.name}")
                        for cmd in port.command.get_commands():
                            Log.info(f"{indent_unit * 3}Command: '{cmd.name}'")
                else:
                    Log.info(f"Block {self.name} has no attributes named 'port'")

    def save(self):
        if self.save_directory:
            save_path = os.path.join(self.save_directory, f"{self.name}.json")
            project_data = {
                "save_directory": self.save_directory,
                "name": self.name,
                "blocks": {block.name: block.save.to_dict() for block in self.blocks}
            }
            with open(save_path, 'w') as file:
                json.dump(project_data, file, indent=4)
            Log.info(f"Project saved successfully to {save_path}")
        else:
            if yes_no_prompt("No save directory set. Do you want to set one now? (yes/no): "):
                self.save_directory = prompt("Please enter the path to save the project to: ")
                self.save()
            else:
                Log.error("Save operation cancelled.")

    def save_as(self):
        self.name = prompt("Please enter the name of the project: ")
        self.save_directory = prompt("Please enter the directory to save the project to: ")

        self.save()
        
    def new(self):
        self.name = prompt("Please enter the name of the project: ")      
        Log.special(f"New project creation complete!")

    def load(self):
        load_path = prompt("Please enter the path of the project to load: ")
        if check_project_path(load_path):
            with open(load_path, 'r') as file:
                project_data = json.load(file)
                if not isinstance(project_data, dict):
                    Log.error("Invalid data format. Expected a dictionary.")
                    return
                self.name = project_data.get("name", None)
                self.save_directory = project_data.get("save_directory", None)

                for block_name, block_data in project_data.get("blocks", {}).items():
                    self.add_block(block_name)
                    # Access the ports module
                    
                    # block_modules = block_data.get("module", {})
                    # port_list = block_modules.get("port", {})
                    # port_modules = port_list.get("module", {})
                    # for port_name, port_data in port_modules.items():
                    #     port_module = port_data.get("module", {})
                    #     connections_list = port_module.get("connections", {})
                    #     for connection_name, connection_data in connections_list.items():
                    #         Log.info(f"Connection: {connection_name}")
                    #         connection_module = connection_data.get("module", {})

                    #         input_port = connection_module.get("input_port", {})
                    #         input_port_attribute = input_port.get("attribute", {})
                    #         for attribute_name, attribute_value in input_port_attribute.items():
                    #             self_block = getattr(self.blocks, block_name)
                    #             self_port = getattr(self_block, port_name)
                    #             self_port.create_connection(attribute_value)



                    #         output_port = connection_module.get("output_port", {})
                    #         output_port_attribute = output_port.get("attribute", {})
        else:
            Log.error("Invalid path. Please try again.")

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
        base_path = os.path.join(self.application_directory, "Block/BlockTypes")
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
        Log.info(f"Added block type: {block_type.name}")

    def add_block(self, block_name=None):
        if not block_name:
            block_type = prompt_selection("Please select the block type: ", self.block_types)
            block = block_type()
            block.set_parent(self)

            self.blocks.append(block)
            Log.info(f"Added block: {block_name}")
        else:
            for block_type in self.block_types:
                if block_type.name == block_name:
                    block = block_type()
                    block.set_parent(self)
                    self.blocks.append(block)
                    Log.info(f"Added block: {block_name}")
    
        
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
        for block in self.blocks:
            Log.info(f"Listing current blocks in project {self.name}")
            Log.info(f"{counter}: {block.name}")

            counter += 1