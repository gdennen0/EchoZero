from src.Utils.tools import prompt_selection, prompt, generate_unique_name
from src.Utils.message import Log
from src.Command.command_controller import CommandController
from src.Project.Block.Input.input import Input
from queue import Queue
import json
import os

class InputController:
    def __init__(self, parent_block):
        self.parent_block = parent_block
        self.name =  "InputController"
        self.type = "InputController"
        self.inputs = []
        self.input_types = []

        self.command = CommandController()
        self.command.add("list", self.list)

    def set_parent_block(self, parent_block):
        self.parent_block = parent_block
        # Log.info(f"Set input controller parent block to {self.parent_block.name}")

    def add_type(self, input_type):
        if input_type not in self.input_types:
            self.input_types.append(input_type)
        else:
            Log.error(f"Input type {input_type} already exists")
        # Log.info(f"Added input type {input_type.name}")

    def add(self, name):
        for input_type in self.input_types:
            if input_type.name == name:
                input_name = generate_unique_name(name, self.inputs)

                new_input = input_type(self.parent_block)
                new_input.name = input_name
                new_input.parent_block = self.parent_block
                self.inputs.append(new_input)
                # Log.info(f"Added new input: {input_name} of type: {input_type.name}")
                return
            
    def pull_all(self):
        Log.info(f"***BEGIN PULLING ALL {self.parent_block.name} INPUTS***")
        pulled_data = []
        Log.info(f"Pulling data from {len(self.inputs)} inputs")
        for input in self.inputs:
            Log.info(f"-> Pulling data from input: {input.name}")
            pulled_data.append(input.pull())
        Log.info(f"***END PULLING ALL {self.parent_block.name} INPUTS***")
        return pulled_data

    def list(self):
        Log.info(f"Block {self.parent_block.name} Inputs:")
        if self.inputs:
            for input in self.inputs:
                Log.info(f"Input: {input.name} ({input.type})")
        else:
            Log.info("No inputs found")

    def get(self, name):
        for input in self.inputs:
            if input.name == name:
                return input
        return None

    def get_all(self):
        return self.inputs

    def items(self):
        return self.inputs

    def set_name(self, name):
        self.name = name
        # Log.info(f"Set port controller name to {self.name}")

    def save(self):
        input_data = [input.save() for input in self.inputs]
        metadata = {
            "name": self.name, 
            "inputs": input_data} if input_data else None
        return metadata
    
    def load(self, metadata):
        if metadata is None:
            Log.warning(f"No metadata found for input controller {self.parent_block.name} / not initialized")
            return
        Log.info(f"Loading input controller for block {self.parent_block.name}")
        self.name = metadata.get("name") # get controller name
        for input_data in metadata.get("inputs"):
            input_name = input_data.get("name")
            for input in self.inputs:
                if input.name == input_name:    
                    input.load(input_data)
                    



