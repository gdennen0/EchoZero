from src.Utils.tools import prompt_selection, prompt, generate_unique_name
from src.Utils.message import Log
from src.Command.command_controller import CommandController
from src.Project.Block.Input.input import Input
from queue import Queue

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
        Log.info(f"Set input controller parent block to {self.parent_block.name}")

    def add_type(self, input_type):
        if input_type not in self.input_types:
            self.input_types.append(input_type)
        else:
            Log.error(f"Input type {input_type} already exists")
        Log.info(f"Added input type {input_type.name}")

    def add(self, name):
        for input_type in self.input_types:
            if input_type.name == name:
                input_name = generate_unique_name(name, self.inputs)

                new_input = input_type(self.parent_block)
                new_input.name = input_name
                new_input.parent_block = self.parent_block
                self.inputs.append(new_input)
                Log.info(f"Added new input: {input_name} of type: {input_type.name}")
                return
            
    def pull_all(self):
        Log.info(f"Pulling all {self.parent_block.name} inputs")
        pulled_data = []
        for input in self.inputs:
            output_data = input.pull()
            pulled_data.append(output_data)
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
        Log.info(f"Set port controller name to {self.name}")

    def save(self):
        input_data = [input.save() for input in self.inputs]
        return {
            "name": self.name, 
            "inputs": input_data} if input_data else None
    
    def load(self, data):
        Log.info(f"Loading input controller for block {self.parent_block.name}")
        self.name = data.get("name") # get controller name
        for input_data in data.get("inputs"):
            input_name = input_data.get("name")
            for input in self.inputs:
                if input.name == input_name:    
                    Log.info(f"Matched local input {input.name} with saved input {input_name}")
                    Log.info(f'input name: {input.name}')
                    Log.info(f'input type: {input.type}')
                    Log.info(f'input data type: {input.data_type}')
                    Log.info(f'input connected output: {input.connected_output}')
                    input.load(input_data)
                    



