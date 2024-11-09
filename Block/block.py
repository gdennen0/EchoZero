from command_module import CommandModule
from message import Log
from abc import ABC, abstractmethod
from Block.part import Part  # Assuming Part is defined in Block/part.py
from tools import prompt_selection

class Block(CommandModule, ABC):
    def __init__(self):
        super().__init__()
        self.name = None
        self.type = None
        self.parts = []
        self.part_types = []
        self.input_types = []
        self.output_types = []
        self.data = None

        self.add_command("add_part", self.add_part)
        self.add_command("remove_part", self.remove_part)
        self.add_command("list_parts", self.list_parts)

    def start(self, input_data):
        Log.info(f"Starting block {self.name}")
        if self._validate_input(input_data):
            result = self._process_parts(input_data)
            if self._validate_output(result):
                return result
            else:
                Log.error(f"Result {result} not in output types {self.output_types}")
                return None
        else:
            Log.error(f"Input data {input_data} is not a valid input type.")
            return None

    def _validate_input(self, input_data):
        if input_data in self.input_types:
            return True
        else:
            Log.error(f"Input data {input_data} is not a valid input type.")
            return False

    def _process_parts(self, input_data):
        result = input_data
        for part in self.parts:
            Log.info(f"Processing with part {part.name}")
            result = part.start(result)
            if result is None:
                Log.error(f"Part {part.name} failed to process the data.")
                return None
        return result

    def _validate_output(self, result):
        return result in self.output_types

    def set_name(self, name):
        self.name = name
        Log.info(f"Set name: {name}")   

    def set_type(self, type):
        self.type = type
        Log.info(f"Set type: {type}")

    def add_part(self, part_name=None):
            if part_name: # if part name is specified, add the part type with that name string
                part_type = str(part_name)
                for part_type in self.part_types: # iterate through part types
                    Log.info(f"Checking part type name '{part_type.name}' against '{part_name}'")
                    if part_type.name == part_name: 
                        self.parts.append(part_type)
                        Log.info(f"Added part: {part_name}")
                        return      
                Log.error(f"Part type {part_name} not found in part types list.")
            else:
                part_type, _ = prompt_selection("Please select a part type to add: ", self.part_types)
                self.parts.append(part_type)
                Log.info(f"Added part: {part_type.name}")

    def remove_part(self, part_name=None):
        if part_name:
            for part in self.parts:
                if part.name == part_name:
                    self.parts.remove(part)
                    Log.info(f"Removed part: {part_name}")
                    return
            Log.error(f"Part {part_name} not found in parts list.")
        else:
            part_type, _ = prompt_selection("Please select a part to remove: ", self.parts)
            self.parts.remove(part_type)
            Log.info(f"Removed part: {part_type.name}")

    def list_parts(self):
        Log.info("Listing parts")
        for part in self.parts:
            Log.info(f"- {part.name} ({part.type})")
        return self.parts

    def clear_parts(self):
        self.parts = []
        Log.info("Cleared all parts")

    def add_part_type(self, part_type):
        self.part_types.append(part_type)
        Log.info(f"Added part type: {part_type}")

    def remove_part_type(self, part_type):
        self.part_types.remove(part_type)
        Log.info(f"Removed part type: {part_type}")

    def list_part_types(self):
        Log.info("Listing part types")
        for pt in self.part_types:
            Log.info(f"- {pt}")
        return self.part_types

    def add_input_type(self, input_type):
        self.input_types.append(input_type)
        Log.info(f"Added input type: {input_type}")

    def remove_input_type(self, input_type):
        if input_type in self.input_types:
            self.input_types.remove(input_type)
            Log.info(f"Removed input type: {input_type}")
        else:
            Log.error(f"Input type {input_type} not found in input types list.")

    def add_output_type(self, output_type):
        self.output_types.append(output_type)
        Log.info(f"Added output type: {output_type}")

    def remove_output_type(self, output_type):
        if output_type in self.output_types:
            self.output_types.remove(output_type)
            Log.info(f"Removed output type: {output_type}")
        else:
            Log.error(f"Output type {output_type} not found in output types list.")

    def list_output_types(self):
        Log.info("Listing output types")
        for ot in self.output_types:
            Log.info(f"- {ot}")
        return self.output_types

    def list_input_types(self):
        Log.info("Listing input types")
        for it in self.input_types:
            Log.info(f"- {it}")
        return self.input_types
    
    def set_data(self, data):
        self.data = data
        Log.info(f"set data:")