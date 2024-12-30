from src.Utils.tools import prompt_selection, prompt, generate_unique_name
from src.Utils.message import Log
from src.Command.command_controller import CommandController
from src.Project.Block.Output.output import Output


class OutputController:
    def __init__(self, parent_block):
        self.parent_block = parent_block
        self.name =  "OutputController"
        self.type = "OutputController"
        self.outputs = []
        self.output_types = []

        self.command = CommandController()
        self.command.add("list", self.list)

    def set_parent_block(self, parent_block):
        self.parent_block = parent_block
        # Log.info(f"Set output controller parent block to {self.parent_block.name}")

    def add_type(self, input_type):
        if input_type not in self.output_types:
            self.output_types.append(input_type)
        else:
            Log.error(f"Output type {input_type} already exists")

    def add(self, name):
        for output_type in self.output_types:
            if output_type.name == name:
                output_name = generate_unique_name(name, self.outputs)
                new_output = output_type(self.parent_block)
                new_output.name = output_name
                new_output.parent_block = self.parent_block
                self.outputs.append(new_output)
                Log.info(f"Added new output: {output_name} of type: {output_type.name}")
                return
            
        Log.info(f"added new output {name} to block {self.parent_block.name}")

    def list(self):
        Log.info(f"Block {self.parent_block.name} Outputs:")
        if self.outputs:
            for output in self.outputs:
                Log.info(f"Output: {output.name} ({output.type})")
        else:
            Log.info("No inputs found")

    def get_output(self, name):
        for output in self.outputs:
            if output.name == name:
                return output
        return None 
    
    def get(self, name):
        return self.get_output(name)
    
    def get_all(self):
        return self.outputs

    def items(self):
        return self.outputs

    def get_outputs(self):
        return self.outputs

    def set_name(self, name):
        self.name = name
        Log.info(f"Set output controller name to {self.name}")

    def push_all(self, new_data):
        """ this method will likely expand in the future but for now if a data type within the new_data list matches the data type of an output, the data is pushed to the output """
        # Log.info(f"Pushing data to block {self.parent_block.name}'s outputs: {new_data}")
        for output in self.outputs:
            for data_item in new_data:
                if output.data_type == data_item.type:
                    output.data.add(data_item)
                    # Log.info(f"Pushed data to output: {output.name} of type: {output.data_type}")

    def save(self):
        output_data = []
        for output in self.outputs:
            output_data.append(output.save())
        if not output_data:
            return None
        save_data = {
            "name": self.name,
            "outputs": output_data
        }
        return save_data
    
    def load(self, data):
        if data:
            self.name = data.get("name")
            for output_data in data.get("outputs"):
                for output_type in self.output_types:
                    if output_type.name == output_data.get("type"):
                        new_output = output_type(self.parent_block)
                        new_output.load(output_data)
                        self.outputs.append(new_output)