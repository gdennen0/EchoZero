from src.Command.command_controller import CommandController
from src.Project.Data.data_controller import DataController
from src.Project.Block.Input.input_controller import InputController
from src.Project.Block.Output.output_controller import OutputController

from src.Utils.tools import prompt_selection, prompt_selection_with_type_and_parent_block
from abc import ABC, abstractmethod     
from src.Utils.message import Log
from src.Utils.tools import gtimer
import os
import json

class Block(ABC):
    def __init__(self):
        self.name = None
        self.type = None
        self.parent = None
        self.data = DataController(self)        
        self.input = InputController(self)
        self.output = OutputController(self)
        self.command = CommandController()
        self.command.add("reload", self.reload)
        self.command.add("connect", self.connect)
        self.command.add("disconnect", self.disconnect)
        self.command.add("list_connections", self.list_connections)
        self.command.add("list_commands", self.command.list_commands)

    def reload(self):
        Log.info(f"Reloading block {self.name}")
        timer = gtimer()
        timer.start()
        self.input.pull_all() # Pull data from connected external output ports to local input ports
        
        input_data = []
        if len(self.input.get_all()) > 0:
            for input in self.input.items():
                for data_item in input.data.get_all():
                    input_data.append(data_item) # add each collected data item from the inputs to the input_data list
        else:
            Log.warning(f"There are no inputs in {self.name}")


        results = self.process(input_data) # process the input data

        if results:
            self.data.clear()
            for result in results:
                if result:
                    self.data.add(result)
                    Log.info(f"Reload result added to data controller: {result.name}")
                else:
                    Log.error(f"Reload result is None")
        else:
            Log.error(f"Reload Process Failed because block {self.name} processing didn't return any results.")

        self.output.push_all(self.data.get_all()) # push the results to the output ports

    @abstractmethod
    def process(self, input_data):
        """ This method is unique to each subclass of block, ensure that each output dataType gets it own value returned """
        processed_data = []
        processed_data.append(input_data)
        return processed_data

    def set_name(self, name):
        self.name = name
        Log.info(f"Updated Blocks name to: '{name}'")   

    def set_type(self, type):
        self.type = type
        Log.info(f"Set type: {type}")

    def set_parent(self, parent):
        self.parent = parent
        Log.info(f"Set parent: {parent.name}")
        
    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "metadata": self.data.get_metadata(),
            "input": self.input.save(),
            "output": self.output.save()
        }

    def save(self, save_dir):
        self.data.save(save_dir)    

    def connect(self):
        if self.input.get_all():
            Log.info(f"Select which {self.name} input you would like to connect")
            input = prompt_selection("Select input port: ", self.input.get_all())
            Log.info(f"Block {self.name} input port: {input.name} selected")
        else:
            Log.error("This block does not have any inputs")
            return

        if self._get_external_outputs():
            Log.info(f"Select which output port to connect {input.name} to")
            external_output = prompt_selection_with_type_and_parent_block(f"Select output port to connect {input.name} to: ", self._get_external_outputs() )
        else:
            Log.error("There are no output external ports")
            return
        
        if input and external_output:
            input.connect(external_output)

    def disconnect(self):
        if self.input.get_all():
            Log.info(f"Select which {self.name} input you would like to disconnect from an external output")
            input = prompt_selection("Select input port: ", self.input.get_all())
            input.disconnect()
        else:
            Log.error("This block does not have any inputs")
            return


    def _get_external_outputs(self):
        # get all the external outputs from the blocks
        external_outputs = []
        for block in self.parent.blocks:
            if block.name == self.name:
                pass
            else:
                for external_output in block.output.get_all():
                    external_outputs.append(external_output)
        return external_outputs
    
    def list_connections(self):
        if self.input.get_all():
            Log.info(f"Listing connections for block '{self.name}'")
            for input in self.input.get_all():
                if input: 
                    Log.info(f"Input: {input.name}, Connection: {input.get_connected_output_address().name}")

        else:
            Log.error("This block does not have any inputs")

        if self.output.get_all():
            Log.info(f"Listing connections for block '{self.name}'")
            for output in self.output.get_all():
                Log.info(f"Output: {output.name}, Connection: {output.get_connected_input_address().name}")
        else:
            Log.error("This block does not have any outputs")


    def get_metadata_from_dir(self, dir):
        block_metadata_path = os.path.join(dir, 'metadata.json')
        with open(block_metadata_path, 'r') as block_metadata_file:
            block_metadata = json.load(block_metadata_file)
        return block_metadata
