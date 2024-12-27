from Command.command_controller import CommandController
from Project.Data.data_controller import DataController
from Project.Block.Input.input_controller import InputController
from Project.Block.Output.output_controller import OutputController

from Utils.tools import prompt_selection, prompt_selection_with_type_and_parent_block
from abc import ABC, abstractmethod
from Utils.message import Log
from Utils.tools import gtimer

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

    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "data": self.data.save(),
            "input": self.input.save(),
            "output": self.output.save()
        }
    
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