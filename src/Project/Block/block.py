from src.Command.command_controller import CommandController
from src.Project.Data.data_controller import DataController
from src.Project.Block.Input.input_controller import InputController
from src.Project.Block.Output.output_controller import OutputController

from src.Utils.tools import prompt_selection, prompt_selection_with_type_and_parent_block, prompt_yes_no
from abc import ABC, abstractmethod     
from src.Utils.message import Log
from src.Utils.tools import prompt
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
        self.command.add("list_data", self.list_data_items)
        self.command.add("list_inputs", self.input.list)
        self.command.add("list_outputs", self.output.list)
        self.command.add("reload_all", self.reload_with_dependencies)
        self.command.add("send_command", self.send_command)

    def reload(self, prompt_user=True):
        Log.info(f"Reloading block {self.name}")
        Log.info(f"Prompting user confirmation: {prompt_user}")
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

        if prompt_user == True:
            proceed = prompt_yes_no("Proceed with reloading? This will override the data currently in the block")
            if proceed:
                if results: 
                    self.data.clear()
                    self.output.clear_data()
                    for result in results:
                        if result:
                            self.data.add(result)
                            Log.info(f"Reload result added to data controller: {result.name}")
                        else:
                            Log.error(f"Reload result is None")
                else:
                    Log.error(f"Reload Process Failed because block {self.name} processing didn't return any results.")
                    
            else:
                Log.error(f"Reload Process Failed because user did not confirm")
        else:
            if results:
                self.data.clear()
                self.output.clear_data()
                for result in results:
                    if result:
                        self.data.add(result)
                        Log.info(f"Reload result added to data controller: {result.name}")

        self.output.push_all(self.data.get_all()) # push the results to the output ports
        Log.info(f"***END PROCESSING BLOCK {self.name}***")

    @abstractmethod
    def process(self, input_data):
        """ This method is unique to each subclass of block, ensure that each output dataType gets it own value returned """
        processed_data = []
        processed_data.append(input_data)
        return processed_data

    def set_name(self, name):
        self.name = name
        Log.info(f"Updated Blocks name to: '{name}'")   

    def get_name(self):
        return self.name

    def set_type(self, type):
        self.type = type
        # Log.info(f"Set type: {type}")

    def set_parent(self, parent):
        self.parent = parent
        # Log.info(f"Set parent: {parent.name}")

    def get_commands(self):
        return self.command.get_commands()
        
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

    def list_data_items(self):
        self.data.list_data_items()

    def list_inputs(self):
        self.input.list()

    def list_outputs(self):
        self.output.list()

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
        
    def disconnect_all(self):
        for block in self.parent.blocks:
            if block.name == self.name:
                pass
            else:
                for input in block.input.get_all():
                    if input.connected_output:
                        if input.connected_output.parent_block.name == self.name:
                            input.disconnect()

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
                if input and input.get_connected_output_address() is not None: 
                    Log.info(f"Input: {input.name}, Connection: {input.get_connected_output_address()}")
                elif input is None:
                    Log.error(f"Input is None")
                elif input.get_connected_output_address() is None:
                    Log.error(f"Inputs connected output address is None")
        else:
            Log.error("This block does not have any inputs")

    def reload_with_dependencies(self, prompt_user=False):
        """
        Build a dependency tree and log the order of reload for each block.

        This method iterates through all blocks to find dependencies and logs
        the order in which blocks would be reloaded based on their dependencies.
        
        Args:
            prompt_user (bool): Whether to prompt the user before reloading. Default is True.
        """
        dependency_tree = {}
        execution_order = []
        in_degree = {}  # Track number of dependencies for each block

        # Initialize the dependency tree and in-degree count
        for block in self.parent.blocks:
            dependency_tree[block.name] = []
            in_degree[block.name] = 0
            
        # Build the dependency tree
        for block in self.parent.blocks:
            for input_port in block.input.get_all():
                if input_port.connected_output:
                    source_block = input_port.connected_output.parent_block.name
                    target_block = block.name
                    dependency_tree[source_block].append(target_block)
                    in_degree[target_block] += 1

        # Find blocks that depend on current block (directly or indirectly)
        dependent_blocks = set()
        queue = [self.name]
        while queue:
            current = queue.pop(0)
            dependent_blocks.add(current)
            queue.extend([block for block in dependency_tree[current] 
                         if block not in dependent_blocks])

        # Reset execution tracking for relevant blocks only
        execution_order = []
        queue = []
        filtered_in_degree = {name: 0 for name in dependent_blocks}

        # Recalculate in_degree for relevant blocks
        for source in dependent_blocks:
            for target in dependency_tree[source]:
                if target in dependent_blocks:
                    filtered_in_degree[target] += 1

        # Find starting blocks among dependents
        for block_name in dependent_blocks:
            if filtered_in_degree[block_name] == 0:
                queue.append(block_name)

        # Process the queue to build execution order
        while queue:
            current_block = queue.pop(0)
            execution_order.append(current_block)

            for dependent_block in dependency_tree[current_block]:
                if dependent_block in dependent_blocks:
                    filtered_in_degree[dependent_block] -= 1
                    if filtered_in_degree[dependent_block] == 0:
                        queue.append(dependent_block)

        # Check for circular dependencies in the subset
        if len(execution_order) != len(dependent_blocks):
            Log.error("Circular dependency detected in block connections!")
            return

        # Log the execution order
        Log.info(f"Block reload order for {self.name} and its dependencies:")
        for i, block_name in enumerate(execution_order, 1):
            Log.info(f"{i}. {block_name}")

        proceed = True
        if prompt_user == True:
            proceed = prompt_yes_no("Proceed with reloading? This will override the data currently in all dependent blocks")
        
        if proceed:
            for block_name in execution_order:
                for block in self.parent.blocks:
                    if block.name == block_name:
                        # Log.info(f"Reloading block: {block_name}")
                        block.reload(prompt_user=False)
                        break
        else:
            Log.error("Reload process cancelled")

    def get_metadata_from_dir(self, dir):
        block_metadata_path = os.path.join(dir, 'metadata.json')
        with open(block_metadata_path, 'r') as block_metadata_file:
            block_metadata = json.load(block_metadata_file)
        return block_metadata

    def send_command(self, command_string=None):
        """
        Sends a command to the parser through the parent project.
        
        This allows blocks to execute commands outside their own scope by
        sending them through the main command parser.
        
        Args:
            command_string (str): The command string to parse and execute
            
        Returns:
            bool: True if command executed successfully, False otherwise
        """
        if command_string:
            result = self.parent.send_command(command_string)
            return result
        else:
            string = prompt("Enter command to send: ")
            result = self.parent.send_command(string)
            return result
