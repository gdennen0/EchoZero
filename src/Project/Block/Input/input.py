from src.Utils.message import Log
from src.Utils.tools import prompt, prompt_selection, prompt_selection_with_type_and_parent_block
from src.Command.command_controller import CommandController
from src.Project.Data.data_controller import DataController
from src.Project.Block.Output.output import Output

class Input:
    """
    - Input is responsible for pulling data from an output port, and storing it in its data controller
    - Can only have one connected output port
    - Can only have one data type
    - only accepts data of the correct data type
    """
    def __init__(self, parent_block):
        self.name = None
        self.type = "Input"
        self.parent_block = parent_block
        self.data_types = []
        self.data_type = None
        self.connected_output = None # can only have one connected output

        self.data = DataController(self) # initialize a data controller
        self.data.set_parent(self)

        self.command = CommandController() # initialize a command controller
        self.command.add("set_name", self.set_name)
        
    def set_parent_block(self, parent_block):
        self.parent_block = parent_block
        Log.info(f"Input {self.name} parent block set to: {parent_block.name}")

    def set_name(self, name=None):
        if name:
            self.name = name
            Log.info(f"Port name set to: {name}")
        else:
            self.name = prompt(f"Please enter the name of the port: ")
            Log.info(f"Port name set to: {self.name}")

    def add_data_type(self, data_type):
        if data_type not in self.data_types:
            self.data_types.append(data_type)
            # Log.info(f"Added data type {data_type} to input {self.name}")
        else:
            Log.error(f"Data type {data_type} already exists in input {self.name}")

    def set_data_type(self, data_type):
        if data_type in self.data_types:
            self.data_type = data_type
            Log.info(f"Port data type set to: {data_type}")
        else:
            Log.error(f"Invalid data type: {data_type}. Valid data types: {self.data_types}")

    def get_connected_output_address(self):
        if self.connected_output:
            return f"{self.connected_output.parent_block.name}.{self.connected_output.type}.{self.connected_output.name}"
        else:
            Log.error(f"No connected output found for {self.parent_block.name}.{self.type}.{self.name}")
            return None

    def pull(self):
        """
        - Pulls data from the connected output port, and stores it in the input data controller
        - Only pulls data of the correct data type
        """
        if self.connected_output:
            self.data.clear() #clear the input data before adding new data
            connected_output_data = self.connected_output.data.get_all()
            if connected_output_data:
                for data_item in connected_output_data:
                    if data_item.type == self.data_type: # Ensure the pulled data is of the correct type
                        self.data.add(data_item)
                        Log.info(f"Pulled data from {self.connected_output.parent_block.name}.{self.connected_output.type}.{self.connected_output.name} to {self.parent_block.name}.{self.type}.{self.name}")
                    else:
                        Log.error(f"Pull data type mismatch. {self.parent_block.name} Input {self.name} expected {self.data_type}, but got {data_item.type}")
            else:
                Log.error(f"No data found in {self.connected_output.parent_block.name}.{self.connected_output.type}.{self.connected_output.name}")

    def connect(self, connected_output):
        if isinstance(connected_output, Output):
            self.connected_output = connected_output
            Log.info(f"Connected '{self.parent_block.name}' {self.type} port '{self.name}' <--> '{connected_output.parent_block.name}' {connected_output.type} port '{connected_output.name}'")
        else:
            Log.error(f"Error connecting Block {self.parent_block.name} port {self.name}. Connection must be an output port, got {type(connected_output)}")
            

    def disconnect(self):
        Log.info(f"Disconnecting '{self.parent_block.name}' {self.type} from '{self.connected_output.parent_block.name}' {self.connected_output.type}")
        self.connected_output = None

    def list_connection(self):
        Log.info(f"Connected output port: {self.connected_output.name}")
            
    def save(self):
        # Create base save data
        metadata = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type
        }
        
        # Add connected output info only if it exists
        if self.connected_output:
            try:
                metadata["connected_output"] = f"{self.connected_output.parent_block.name}.output.{self.connected_output.name}"
            except AttributeError:
                metadata["connected_output"] = None
        else:
            metadata["connected_output"] = None
            
        return metadata
    
    def load(self, metadata):
        self.name = metadata.get("name")
        self.type = metadata.get("type")
        self.data_type = metadata.get("data_type")
    
        connected_output = metadata.get("connected_output")
        if connected_output:
            connected_block_name = connected_output.split('.')[0]
            connected_block = self.parent_block.parent.get_block(connected_block_name)
            output_name = connected_output.split('.')[2]
            connected_output = connected_block.output.get(output_name)
            self.connect(connected_output)



