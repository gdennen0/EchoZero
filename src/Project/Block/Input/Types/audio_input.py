from src.Project.Block.Input.input import Input
from src.Utils.message import Log
class AudioInput(Input):
    name = "AudioInput"
    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "AudioInput" 
        self.type = "AudioInput"
        self.data_type = "AudioData"

    def save(self):
        # Create base save data
        metadata = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type,
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
        connected_output = metadata.get("connected_output")
        if connected_output:
            connected_block_name = connected_output.split('.')[0]
            connected_block = self.parent_block.parent.get_block(connected_block_name)
            output_name = metadata.get("connected_output").split('.')[2]
            connected_output = connected_block.output.get(output_name)
            self.connect(connected_output)
        else:
            Log.error(f"Block '{self.parent_block.name}' is not connected to any outputs")

