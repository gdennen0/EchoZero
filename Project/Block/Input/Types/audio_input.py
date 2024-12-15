from Project.Block.Input.input import Input
from Utils.message import Log
class AudioInput(Input):
    name = "AudioInput"
    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "AudioInput" 
        self.type = "AudioInput"
        self.data_type = "AudioData"

    def save(self):
        # Create base save data
        save_data = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type,
            "data": self.data.save()
        }
        
        # Add connected output info only if it exists
        if self.connected_output:
            try:
                save_data["connected_output"] = f"{self.connected_output.parent_block.name}.output.{self.connected_output.name}"
            except AttributeError:
                save_data["connected_output"] = None
        else:
            save_data["connected_output"] = None
            
        return save_data
    
    def load(self, data):
        self.name = data.get("name")
        connected_output = data.get("connected_output")
        if connected_output:
            connected_block_name = connected_output.split('.')[0]
            connected_block = self.parent_block.parent.get_block(connected_block_name)
            output_name = data.get("connected_output").split('.')[2]
            connected_output = connected_block.output.get(output_name)
            self.connect(connected_output)
        else:
            Log.error(f"Block '{self.name}' is not connected to any output")

