from src.Project.Block.Output.output import Output

class AudioOutput(Output):
    name = "AudioOutput"
    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "AudioOutput" 
        self.type = "AudioOutput"
        self.data_type = "AudioData"

    def save(self):
        # Create base save data
        metadata = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type,
        }       
        return metadata
    
    def load(self, metadata):
        self.name = metadata.get("name")