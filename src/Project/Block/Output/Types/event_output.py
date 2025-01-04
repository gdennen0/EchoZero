from src.Project.Block.Output.output import Output

class EventOutput(Output):
    name = "EventOutput"
    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "EventOutput" 
        self.type = "EventOutput"
        self.data_type = "EventData"

    def save(self):
        metadata = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type,
        }   
        return metadata
    
    def load(self, metadata):
        self.name = metadata.get("name")
