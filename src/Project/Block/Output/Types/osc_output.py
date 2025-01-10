from src.Project.Block.Output.output import Output
from src.Utils.message import Log
from src.Utils.tools import prompt

class OSCOutput(Output):
    name = "OSCOutput"

    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "OSCOutput"
        self.type = "OSCOutput"
        self.data_type = "OSCData"

    def push(self):
        # Implement pushing data via OSC if needed
        Log.info(f"Pushing data for OSCOutput '{self.name}'")
        super().push()

    def save(self):
        metadata = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type
        }
        return metadata

    def load(self, metadata):
        self.name = metadata.get("name")
        self.type = metadata.get("type")
        self.data_type = metadata.get("data_type")