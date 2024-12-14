from Project.Block.Output.output import Output
from Utils.message import Log
from Utils.tools import prompt

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
        return {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type
        }

    def load(self, data):
        self.name = data.get("name")
        self.type = data.get("type")
        self.data_type = data.get("data_type")