from Project.Block.Input.input import Input
from Utils.message import Log
from Utils.tools import prompt

class OSCInput(Input):
    name = "OSCInput"

    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "OSCInput"
        self.type = "OSCInput"
        self.data_type = "OSCData"

    def pull(self):
        # Implement pulling data via OSC if needed
        Log.info(f"Pulling data for OSCInput '{self.name}'")
        return super().pull()

    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type,
            "connected_output": self.connected_output.name if self.connected_output else None
        }

    def load(self, data):
        self.name = data.get("name")
        self.type = data.get("type")
        self.data_type = data.get("data_type")
        connected_output = data.get("connected_output")
        if connected_output:
            # Implement logic to reconnect to the external output port
            pass