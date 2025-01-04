from src.Project.Block.Input.input import Input
from src.Utils.message import Log
from src.Utils.tools import prompt
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
        metadata = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type,
            "connected_output": self.connected_output.name if self.connected_output else None
        }
        return metadata

    def load(self, metadata):
        self.name = metadata.get("name")
        self.type = metadata.get("type")
        self.data_type = metadata.get("data_type")
        connected_output = metadata.get("connected_output")
        if connected_output:
            connected_block_name = connected_output.split('.')[0]
            connected_block = self.parent_block.parent.get_block(connected_block_name)
            output_name = metadata.get("connected_output").split('.')[2]
            connected_output = connected_block.output.get(output_name)
            self.connect(connected_output)