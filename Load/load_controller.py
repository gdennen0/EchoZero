import json
from message import Log

class LoadController:
    def __init__(self, parent):
        self.parent = parent

    def from_dict(self, data):
        """Load the attributes from a dictionary."""
        if not isinstance(data, dict):
            Log.error("Invalid data format. Expected a dictionary.")
            return

        for block in data.get("blocks", []):
            self.parent.add_block(block)