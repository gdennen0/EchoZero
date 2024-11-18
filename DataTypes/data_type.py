from message import Log

class DataType:
    def __init__(self):
        Log.info(f"Creating Instance of the DataType Object")

        self.name = None
        self.description = None
        self.data = None

    def set_data(self, data):
        self.data = data
        Log.info(f"data set")

    def set_name(self, name):
        self.name = name
        Log.info(f"Set DataType name attribite to: '{name}'")

    def set_description(self, description):
        self.description = description
        Log.info(f"Set description: {description}")
