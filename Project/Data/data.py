from Utils.message import Log

"""
Generic structure to hold data
"""
class Data:
    def __init__(self):
        self.name = None
        self.type = None
        self.description = None

        self.data = None

    def set_data(self, data):
        self.data = data
        Log.info(f"Data Object '{self.name}' updated")

    def set_type(self, type):
        self.type = type

    def set_name(self, name):
        self.name = name
        Log.info(f"Set DataType name attribute to: '{name}'")

    def set_description(self, description):
        self.description = description
        Log.info(f"Set description: {description}")
