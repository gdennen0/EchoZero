from src.Utils.message import Log

"""
Generic structure to hold data
"""
class Data:
    """
    Generic structure to hold data
    """
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

    def get_data_type(self, name):
        for data_type in self.data_types:
            if data_type.name == name:
                return data_type
        return None
