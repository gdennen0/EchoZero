from message import Log
"""
    Parent class to build stem subclasses off of
    methods to set attributes and validate input

"""

class stem:
    def __init__(self):
        self.name = None
        self.type = None
        self.data = None
        self.tensor = None

    def set_name(self, name):
        if isinstance(name, str):
            self.name = name
            Log.info(f"Set stem name {name}")
        else:
            Log.error("Name must be a string")

    def set_type(self, type):
        if isinstance(type, str):
            self.type = type
            Log.info(f"Set type to {type}")
        else:
            Log.error("Type must be a string")

    def set_data(self, data):
        if data is not None:
            self.data = data
            Log.info(f"set data to data object")
        else:
            Log.error("Data cannot be None")
        
    def set_tensor(self, tensor):
        if tensor is not None:
            self.tensor = tensor
            Log.info(f"Set tensor to tensor object")
        else:
            Log.error("tensor cannot be None")
        
