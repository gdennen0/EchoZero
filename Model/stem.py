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
        else:
            raise ValueError("Name must be a string")

    def set_type(self, type):
        if isinstance(type, str):
            self.type = type
        else:
            raise ValueError("Type must be a string")

    def set_data(self, data):
        if data is not None:
            self.data = data
        else:
            raise ValueError("Data cannot be None")
