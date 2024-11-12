class OutputPort:
    def __init__(self, name, data):
        self.type = "output"
        self.name = name
        self.data = data

class InputPort:
    def __init__(self, name, data):
        self.type = "input"
        self.name = name
        self.data = data


