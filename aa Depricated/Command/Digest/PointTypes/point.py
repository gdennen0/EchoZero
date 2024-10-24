from message import Log

# POINT CLASS
# The parent class for all point sub-objects, applicable broadly
class Point:
    def __init__(self):
        self.name = None
        self.type = None
        self.description = None

    def apply(self, data):
        Log.info(f"Point transform | Applying transformation: {self.name}")
        return data
    


        