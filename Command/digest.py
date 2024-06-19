from message import Log
"""
    Digest pipeline

    Three Distinct Elements

    1.PreProcess

    2.Analyze

    3.Post Process

"""

class Digest:
    def __init__(self):
        self.points = []

    def start(self, data):
        x = data
        if enumerate(self.points) >= 1:
            for index, point in self.points:
                Log.info
                step = point.process(x)
                x = step

    def add(self, point):
        Log.info(f"added point '{point.name}'")
        self.points.append(point)

    def remove(self, point):
        Log.info(f"removed point '{point.name}'")
        self.points.remove(point)

    def list_points(self):
        Log.info('*' * 20 + "POINT OBJECTS" + '*' * 20)
        for index, point in enumerate(self.points):
            Log.info(f"Index: {index}, Audio Name: {point.name}")
        Log.info('*' * 53)

class Point:
    def __init__(self):
        self.name = None
        self.type = None
        self.description = None

    def process(self, data):
        Log.info(f"point transform | applying transformation {self.name}")
        return data
    