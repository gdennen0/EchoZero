from message import Log
from Point.point import Onset

"""
    Digest pipeline

    Three Distinct Elements

    1.PreProcess

    2.Analyze

    3.Post Process

"""

class PreProcess:
    def __init__(self):
        self.points = []

    def start(self, d):
        Log.info("Initializing pre processing of data")
        if enumerate(self.points) >= 1:
            for index, point in self.points:
                Log.info(f"Applying pre transform point '{point.name}', at index {index}")
                d = point.apply(d)
        return d
    
    def add(self, point):
        Log.info(f"added point '{point.name}'")
        self.points.append(point)

    def remove(self, point):
        Log.info(f"removed point '{point.name}'")
        self.points.remove(point)

    def list(self):
        Log.info('*' * 20 + "PreProcess POINT OBJECTS" + '*' * 20)
        for index, point in enumerate(self.points):
            Log.info(f"Index: {index}, Audio Name: {point.name}")
        Log.info('*' * 64)
class Analyze:
    def __init__(self):
        self.points = []
        self.events = []

    def add(self, point_object):
        Log.info("adding point object into analyze points list")
        self.points.append(point_object)

    def start(self, d):
        Log.info("Initializing analysis of data")
        if enumerate(self.points) >= 1:
            for index, point in self.points:
                Log.info(f"Index: {index}, Audio Name: {point.name}")
                d = point.apply(d)
                self.events.append((point.type, d)) # stores the type and the data list
class Digest:
    def __init__(self):
        self.preprocess = PreProcess()
        self.analyze = Analyze()
        self.point_types = {Onset()}

    def list_point_types(self):
        # Log.list("point types", self.point_types)
        Log.list("Point Types", self.point_types, atrib="type")

    def start(self, data):
        Log.info("Begin Digest")
        pre_processed_data = self.preprocess.start(data)
        event_list = self.analyze.start(pre_processed_data)

        return event_list
