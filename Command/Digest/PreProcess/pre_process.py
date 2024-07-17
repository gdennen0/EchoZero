from PointTypes.high_pass_filter import HighPassFilter
from message import Log

class PreProcess:
    def __init__(self):
        self.points = []
        self.point_types = [    #list of filter objects that are initialized
            HighPassFilter(),
        ]

    def get_commands(self):
        return {
            "start": self.start,
            "add": self.add,
            "remove": self.remove,
        }

    def start(self, data):
        Log.info("Initializing pre processing of data")
        if len(self.points) >= 1:
            for index, point in enumerate(self.points):
                Log.info(f"Applying pre transform point '{point.name}', at index {index}")
                data = point.apply(data)
            Log.info("Completed PreProcess") 
            return data
        else:
            Log.info("No PreProcess points found, bypassing preprocess")
            return data
        
    def add(self, point_object_type_index=None):
        while True:
            try:
                self.list_point_types()
                point_object_type_index = int(prompt("What is the index of the Analyze point type you'd like to add? "))
                if point_object_type_index < len(self.point_types):
                    break
                else:
                    Log.error("Invalid index. Please try again.")
            except ValueError:
                Log.error("Invalid input. Please enter a valid integer.")
                
        point_type = self.point_types[point_object_type_index]
        self.points.append(point_type)
        self.list_points()
        Log.info(f"Added point object: '{point_type.type}' into PreProcess")


    def remove(self, point_object_type_index=None):
        while True:
            try:
                self.list_point_types()
                point_object_type_index = int(prompt("What is the index of the Analyze point type you'd like to add? "))
                if point_object_type_index < len(self.point_types):
                    break
                else:
                    Log.error("Invalid index. Please try again.")
            except ValueError:
                Log.error("Invalid input. Please enter a valid integer.")
            
        point_type = self.point_types[point_object_type_index]
        self.points.remove(point_type)
        self.list_points()
        Log.info(f"Removed point: '{point_type.type}' from PreProcess")

    def list_point_types(self):
        Log.list("Point Types", self.point_types, atrib="type")

    def list_points(self):
        Log.list("PreProcess Points", self.points, atrib="type")
