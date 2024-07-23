from Command.Digest.PointTypes.onset import Onset
from Command.Digest.PointTypes.percussion_feature_extractor import PercussionFeatureExtractor
from Command.command_module import CommandModule
from message import Log
from tools import prompt

class Analyze(CommandModule):
    def __init__(self):
        super().__init__()
        self.points = []
        self.point_types = [
            Onset(),
            PercussionFeatureExtractor(),
        ]
        self.set_name("Analyze")
        self.add_command("start", self.start)
        self.add_command("add", self.add)
        self.add_command("list_point_types", self.list_point_types)
        self.add_command("list_points", self.list_points)

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

        point = self.point_types[point_object_type_index]
        self.points.append(point)
        self.list_point_types()
        Log.info(f"Added point object: '{point.type}' into Analyze")


    def list_point_types(self):
        Log.list("Available point types: ", self.point_types, atrib="type")

    def list_points(self):
        Log.list("Active point objects: ", self.points, atrib="type")

    def start(self, audio_object): # pass start the audio data
        result_table = []
        Log.info("Initializing analysis of data")
        if len(self.points) >= 1:
            for index, point in enumerate(self.points):
                Log.info(f"Point Index: {index}, Name: {point.name}")
                point.apply(audio_object)
            Log.info("Analysis complete")
            return result_table
        else: 
            Log.error("There are no points in the analyze instance")
            # result_object = Result(d, "Audio") # Standardizing the result object so it can always be understood properly
            # result_table.append(result_object)
            return result_table
        
class Result:
    def __init__(self, data, type):
        self.data = data
        self.type = type