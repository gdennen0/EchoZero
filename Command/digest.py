from message import Log
from tools import prompt
from Point.point import Onset, HighPassFilter

"""
    Digest pipeline

    Three Distinct Elements

    1.PreProcess

    2.Analyze

    3.Post Process

"""

class Digest:
    def __init__(self, model):
        self.model = model
        self.preprocess = PreProcess()
        self.analyze = Analyze()
        self.results = []

    def list_results(self):
        # print(self.results)
        # Log.list("Analyzation results", self.results, atrib="type")
        Log.list("Analyzation results", self.results, atrib="data")

    def start(self, audio_index=None):
        Log.info("Begin Digest")
        if not audio_index:
            audio_index = int(prompt("What is the Index of the audio object you would like to digest? "))
        # Main Pipeline
        audio = self.model.audio.get_audio(audio_index)
        pre_processed_data = self.preprocess.start(audio) # run the pre process loop
        result = self.analyze.start(pre_processed_data) # run the analyze loop
        if len(result) >=1: # if the length of the list is greather than or euqal to 1
            for index, item in enumerate(result):   
                self.results.append(item)

# Generic Result Object 
class Result:
    def __init__(self, data, type):
        self.data = data
        self.type = type

class PreProcess:
    def __init__(self):
        self.points = []
        self.point_types = [
            HighPassFilter(),
        ]

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
                point_object_type_index = int(prompt("What is the index of the Analyze point type you'd like to add? "))
                if point_object_type_index < len(self.point_types):
                    break
                else:
                    Log.error("Invalid index. Please try again.")
            except ValueError:
                Log.error("Invalid input. Please enter a valid integer.")
                
        point_type = self.point_types[point_object_type_index]
        Log.info(f"adding point object {point_type.type} into analyze points list")
        self.points.append(point_type)

    def remove(self, point):
        Log.info(f"removed point '{point.name}'")
        self.points.remove(point)

    def list_point_types(self):
        Log.list("Point Types", self.point_types, atrib="type")

    def list_points(self):
        Log.list("PreProcess Points", self.points, atrib="type")

class Analyze:
    def __init__(self):
        self.points = []
        self.point_types = [
            Onset(),
        ]

    def add(self, point_object_type_index=None):
        while True:
            try:
                point_object_type_index = int(prompt("What is the index of the Analyze point type you'd like to add? "))
                if point_object_type_index < len(self.point_types):
                    break
                else:
                    Log.error("Invalid index. Please try again.")
            except ValueError:
                Log.error("Invalid input. Please enter a valid integer.")

        point_type = self.point_types[point_object_type_index]
        Log.info(f"adding point object {point_type.type} into analyze points list")
        self.points.append(point_type)

    def list_point_types(self):
        Log.list("Point Types", self.point_types, atrib="type")

    def list_points(self):
        Log.list("Point Objects", self.points, atrib="type")

    def start(self, d):
        result_table = []
        Log.info("Initializing analysis of data")
        if len(self.points) >= 1:
            for index, point in enumerate(self.points):
                Log.info(f"Point Index: {index}, Name: {point.name}")
                data = point.apply(d)
                result_object = Result(data, point.type) # Standardizing the result object so it can always be understood properly
                result_table.append(result_object)
            Log.info("Analysis complete")
            return result_table
        else: 
            Log.error("There are no points in the analyze instance")
            result_object = Result(d, "Audio") # Standardizing the result object so it can always be understood properly
            result_table.append(result_object)
            return result_table