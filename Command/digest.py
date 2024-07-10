from Model.audio import event_pool, event
from message import Log
from tools import prompt, prompt_selection
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
        Log.list("Analyzation results", self.results, atrib="data")

    def start(self):
        Log.info("Begin Digest")
        object = self.select_audio_to_digest()
        # Main Pipeline
        pre_processed_data = self.preprocess.start(object.audio) # run the pre process loop
        result_list = self.analyze.start(pre_processed_data) # run the analyze loop, this returns a list of result objects of the analyze process
        for result_object in result_list:
            Log.info(f"Generated result for {result_object.type}")
            data = result_object.data 
            ep = event_pool() # create a new instance of  
            ep.set_name = result_object.type
            qty = 0
            for frame_number in data:
                Log.info(f"adding event for frame number: {frame_number}")
                e = event()
                e.set_frame(frame_number)
                e.set_name("Default")
                e.set_category("Default")
                ep.add_event(e)
                qty = qty + 1
            object.add_event_pool(ep)
            Log.info(f"Generated event pool and populated {qty} event objects")

    def select_audio_to_digest(self):
        audio_selections = []
        a, _  = prompt_selection("Select an audio object to operate on: ", self.model.audio.objects)
        audio_selections.append("Self")
        if a.stems:
            for stem in a.stems:
                audio_selections.append(stem.name)
        sel_obj, selection = prompt_selection("Select audio to analyze", audio_selections)
        if isinstance(selection, int):
            if selection == 0:
                Log.info(f"Selected original audio from audio object {a.name}")
                return a
            elif selection > 0:
                s = a.stems[selection - 1]  # Corrected indexing
                Log.info(f"Selected stem {s.name} from audio object {a.name}")
                return s
            
        elif isinstance(selection, str):
            if selection == "Self":
                Log.info(f"Selected original audio from audio object {a.name}")
                return a
            else:
                for stem in a.stems:
                    if stem.name == selection:
                        Log.info(f"Selected stem {stem.name} from audio object {a.name}")
                        return stem
        
        # d_selection, _  = prompt_selection("Select audio to analyze for audio object {}")

        Log.error("Invalid selection")
        return None


# Generic Result Object 
class Result:
    def __init__(self, data, type):
        self.data = data
        self.type = type

class PreProcess:
    def __init__(self):
        self.points = []
        self.point_types = [    #list of filter objects that are initialized
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

class Analyze:
    def __init__(self):
        self.points = []
        self.point_types = [
            Onset(),
        ]

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
        self.list_point_types()
        Log.info(f"Added point object: '{point_type.type}' into Analyze")


    def list_point_types(self):
        Log.list("Available point types: ", self.point_types, atrib="type")

    def list_points(self):
        Log.list("Active point objects: ", self.points, atrib="type")

    def start(self, d): # pass start the audio data
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