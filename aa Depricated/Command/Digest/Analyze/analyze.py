from Command.Digest.PointTypes.onset import Onset
from Command.Digest.PointTypes.percussion_feature_extractor import PercussionFeatureExtractor
from Command.Digest.PointTypes.exctract_percussion_events import ExtractPercussionEvents
from Command.command_module import CommandModule
from message import Log
from tools import prompt, prompt_selection

class Analyze(CommandModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.points = []
        self.point_types = [
            Onset(),
            PercussionFeatureExtractor(),
            ExtractPercussionEvents(),
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

    def start(self): 
        """
        Applys each points analysis/transformation to the audio object
        
        """
        audio_object = self.select_audio_to_digest()
        Log.info("[Digest][Analyze] Analysis point Begin")
        if len(self.points) >= 1:
            for index, point in enumerate(self.points):
                Log.info(f"[Digest][Analyze] Applying Point: {point.name} to {audio_object.name}")
                point.apply(audio_object)
            Log.info("[Digest][Analyze] Analysis points completed")
        else: 
            Log.error("There are no points in the analyze instance")


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