from Model.event_pool import EventPool
from Model.event import Event
from message import Log
from tools import prompt_selection
from .Analyze.analyze import Analyze
from .PreProcess.pre_process import PreProcess
from Command.command_module import CommandModule


class Digest(CommandModule):
    def __init__(self, model, settings):
        super().__init__(model=model)
        self.settings = settings
        self.set_name("Digest")
        self.add_command("results", self.results)
        self.add_command("start", self.start)
        self.preprocess = PreProcess(self.settings)
        self.analyze = Analyze()
        self.add_sub_module(self.preprocess)
        self.add_sub_module(self.analyze)

    def results(self):
        Log.list("Analyzation results", self.results, atrib="data")

    def start(self):
        Log.info("Begin Digest")
        object = self.select_audio_to_digest()
        # Main Pipeline
        # pre_processed_data = self.preprocess.start(object) # run the pre process loop
        self.analyze.start(object) # run the analyze loop, this returns a list of result objects of the analyze process
        # for result_object in result_list:
        #     Log.info(f"Generated result for {result_object.type}")
        #     data = result_object.data 
        #     ep = EventPool() # create a new instance of  
        #     ep.set_name = result_object.type
        #     qty = 0
        #     for frame_number in data:
        #         Log.info(f"adding event for frame number: {frame_number}")
        #         e = Event()
        #         e.set_frame(frame_number)
        #         e.set_name("Default")
        #         e.set_category("Default")
        #         ep.add_event(e)
        #         qty = qty + 1
        #     object.add_event_pool(ep)
        #     Log.info(f"Generated event pool and populated {qty} event objects")

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
