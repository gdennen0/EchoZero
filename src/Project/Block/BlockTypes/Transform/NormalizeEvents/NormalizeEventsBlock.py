from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput

from src.Utils.message import Log
from lib.audio_separator.separator.separator import Separator
from src.Utils.tools import prompt_selection, prompt
import os
from src.Utils.message import Log
import json

# Separates audio file into stems
class NormalizeEventsBlock(Block):
    name = "NormalizeEvents"
    type = "NormalizeEvents"
    
    def __init__(self):
        super().__init__()
        self.name = "NormalizeEvents"
        self.type = "NormalizeEvents"

        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")
        
        self.command.add("set_bpm", self.set_bpm)
        self.command.add("set_normalization_threshold", self.set_normalization_threshold)
        self.command.add("batch_normalize", self.batch_normalize)

        self.bpm = None
        self.normalization_threshold = None

        Log.info(f"QuantizeEvents initialized")

    def batch_normalize(self):
        """
        Normalizes the events to the BPM and normalization threshold
        """
        Log.info("Starting batch normalization")
        Log.debug(f"BPM set to: {self.bpm}, Normalization Threshold set to: {self.normalization_threshold}")

        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                event_item.time = self.__normalize_event(event_item.time)
                Log.info(f"Normalized event {event_item.name}: {event_item.time}")

    def set_normalization_threshold(self):
        """
        Sets the normalization threshold variable for the events
        """
        self.normalization_dict = {"1": 1, "1/2": 2, "1/4": 4, "1/8": 8, "1/16": 16, "1/32": 32, "1/64": 64}
        self.normalization_threshold = prompt_selection("Enter the normalization threshold: ", 
                                                       ["1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64"])
        self.normalization_threshold = self.normalization_dict[self.normalization_threshold]
        Log.info(f"Normalization threshold set to {self.normalization_threshold}")


    def set_bpm(self):
        """
        Sets the BPM variable for the events
        """
        while True:
            input_bpm = prompt("Enter the BPM to quantize to: ")
            if input_bpm is None:
                Log.error("BPM not set")
                continue  # Ensures the loop continues if no input is given

            try:
                self.bpm = float(input_bpm)  # Attempt to convert input to float
            except ValueError:
                Log.error("BPM must be a number")
                continue  # If conversion fails, log error and reprompt

            if isinstance(self.bpm, (int, float)):
                Log.info(f"BPM set to {self.bpm}")
                break  # Exit loop if BPM is successfully set as a number
            else:
                Log.error("BPM must be a number")

    def __normalize_event(self, event_time):
        """
        Normalizes the event to the BPM and normalization threshold
        """
        original_time = event_time

        interval = 60 / (self.bpm * self.normalization_threshold)

        event_time = round(event_time / interval) * interval
        Log.info(f"Normalized event from: {original_time}s to {event_time}s")
    
    def process(self, input_data):
        # SendMAEvents may not need to process input data, but implement if necessary
        return input_data
        
    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())
