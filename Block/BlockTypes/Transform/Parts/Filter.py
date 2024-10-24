from message import Log
from Block.part import Part

class Filter(Part):
    def __init__(self):
        super().__init__()
        self.name = "Filter"
        self.block_type = "Transform"
        self.filter_types = ["lowpass", "highpass", "bandpass", "bandstop"]
        self.filter_type = None
        self.add_command("list_filter_types", self.ListFilterTypes)
        self.add_command("set_filter_type", self.SetFilterType)
        self.add_command("start", self.Start)

    def list_filter_types(self):
        return self.filter_types

    def set_filter_type(self, filter_type):
        if filter_type in self.filter_types:
            self.filter_type = filter_type
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")

    def start(self, audio):
        Log.info(f"Filter {self.name} started")
        return audio
