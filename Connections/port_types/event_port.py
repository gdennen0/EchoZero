from .port import Port
from DataTypes.event_data_type import EventData
from tools import Log

class AudioPort(Port):
    name = "AudioPort"

    def __init__(self):
        super().__init__()
        self.data_type = EventData()
        self.set_name("EventData")
