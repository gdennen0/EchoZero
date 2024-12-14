from ..port import Port
from Data.Types.event_data import EventData

class EventPort(Port):
    name = "EventPort"
    def __init__(self):
        super().__init__()
        self.attribute.set("name", "EventData") # maybe unnecessary now?
        self.attribute.set("data_type", "EventData")
