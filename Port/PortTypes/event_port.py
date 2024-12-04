from ..port import Port
from Data.Types.event_data import EventData

class EventPort(Port):
    name = "EventPort"
    def __init__(self):
        super().__init__()
        self.data_type = EventData()
        self.set_name("EventData") # maybe unnecessary now?
