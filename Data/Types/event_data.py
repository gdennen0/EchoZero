from Data.data import Data

class EventData(Data):
    name = "EventData"
    def __init__(self):
        super().__init__()
        self.items = []
        self.set_name("EventData") # maybe unnecessary now?
        self.description = "A collection of events"
        pass

    def add_item(self, item):
        self.items.append(item)

    def to_dict(self):
        return {"items": [item.to_dict() for item in self.items]}
