from Project.Data.data import Data

class EventData(Data):
    name = "EventData"
    def __init__(self):
        super().__init__()
        self.name = "EventData" 
        self.type = "EventData"
        self.description = "A collection of events"
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def save(self):
        return {"name":self.name, 
                "description":self.description, 
                "items":[item.save() for item in self.items]
                }
    
    def load(self, data):
        self.name = data["name"]
        self.description = data["description"]
        self.items = data["items"]
        for item in self.items:
            item.load(item) 