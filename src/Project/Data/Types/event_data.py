from src.Project.Data.data import Data

class EventData(Data):
    name = "EventData"
    def __init__(self):
        super().__init__()
        self.name = "EventData" 
        self.type = "EventData"
        self.description = "A collection of events"
        self.items = []
        self.source = None

    def add_item(self, item):
        self.items.append(item)

    def set_source(self, source):
        self.source = source

    def get_source(self):
        return self.source
    
    def get_all(self):
        return self.items

    def save(self):
        return {"name":self.name, 
                "type":self.type,
                "description":self.description, 
                "items":[item.save() for item in self.items]
                }
    
    def load(self, data):
        self.name = data["name"]
        self.description = data["description"]
        self.items = data["items"]
        for item in self.items:
            item.load(item) 