from src.Project.Data.data import Data

class EventItem(Data):
    name = "EventItem"
    def __init__(self,):
        super().__init__()
        self.name = "EventItem" 
        self.type = "EventItem"
        self.description = "no description set"
        self.time = None
        self.source = None
        self.confidence = None
        self.classification = None

    def save(self):
        return {"name":self.name, 
                "description":self.description,
                "type":self.type,
                "time":self.time,
                "source":self.source
                }
    
    def load(self, data):
        self.name = data["name"]
        self.description = data["description"]
        self.data = data["data"]
        self.time = data["time"]
        self.source = data["source"]

    def set_classification(self, classification):
        self.classification = classification

    def set_confidence(self, confidence):
        self.confidence = confidence
