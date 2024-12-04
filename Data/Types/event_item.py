from Data.data import Data

class EventItem(Data):
    name = "EventItem"
    def __init__(self,):
        super().__init__()
        self.set_name("EventItem") # maybe unnecessary now?
        self.set_description("An event item")

    def to_dict(self):
        return {"name":self.name, 
                "description":self.description}
