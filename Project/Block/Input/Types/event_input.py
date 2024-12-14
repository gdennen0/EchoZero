from Project.Block.Input.input import Input

class EventInput(Input):
    name = "EventInput"
    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "EventInput" 
        self.type = "EventInput"
        self.data_type = "EventData"

    def save(self):
        save_data = {
            "name": self.name,
            "type": self.type,
            "data_type": self.data_type,
            "data": self.data.save()
        }   
        return save_data
    
    def load(self, data):
        self.name = data.get("name")
