from src.Project.Block.Input.input import Input

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
            "data": self.data.save(),
        }   
        # Add connected output info only if it exists
        if self.connected_output:
            try:
                save_data["connected_output"] = f"{self.connected_output.parent_block.name}.output.{self.connected_output.name}"
            except AttributeError:
                save_data["connected_output"] = None
        else:
            save_data["connected_output"] = None
            
        return save_data
    
    def load(self, data):
        self.name = data.get("name")
        self.type = data.get("type")
        self.data_type = data.get("data_type")
        connected_output = data.get("connected_output")
        if connected_output:
            connected_block_name = connected_output.split('.')[0]
            connected_block = self.parent_block.parent.get_block(connected_block_name)
            output_name = data.get("connected_output").split('.')[2]
            connected_output = connected_block.output.get(output_name)
            self.connect(connected_output)