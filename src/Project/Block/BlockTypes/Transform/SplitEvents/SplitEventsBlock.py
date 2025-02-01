from src.Project.Block.block import Block
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Data.Types.event_data import EventData

from src.Utils.message import Log
from src.Utils.tools import prompt_selection, prompt

class SplitEventsBlock(Block):
    """
    SplitEventsBlock is a block that takes eventdata objects and splits them into other/multiple eventdata objects.
    """

    name = "SplitEvents"
    type = "SplitEvents"
    
    def __init__(self):
        super().__init__()
        self.name = "SplitEvents"
        self.type = "SplitEvents"

        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.command.add("set_split_type", self.set_split_type)

        self.split_type_options = ["classification",]
        self.split_type = None


    def set_split_type(self, split_type=None):
        if split_type:
            self.split_type = split_type
        else:
            self.split_type = prompt_selection("Enter the split type: ", self.split_type_options)
        Log.info(f"Split type set to {self.split_type}")


    def process(self, input_data):
        # Initialize as dictionary instead of list
        classification_options = {}
        
        if self.split_type == "classification":
            # Collect events by classification
            for old_event_data_item in input_data:
                for event in old_event_data_item.get_all():
                    if event.get_classification() not in classification_options:
                        classification_options[event.get_classification()] = []
                    classification_options[event.get_classification()].append(event)

            # Create new event data items for each classification
            new_event_data_items = []
            for classification in classification_options:
                event_data = EventData()
                event_data.set_name(f"{classification}Events")
                event_data.set_type("EventData")
                for event in classification_options[classification]:
                    event_data.add_item(event)
                new_event_data_items.append(event_data)

            return new_event_data_items
        else:
            Log.error(f"Invalid split type: {self.split_type}")
            return input_data
        
    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "split_type": self.split_type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        if block_metadata.get("split_type"):
            self.set_split_type(split_type=block_metadata.get("split_type"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())
