from DataTypes.data_type import DataType
class EventItem(DataType):
    def __init__(self):
        super().__init__()
        self.set_name("EventItem")
        self.set_description("An event item")

