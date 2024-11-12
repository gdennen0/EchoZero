from DataTypes.data_type import DataType
from message import Log
import os
from tools import create_audio_data

class EventData(DataType):
    def __init__(self):
        super().__init__()
        self.items = []
        pass

    def add_item(self, item):
        self.items.append(item)