from .port import Port
from DataTypes.audio_data_type import AudioData
from tools import Log

class AudioPort(Port):
    name = "AudioPort"

    def __init__(self):
        super().__init__()
        self.data_type = AudioData()
        self.set_name("AudioPort")