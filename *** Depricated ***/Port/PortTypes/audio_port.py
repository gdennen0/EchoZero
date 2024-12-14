from Port.port import Port
from Data.Types.audio_data import AudioData

class AudioPort(Port):
    name = "AudioPort"
    def __init__(self):
        super().__init__()
        self.attribute.set("name", "AudioPort") # maybe unnecessary now?
        self.attribute.set("data_type", "AudioData")

