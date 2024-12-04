from Port.port import Port
from Data.Types.audio_data import AudioData

class AudioPort(Port):
    name = "AudioPort"
    def __init__(self):
        super().__init__()
        self.data_type = AudioData()
        self.set_name("AudioPort") # maybe unnecessary now?
