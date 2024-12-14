from Project.Blockblock import Block
from Utils.message import Log
from Data.Types.audio_data import AudioData
from Port.PortTypes.audio_port import AudioPort

class TransformAudioBlock(Block):
    name = "TransformAudio"
    def __init__(self):
        super().__init__()
        self.set_name("TransformAudio")  # maybe uncessary now
        self.type = "Transform"

        self.port.add_port_type(AudioPort)
        self.port.add_input("AudioPort")
        self.port.add_output("AudioPort")

