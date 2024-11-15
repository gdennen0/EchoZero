from Block.block import Block
from message import Log
from Block.BlockTypes.Transform.Parts.GenericFilter import GenericFilter
from DataTypes.audio_data_type import AudioData
from Block.BlockTypes.Transform.Parts.ExtractDrums import ExtractDrums
from Connections.port_types.audio_port import AudioPort

class TransformAudioBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "TransformAudio"
        self.type = "Transform"
        self.add_part_type(GenericFilter())
        self.add_part_type(ExtractDrums())

        self.add_input_type(AudioData())
        self.add_output_type(AudioData())

        self.add_port_type(AudioPort)
        self.add_output_port("AudioPort")
        self.link_port_attribute("output", "AudioPort", "data")
        self.add_input_port("AudioPort")
        self.link_port_attribute("input", "AudioPort", "data")

        self.add_command("clear_parts", self.clear_parts)
        self.add_command("start", self.start)
        self.add_command("list_part_types", self.list_part_types)

    def start(self, data):
        Log.info(f"TransformAudio Block started")
        result = data
        for part in self.parts:
            result = part.start(result)
            Log.info(f"TransformAudio Part {part.name} completed")
            self.set_data(result)

        return result