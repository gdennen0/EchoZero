from Block.block import Block
from message import Log
from Block.BlockTypes.Transform.Parts.Filter import Filter


class TransformAudioBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "TransformAudio"
        self.type = "Transform"
        self.add_part_type(Filter())
        self.add_command("add_part", self.AddPart)
        self.add_command("remove_part", self.RemovePart)
        self.add_command("list_parts", self.ListParts)
        self.add_command("clear_parts", self.ClearParts)
        self.add_command("start", self.start)
        self.add_command("list_part_types", self.list_part_types)

    def start(self, audio):
        result = audio
        for part in self.parts:
            result = part.start(result)
            Log.info(f"TransformAudio Part {part.name} completed")

        return result