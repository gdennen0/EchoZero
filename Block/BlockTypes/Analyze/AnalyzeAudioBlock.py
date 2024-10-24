from Block.block import Block
from message import Log

class AnalyzeAudioBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "AnalyzeAudio"
        self.parts = [] 

    def add_part(self, part):
        self.parts.append(part)

    def remove_part(self, part):
        self.parts.remove(part)

    def start(self, audio):
        AnalysisResults = []
        for part in self.parts:
            result = part.start(audio)
            AnalysisResults.append(result)
            Log.info(f"AnalyzeAudio Part {part.name} completed")
        return AnalysisResults
    