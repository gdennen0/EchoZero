from Block.block import Block
from message import Log
from DataTypes.audio_data_type import AudioData

class AnalyzeAudioBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "AnalyzeAudio"
        self.parts = [] 
        self.add_input_type(AudioData)
        self.add_output_type(AudioData)

    def add_part(self, part):
        self.parts.append(part)

    def remove_part(self, part):
        self.parts.remove(part)

    def start(self, audio):
        Log.info(f"AnalyzeAudioBlock started")
        if not isinstance(audio, AudioData):
            Log.error(f"Input is not an AudioData object")
            return
        
        AnalysisResults = []
        for part in self.parts:
            result = part.start(audio)
            AnalysisResults.append(result)
            Log.info(f"AnalyzeAudio Part {part.name} completed")
        return AnalysisResults
    