from Project.Blockblock import Block
from Utils.message import Log
from Data.Types.audio_data import AudioData

class AnalyzeAudioBlock(Block):
    name = "AnalyzeAudio"
    def __init__(self):
        super().__init__()
        self.set_name("AnalyzeAudio")  # maybe unnecessary now?

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
    