from Block.block import Block
from message import Log
from DataTypes.audio_data_type import AudioData
from Block.BlockTypes.Analyze.Parts.OnsetDetection import OnsetDetection
from tools import prompt_selection

class AnalyzeAudioBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "AnalyzeAudio"
        self.parts = [] 
        self.add_input_type(AudioData())
        self.add_output_type(AudioData())
        self.add_part_type(OnsetDetection())

    # def add_part(self, part_name=None):
    #     if part_name:
    #         for part_type in self.part_types:
    #             if part_type.name == part_name:
    #                 self.parts.append(part_type)
    #                 Log.info(f"Added part: {part_type.name}")
    #                 return
    #     else:
    #         part_type, _= prompt_selection("Select Part", self.part_types)
    #         self.parts.append(part_type)
    #         Log.info(f"Added part: {part_type.name}")
    #         return
    #     Log.error(f"Unforturnately no part found")
        

    # def remove_part(self, part):
    #     self.parts.remove(part)

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
    