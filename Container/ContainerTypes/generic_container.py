from Container.container import Container
from Block.BlockTypes.Analyze.AnalyzeAudioBlock import AnalyzeAudioBlock
from Block.BlockTypes.Export.ExportBlock import ExportBlock
from Block.BlockTypes.Load.LoadAudioBlock import LoadAudioBlock
from Block.BlockTypes.Transform.TransformAudioBlock import TransformAudioBlock 

class GenericContainer(Container):
    def __init__(self):
        super().__init__()
        self.name = "generic"

        self.add_block_type("loadAudio", LoadAudioBlock)
        self.add_block_type("transformAudio", TransformAudioBlock)
        self.add_block_type("analyzeAudio", AnalyzeAudioBlock)
        self.add_block_type("export", ExportBlock)

        self.add_block("loadAudio")
        self.add_block("transformAudio")
        self.add_block("analyzeAudio")
        self.add_block("export")

    def start(self):
        audio = self.blocks["loadAudio"].start()
        transformed_audio = self.blocks["transformAudio"].start(audio)
        AnalysisResults = self.blocks["analyzeAudio"].start(transformed_audio)
        self.blocks["export"].start(AnalysisResults)

