from Container.container import Container
from Block.BlockTypes.Analyze.AnalyzeAudioBlock import AnalyzeAudioBlock
from Block.BlockTypes.Export.ExportBlock import ExportBlock
from Block.BlockTypes.Load.LoadAudioBlock import LoadAudioBlock
from Block.BlockTypes.Transform.TransformAudioBlock import TransformAudioBlock 
from Block.BlockTypes.Export.export_audio import ExportAudioBlock
from Block.BlockTypes.Export.export_audio_spectrogram import ExportAudioSpectrogramBlock
from Block.BlockTypes.Export.export_ma3 import ExportMA3Block
class GenericContainer(Container):
    def __init__(self):
        super().__init__()
        self.name = "generic"

        self.add_block_type("loadAudio", LoadAudioBlock)
        self.add_block_type("transformAudio", TransformAudioBlock)
        self.add_block_type("analyzeAudio", AnalyzeAudioBlock)
        self.add_block_type("exportAudio", ExportAudioBlock)
        self.add_block_type("exportAudioSpectrogram", ExportAudioSpectrogramBlock)
        self.add_block_type("exportma3", ExportMA3Block)

        self.add_block("loadAudio")
        self.add_block("transformAudio")
        self.add_block("analyzeAudio")
        self.add_block("exportAudio")
        self.add_block("exportAudioSpectrogram")
        self.add_block("exportma3")
