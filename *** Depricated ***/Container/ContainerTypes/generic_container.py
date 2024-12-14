from Container.container import Container
from Project.BlockBlockTypes.Analyze.AnalyzeAudioBlock import AnalyzeAudioBlock
from Project.BlockBlockTypes.Export.ExportBlock import ExportBlock
from Project.BlockBlockTypes.Load.LoadAudioBlock import LoadAudioBlock
from Project.BlockBlockTypes.Transform.TransformAudioBlock import TransformAudioBlock 
from Project.BlockBlockTypes.Export.ExportAudioBlock import ExportAudioBlock
from Project.BlockBlockTypes.Export.ExportAudioSpectrogramBlock import ExportAudioSpectrogramBlock
from Project.BlockBlockTypes.Export.ExportMA3Block import ExportMA3Block
class GenericContainer(Container):
    name = "Generic"
    def __init__(self):
        super().__init__()
        self.set_name("generic")

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
