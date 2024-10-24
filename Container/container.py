from Block.BlockTypes.Analyze.AnalyzeAudioBlock import AnalyzeAudioBlock
from Block.BlockTypes.Export.ExportBlock import ExportBlock
from Block.BlockTypes.Load.LoadAudioBlock import LoadAudioBlock
from Block.BlockTypes.Transform.TransformAudioBlock import TransformAudioBlock   


# Generic Container

class Container:
    def __init__(self):
        self.blocks = {}
        self.block_types = {
            "loadAudio": LoadAudioBlock(),
            "transformAudio": TransformAudioBlock(),
            "analyzeAudio": AnalyzeAudioBlock(),
            "export": ExportBlock(),
        }
        self.commands = {}

    def add_block(self, block):
        if hasattr(block, 'name'):
            self.blocks[block.name] = block
        else:
            raise AttributeError("The block does not have a 'name' attribute")
        
    def remove_block(self, block_name):
        if block_name in self.blocks:
            del self.blocks[block_name]
        else:
            raise ValueError(f"Block with name '{block_name}' not found in container")
        
    def list_blocks(self):
        for block_name, block in self.blocks.items():
            print(f"{block_name}: {block.__class__.__name__}")
    

    def start(self):
        audio = self.blocks["load_audio"].start()
        transformed_audio = self.blocks["transform_audio"].start(audio)
        AnalysisResults = self.blocks["analyze_audio"].start(transformed_audio)
        self.blocks["export"].start(AnalysisResults)



            


  