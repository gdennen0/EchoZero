from Block.block import Block
from message import Log
class ExportBlock(Block):
    name = "Export"
    # The export block is used to export the analysis results to a file
    def __init__(self):
        super().__init__()
        self.set_name("Export") # maybe unnecessary now?
        self.parts = []

    def add_part(self, part):
        self.parts.append(part)

    def remove_part(self, part):
        self.parts.remove(part)

    def start(self, AnalysisResults):
        Log.info(f"Export Block started")
        if AnalysisResults:
            for result in AnalysisResults:
                Log.info(f"Exporting a result")

